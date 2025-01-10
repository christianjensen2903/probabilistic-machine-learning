# This UNET-style prediction model was originally included as part of the Score-based generative modelling tutorial
# by Yang Song et al: https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from util import GaussianFourierProjection, Dense, set_all_seeds
from torchvision import datasets, transforms, utils  # type: ignore
from tqdm.auto import tqdm  # type: ignore
import matplotlib.pyplot as plt
import math


class Encoder(nn.Module):
    def __init__(self, num_classes, channels=[32, 64, 128, 256], embed_dim=256):
        """Initialize a time-dependent score-based network.

        Args:
        marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
        channels: The number of channels for feature maps of each resolution.
        embed_dim: The dimensionality of Gaussian random feature embeddings.
        """
        super().__init__()
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )
        # Encoding layers where the resolution decreases
        self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])
        self.linear = nn.Linear(1024, num_classes)

        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, x, t):
        embed = self.act(self.embed(t))
        # Encoding path
        h1 = self.conv1(x)
        ## Incorporate information from t
        h1 += self.dense1(embed)
        ## Group normalization
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)
        h2 = self.conv2(h1)
        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)
        h3 = self.conv3(h2)
        h3 += self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)
        h4 = self.conv4(h3)
        h4 += self.dense4(embed)
        # shape h4 : [256,256,2,2]
        h4 = self.gnorm4(h4).reshape(x.shape[0], -1)
        # [256, 256*2*2]
        h4 = self.act(h4)
        return self.linear(h4)  # [256, 10]


class Classifier(nn.Module):
    def __init__(self, network, T=100, beta_1=1e-4, beta_T=2e-2):
        """Initialize a time-dependent score-based network.

        Args:
        marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
        channels: The number of channels for feature maps of each resolution.
        embed_dim: The dimensionality of Gaussian random feature embeddings.
        """
        super().__init__()

        # Normalize time input before evaluating neural network
        # Reshape input into image format and normalize time value before sending it to network model
        self._network = network
        self.network = lambda x, t: (
            self._network(x.reshape(-1, 1, 28, 28), (t.squeeze() / T))
        )

        # Total number of time steps
        self.T = T

        # Registering as buffers to ensure they get transferred to the GPU automatically
        self.register_buffer("beta", torch.linspace(beta_1, beta_T, T + 1))
        self.register_buffer("alpha", 1 - self.beta)
        self.register_buffer("alpha_bar", self.alpha.cumprod(dim=0))

    def forward_diffusion(self, x0, t, epsilon):
        """
        q(x_t | x_0)
        Forward diffusion from an input datapoint x0 to an xt at timestep t, provided a N(0,1) noise sample epsilon.
        Note that we can do this operation in a single step

        Parameters
        ----------
        x0: torch.tensor
            x value at t=0 (an input image)
        t: int
            step index
        epsilon:
            noise sample

        Returns
        -------
        torch.tensor
            image at timestep t
        """

        mean = torch.sqrt(self.alpha_bar[t]) * x0
        std = torch.sqrt(1 - self.alpha_bar[t])

        return mean + std * epsilon

    def loss(self, x0, c):

        t = torch.randint(1, self.T, (x0.shape[0], 1)).to(x0.device)

        # Sample noise
        epsilon = torch.randn_like(x0)

        xt = self.forward_diffusion(x0, t, epsilon)

        return nn.CrossEntropyLoss()(self.network(xt, t), c)


def evaluate(model, dataloader, device):
    """Evaluate the model on the given dataset."""
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for x, c in dataloader:
            x, c = x.to(device), c.to(device)

            t = torch.randint(1, model.T, (x.shape[0], 1)).to(device)
            epsilon = torch.randn_like(x)
            xt = model.forward_diffusion(x, t, epsilon)

            predictions = model.network(xt, t).argmax(dim=1)  # Get predicted class
            correct += (predictions == c).sum().item()
            total += c.size(0)

    accuracy = correct / total
    return accuracy


if __name__ == "__main__":
    set_all_seeds(42)
    # Parameters
    T = 1000
    learning_rate = 1e-3
    epochs = 100
    batch_size = 256
    num_classes = 10
    save_path = "./mnist_unet_model.pth"

    # Rather than treating MNIST images as discrete objects, as done in Ho et al 2020,
    # we here treat them as continuous input data, by dequantizing the pixel values (adding noise to the input data)
    # Also note that we map the 0..255 pixel values to [-1, 1], and that we process the 28x28 pixel values as a flattened 784 tensor.
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: x + torch.rand(x.shape) / 255
            ),  # Dequantize pixel values
            transforms.Lambda(lambda x: (x - 0.5) * 2.0),  # Map from [0,1] -> [-1, -1]
            transforms.Lambda(lambda x: x.flatten()),
        ]
    )

    # Download and transform train dataset
    dataloader_train = torch.utils.data.DataLoader(
        datasets.MNIST("./mnist_data", download=True, train=True, transform=transform),
        batch_size=batch_size,
        shuffle=True,
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: x + torch.rand(x.shape) / 255
            ),  # Dequantize pixel values
            transforms.Lambda(lambda x: (x - 0.5) * 2.0),  # Map from [0,1] -> [-1,1]
            transforms.Lambda(lambda x: x.flatten()),
        ]
    )

    # Download and transform test dataset
    dataloader_test = torch.utils.data.DataLoader(
        datasets.MNIST(
            "./mnist_data", download=True, train=False, transform=test_transform
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    # Select device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Construct Unet
    # The original ScoreNet expects a function with std for all the
    # different noise levels, such that the output can be rescaled.
    # Since we are predicting the noise (rather than the score), we
    # ignore this rescaling and just set std=1 for all t.

    # model = Classifier(num_classes)

    encoder = Encoder(num_classes=num_classes)
    model = Classifier(encoder).to(device)

    # Construct optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Setup simple scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9999)

    def reporter(model):
        """Callback function used for plotting images during training"""

        # Switch to eval mode
        model.eval()

        with torch.no_grad():
            nsamples = 10
            samples = model.sample((nsamples, 28 * 28)).cpu()

            # Map pixel values back from [-1,1] to [0,1]
            samples = (samples + 1) / 2
            samples = samples.clamp(0.0, 1.0)

            # Plot in grid
            grid = utils.make_grid(samples.reshape(-1, 1, 28, 28), nrow=nsamples)
            plt.gca().set_axis_off()
            plt.imshow(transforms.functional.to_pil_image(grid), cmap="gray")
            plt.show()

    total_steps = len(dataloader_train) * epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    best_accuracy = 0.0
    for epoch in range(epochs):

        # Switch to train mode
        model.train()

        for i, (x, c) in enumerate(dataloader_train):
            x, c = x.to(device), c.to(device)
            optimizer.zero_grad()

            loss = model.loss(x, c)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Update progress bar
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                epoch=f"{epoch + 1}/{epochs}",
                lr=f"{scheduler.get_last_lr()[0]:.2E}",
            )
            progress_bar.update()

        # Evaluate the model after each epoch
        train_accuracy = evaluate(model, dataloader_train, device)
        test_accuracy = evaluate(model, dataloader_test, device)
        print(
            f"Epoch {epoch + 1}/{epochs} - Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}"
        )

        # Save the model if it achieves a new best accuracy
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(encoder.state_dict(), save_path)
            print(
                f"New best accuracy: {best_accuracy:.4f}. Model saved to {save_path}."
            )

    print("Training complete.")
    print(f"Best Test Accuracy: {best_accuracy:.4f}")
