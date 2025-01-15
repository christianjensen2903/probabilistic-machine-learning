import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms, utils  # type: ignore
from tqdm.auto import tqdm  # type: ignore
import matplotlib.pyplot as plt
import math
import random
from util import (
    GaussianFourierProjection,
    Dense,
    set_all_seeds,
    ExponentialMovingAverage,
)
from classifier import Encoder


class ScoreNet(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(
        self,
        marginal_prob_std,
        channels=[32, 64, 128, 256],
        embed_dim=256,
        num_classes=None,
    ):
        """Initialize a time-dependent score-based network.

        Args:
          marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
          channels: The number of channels for feature maps of each resolution.
          embed_dim: The dimensionality of Gaussian random feature embeddings.
        """
        super().__init__()

        self.num_classes = num_classes
        # Gaussian random feature embedding layer for time
        self.t_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

        if self.num_classes is not None:
            self.c_embed = nn.Embedding(num_classes, embed_dim)

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

        # Decoding layers where the resolution increases
        self.tconv4 = nn.ConvTranspose2d(
            channels[3], channels[2], 3, stride=2, bias=False
        )
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        self.tconv3 = nn.ConvTranspose2d(
            channels[2] + channels[2],
            channels[1],
            3,
            stride=2,
            bias=False,
            output_padding=1,
        )
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.tconv2 = nn.ConvTranspose2d(
            channels[1] + channels[1],
            channels[0],
            3,
            stride=2,
            bias=False,
            output_padding=1,
        )
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, stride=1)

        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t, c=None):
        # Obtain the Gaussian random feature embedding for t
        embed = self.act(self.t_embed(t))

        if self.num_classes is not None and c is not None:
            embed = embed + self.c_embed(c)

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
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)

        # Decoding path
        h = self.tconv4(h4)
        ## Skip connection from the encoding path
        h += self.dense5(embed)
        h = self.tgnorm4(h)
        h = self.act(h)
        h = self.tconv3(torch.cat([h, h3], dim=1))
        h += self.dense6(embed)
        h = self.tgnorm3(h)
        h = self.act(h)
        h = self.tconv2(torch.cat([h, h2], dim=1))
        h += self.dense7(embed)
        h = self.tgnorm2(h)
        h = self.act(h)
        h = self.tconv1(torch.cat([h, h1], dim=1))

        # Normalize output
        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h


class ClassifierFreeDDPM(nn.Module):

    def __init__(
        self,
        network: nn.Module,
        T: int = 100,
        beta_1: float = 1e-4,
        beta_T: float = 2e-2,
        drop_prob: float = 0.1,
    ):

        super(ClassifierFreeDDPM, self).__init__()

        self._network = network
        self.network = lambda x, t, c: (
            self._network(x.reshape(-1, 1, 28, 28), (t.squeeze() / T), c)
        ).reshape(-1, 28 * 28)

        # Total number of time steps
        self.T = T
        self.drop_prob = drop_prob

        # Registering as buffers to ensure they get transferred to the GPU automatically
        self.register_buffer("beta", torch.linspace(beta_1, beta_T, T + 1))
        self.register_buffer("alpha", 1 - self.beta)
        self.register_buffer("alpha_bar", self.alpha.cumprod(dim=0))

    def forward(self, x0: torch.Tensor, c: torch.Tensor | None) -> torch.Tensor:

        t = torch.randint(1, self.T, (x0.shape[0], 1)).to(x0.device)

        epsilon = torch.randn_like(x0)

        mean = torch.sqrt(self.alpha_bar[t]) * x0
        std = torch.sqrt(1 - self.alpha_bar[t])
        xt = mean + std * epsilon

        if random.random() < self.drop_prob:
            c = None

        return nn.MSELoss()(epsilon, self.network(xt, t, c))

    def reverse_diffusion(
        self, xt: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, eps: torch.Tensor
    ) -> torch.Tensor:

        mean = (
            1.0
            / torch.sqrt(self.alpha[t])
            * (xt - (self.beta[t]) / torch.sqrt(1 - self.alpha_bar[t]) * eps)
        )
        std = torch.where(
            t > 0,
            torch.sqrt(
                ((1 - self.alpha_bar[t - 1]) / (1 - self.alpha_bar[t])) * self.beta[t]
            ),
            0,
        )

        return mean + std * noise

    def get_gradient(
        self, xt: torch.Tensor, t: torch.Tensor, c: torch.Tensor, classifier: nn.Module
    ) -> torch.Tensor:
        with torch.enable_grad():
            x_in = xt.clone().requires_grad_(True)

            # Get classifier predictions
            logits = classifier(x_in.reshape(-1, 1, 28, 28), t.squeeze() / self.T)
            log_probs = F.log_softmax(logits, dim=-1)

            # Select log probability for target class
            selected_logits = log_probs[range(len(c)), c]

            gradient = torch.autograd.grad(selected_logits.sum(), x_in)[0]

        return gradient

    @torch.no_grad()
    def sample(
        self,
        shape: tuple,
        w: float,
        c: torch.Tensor,
        classifier: nn.Module | None = None,
    ) -> torch.Tensor:
        xT = torch.randn(shape).to(self.beta.device)

        xt = xT
        for i in range(self.T, 0, -1):
            # Convert 0 to a tensor of zeros when i=1
            noise = torch.randn_like(xT) if i > 1 else torch.zeros_like(xT)
            t = torch.tensor(i).expand(xt.shape[0], 1).to(self.beta.device)

            if classifier is not None:
                eps = self.network(xt, t, c) - w * self.get_gradient(
                    xt, t, c, classifier
                ) * torch.sqrt(1 - self.alpha_bar[t])
            else:
                # eps = (1 + w) * self.network(xt, t, c) - w * self.network(xt, t, None)
                eps = w * self.network(xt, t, c) - (w - 1) * self.network(
                    xt, t, None
                )  # changed 14-01-2025

            xt = self.reverse_diffusion(xt, t, noise, eps)

        return xt


def train(
    model,
    optimizer,
    scheduler,
    dataloader,
    epochs,
    device,
    ema=True,
    per_epoch_callback=None,
):

    set_all_seeds(42)

    total_steps = len(dataloader) * epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    if ema:
        ema_global_step_counter = 0
        ema_steps = 10
        ema_adjust = dataloader.batch_size * ema_steps / epochs
        ema_decay = 1.0 - 0.995
        ema_alpha = min(1.0, (1.0 - ema_decay) * ema_adjust)
        ema_model = ExponentialMovingAverage(
            model, device=device, decay=1.0 - ema_alpha
        )

    for epoch in range(epochs):

        # Switch to train mode
        model.train()

        global_step_counter = 0
        for i, (x, c) in enumerate(dataloader):
            x = x.to(device)
            c = c.to(device)
            optimizer.zero_grad()
            loss = model.forward(x, c)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Update progress bar
            progress_bar.set_postfix(
                loss=f"⠀{loss.item():12.4f}",
                epoch=f"{epoch+1}/{epochs}",
                lr=f"{scheduler.get_last_lr()[0]:.2E}",
            )
            progress_bar.update()

            if ema:
                ema_global_step_counter += 1
                if ema_global_step_counter % ema_steps == 0:
                    ema_model.update_parameters(model)

        if per_epoch_callback:
            per_epoch_callback(ema_model.module if ema else model)


if __name__ == "__main__":
    set_all_seeds(42)

    # Parameters
    T = 1000
    learning_rate = 1e-3
    epochs = 100
    batch_size = 256
    num_classes = 10

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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mnist_unet = ScoreNet((lambda t: torch.ones(1).to(device)), num_classes=num_classes)

    model = ClassifierFreeDDPM(mnist_unet, T=T).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9999)

    classifier = Encoder(num_classes)

    classifier.load_state_dict(torch.load("./mnist_unet_model.pth"))

    classifier = classifier.to(device)

    def reporter(model):
        """Callback function used for plotting images during training"""

        model.eval()

        with torch.no_grad():
            nsamples = 10
            class_labels = torch.arange(num_classes).repeat_interleave(
                nsamples // num_classes
            )

            if len(class_labels) < nsamples:
                extra_labels = torch.arange(nsamples % num_classes)
                class_labels = torch.cat((class_labels, extra_labels))

            samples = model.sample(
                (nsamples, 28 * 28),
                w=1,
                c=class_labels.to(device),
                classifier=classifier,
            ).cpu()

            samples = (samples + 1) / 2
            samples = samples.clamp(0.0, 1.0)

            grid = utils.make_grid(samples.reshape(-1, 1, 28, 28), nrow=nsamples)
            plt.gca().set_axis_off()
            plt.imshow(transforms.functional.to_pil_image(grid), cmap="gray")
            plt.savefig("samples.png")

    train(
        model,
        optimizer,
        scheduler,
        dataloader_train,
        epochs=epochs,
        device=device,
        ema=True,
        per_epoch_callback=reporter,
    )

    torch.save(model.state_dict(), "classifier_free_model.pth")
