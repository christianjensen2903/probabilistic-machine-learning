from torcheval import metrics
import torch
from torchvision import datasets  # type: ignore
from baseline_model import ScoreNet as baseline_model_obj, DDPM
from classifier_free_guided import (
    ScoreNet as classifier_free_model_obj,
    ClassifierFreeDDPM,
)
from tqdm.auto import tqdm  # type: ignore
import numpy as np
from torch import nn
import torch.nn.functional as F
from classifier import Encoder
from typing import Any

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def calculate_fid(
    pred: torch.Tensor,
    target: torch.Tensor,
    device: str,
    batch_size: int = 32,
) -> float:
    pred = pred.clone().repeat(1, 3, 1, 1)
    target = target.clone().repeat(1, 3, 1, 1)

    fid_calculator = metrics.FrechetInceptionDistance()
    fid_calculator = fid_calculator.to(device)

    for i in range(0, target.size(0), batch_size):
        end_i = min(i + batch_size, target.size(0))
        target_batch = target[i:end_i]
        fid_calculator.update(target_batch, is_real=True)

    for i in range(0, pred.size(0), batch_size):
        end_i = min(i + batch_size, pred.size(0))
        pred_batch = pred[i:end_i]
        fid_calculator.update(pred_batch, is_real=False)

    return fid_calculator.compute().item()


def calc_is(pred: torch.Tensor, classifier: nn.Module, device: str, eps=1e-16) -> float:
    logits = (
        classifier(pred.to(device), torch.ones(pred.size(0)).to(device)).cpu().detach()
    )
    p_yx = F.softmax(logits, dim=-1).numpy()

    # calculate p(y)
    p_y = np.expand_dims(p_yx.mean(axis=0), 0)

    # kl divergence for each image
    kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
    # sum over classes
    sum_kl_d = kl_d.sum(axis=1)
    # average over images
    avg_kl_d = sum_kl_d.mean()
    # undo the logs
    is_score = np.exp(avg_kl_d)
    return float(is_score)


"""def calc_is(pred, device):
    pred = (pred.clone().repeat(1, 3, 1, 1)*255).to(torch.uint8).to(device)
    is_calculator = InceptionScore()
    is_calculator = is_calculator.to(device)

    is_calculator.update(pred)

    score, _ = is_calculator.compute()

    return score"""


T = 1000
n_samples = 1000
batch_size = 250
num_classes = 10


# Load baseline classifier
baseline_obj = baseline_model_obj((lambda t: torch.ones(1).to(device)))
baseline = DDPM(baseline_obj, T=T).to(device)
baseline.load_state_dict(torch.load("baseline_model.pth"))

# Load model classifier free model
classifier_free_obj = classifier_free_model_obj(
    (lambda t: torch.ones(1).to(device)), num_classes=num_classes
)
model = ClassifierFreeDDPM(classifier_free_obj, T=T).to(device)
model.load_state_dict(torch.load("classifier_free_model.pth"), strict=False)


inception_model = Encoder(num_classes=num_classes)
inception_model.load_state_dict(torch.load("./mnist_unet_no_noise.pth"))


dataset = datasets.MNIST("./mnist_data", download=True, train=False)
real_samples = dataset.data.unsqueeze(1).float() / 255


ws = [1.1, 1.2, 1.5, 2, 3, 4]
result_dict: dict[str, Any] = {}

print("Evaluating Classifier Free Model")
result_dict["classifier_free"] = {}
for w in ws:

    result_dict["classifier_free"][w] = {}

    samples: list[torch.Tensor] = []
    pbar = tqdm(total=n_samples, desc="Generating Samples")
    while len(samples) < n_samples:
        c = torch.arange(num_classes).repeat_interleave(batch_size // num_classes)

        new_samples = model.sample((batch_size, 28 * 28), w=w, c=c.to(device)).cpu()
        new_samples = (new_samples + 1) / 2
        new_samples = new_samples.clamp(0.0, 1.0)
        samples.extend(new_samples.reshape(-1, 1, 28, 28))
        pbar.update(len(new_samples))

    samples = samples[:n_samples]
    pbar.close()

    samples_tensor = torch.stack(samples)

    fid = calculate_fid(samples_tensor, real_samples, device)
    inception_score = calc_is(samples_tensor, inception_model, device)

    print(f"{w}: FID={fid}, IS={inception_score}")
    result_dict["classifier_free"][w]["FID"] = fid
    result_dict["classifier_free"][w]["IS"] = inception_score


print("\n" + "-" * 50 + "\n")
print("Evaluating Classifier Guided Model")
result_dict["classifier_guided"] = {}


classifier = Encoder(num_classes)
classifier.load_state_dict(torch.load("./mnist_unet_model.pth"))
classifier = classifier.to(device)

for w in ws:

    result_dict["classifier_guided"][w] = {}
    samples = []
    pbar = tqdm(total=n_samples, desc="Generating Samples")
    while len(samples) < n_samples:
        c = torch.arange(num_classes).repeat_interleave(batch_size // num_classes)

        new_samples = model.sample(
            (batch_size, 28 * 28), w=w, c=c.to(device), classifier=classifier
        ).cpu()
        new_samples = (new_samples + 1) / 2
        new_samples = new_samples.clamp(0.0, 1.0)
        samples.extend(new_samples.reshape(-1, 1, 28, 28))
        pbar.update(len(new_samples))

    samples = samples[:n_samples]
    pbar.close()

    samples_tensor = torch.stack(samples)

    fid = calculate_fid(samples_tensor, real_samples, device)
    inception_score = calc_is(samples_tensor, inception_model, device)

    print(f"{w}: FID={fid}, IS={inception_score}")
    result_dict["classifier_guided"][w]["FID"] = fid
    result_dict["classifier_guided"][w]["IS"] = inception_score

print("\n" + "-" * 50 + "\n")
print("Evaluating Baseline")
result_dict["baseline"] = {}
samples = []
pbar = tqdm(total=n_samples, desc="Generating Samples")
while len(samples) < n_samples:
    new_samples = baseline.sample((batch_size, 28 * 28)).cpu()
    new_samples = (new_samples + 1) / 2
    new_samples = new_samples.clamp(0.0, 1.0)
    samples.extend(new_samples.reshape(-1, 1, 28, 28))
    pbar.update(len(new_samples))

samples = samples[:n_samples]
pbar.close()

samples_tensor = torch.stack(samples)

fid = calculate_fid(samples_tensor, real_samples, device)
inception_score = calc_is(samples_tensor, inception_model, device)
print(f"FID={fid}, IS={inception_score}")

result_dict["baseline"]["FID"] = fid
result_dict["baseline"]["IS"] = inception_score


print(result_dict)

import json

# Save dictionary to file
with open("results.json", "w") as f:
    json.dump(result_dict, f)
