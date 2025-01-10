from torcheval import metrics
from torchmetrics.image.inception import InceptionScore
import torch
from torchvision import datasets, transforms
from baseline_model import ScoreNet as baseline_model_obj, DDPM
from classifier_free_guided import ScoreNet as classifier_free_model_obj, ClassifierFreeDDPM
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def calculate_fid(
    pred: torch.Tensor,
    target: torch.Tensor,
    device: str,
    batch_size: int = 32,
) -> float:
    #print(pred)
    #print("len_pred",len(pred))
    #print("len(pred[0].shape)", len(pred[0].shape))
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

    return fid_calculator.compute()

def calc_is(pred, device):
    pred = (pred.clone().repeat(1, 3, 1, 1)*255).to(torch.uint8).to(device)
    is_calculator = InceptionScore()
    is_calculator = is_calculator.to(device)

    is_calculator.update(pred)

    score, _ = is_calculator.compute()

    return score


T = 1000
n_samples = 2500
batch_size = 250
num_classes = 10


# Load baseline classifier
baseline_obj = baseline_model_obj((lambda t: torch.ones(1).to(device)))
model_baseline = DDPM(baseline_obj, T=T).to(device)
model_baseline.load_state_dict(torch.load("baseline_model.pth"))

# Load model classifier free model
classifier_free_obj = classifier_free_model_obj((lambda t: torch.ones(1).to(device)),num_classes=num_classes)
model_classifier_free = ClassifierFreeDDPM(classifier_free_obj, T=T).to(device)
model_classifier_free.load_state_dict(torch.load("classifier_free_model.pth"), strict=False)


dataset = datasets.MNIST('./mnist_data', download=True, train=False)
real_samples = dataset.data.unsqueeze(1).float() / 255


ws = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4]
result_dict = {w : {"FID": 0, "IS": 0} for w in ws} | {"baseline" : {"FID": 0, "IS":0}}

print("Evaluating Classifier Free Model")
for w in ws:
    
    #class_labels = torch.arange(num_classes).repeat_interleave(n_samples // num_classes)
    #print(class_labels)
    #print(class_labels.shape)
    samples = []
    pbar = tqdm(total=n_samples, desc="Generating Samples")
    while len(samples) < n_samples:
        c = torch.arange(num_classes).repeat_interleave(batch_size // num_classes)
        #print(c.shape)
        #print(len(c))
        #print(c)
        new_samples = model_classifier_free.sample((batch_size,28*28), w=w, c=c.to(device)).cpu()
        #print("new_samples_shape", new_samples.shape)
        new_samples = (new_samples+1)/2 
        new_samples = new_samples.clamp(0.0, 1.0)
        #samples.extend(new_samples) sådan var det før
        samples.extend(new_samples.reshape(-1, 1, 28, 28))
        pbar.update(len(new_samples))
        
    samples = samples[:n_samples]
    pbar.close()
    
    samples = torch.stack(samples) #added FAK
    #print(samples.shape,"fak-look-here")

    fid = calculate_fid(samples, real_samples, device)
    inception_score = calc_is(samples, device)

    print(f"{w}: FID={fid}, IS={inception_score}")      
    result_dict[w]["FID"] = fid.item()
    result_dict[w]["IS"] = inception_score.item()

print("\n" + "-" * 50 + "\n")
print("Evaluating Baseline")

samples = []
pbar = tqdm(total=n_samples, desc="Generating Samples")
while len(samples) < n_samples:
    new_samples = model_baseline.sample((batch_size,28*28)).cpu()
    #print("new_samples_shape", new_samples.shape)
    new_samples = (new_samples+1)/2 
    new_samples = new_samples.clamp(0.0, 1.0)
    samples.extend(new_samples.reshape(-1, 1, 28, 28))
    pbar.update(len(new_samples))

samples = samples[:n_samples]
pbar.close()

samples = torch.stack(samples)

#print(samples.shape)

fid = calculate_fid(samples, real_samples, device)
inception_score = calc_is(samples, device)
print(f"FID={fid}, IS={inception_score}") 

result_dict["baseline"]["FID"] = fid.item()
result_dict["baseline"]["IS"] = inception_score.item()


print(result_dict)

import json

# Save dictionary to file
with open('results.json', 'w') as f:
    json.dump(result_dict, f)