# general 
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from PIL import Image
import numpy as np
import requests
import random

# torch & hugging face
import clip
import torch
from torchvision import models, transforms as T
from torchvision.transforms import ToPILImage
to_pil = ToPILImage()

# module
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# custom
from task0_dataset.dataloader import get_data_loader

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# random seed
random.seed(42)


# mode (zs inference, linear probe)
MODE = "probe"
if MODE == "inference": b_sz = 1
else: b_sz = 128


# ----------------------------------------------------------------
# data config 
# ----------------------------------------------------------------

print("")
data_tf = T.Compose([T.ToTensor()])  
config = {
    'test_data_root': 'task1_classification/data/test',
    'easy_test':  1000,
    'med_test':   1000, 
    'hard_test':  1000, 
    'batch_size': b_sz, 
}
test_easy_loader, cls_to_idx = get_data_loader(
    # data
    root = config.get('test_data_root',"task1_classification/data/train"),
    # select from
    difficulties = ["easy"],
    lengths      = [3, 4, 5, 6],
    # how many
    total_images = config.get('easy_test', 1_000),
    shots        = None,    
    # misc
    transform    = data_tf,
    seed         = 123,
    batch_size   = config.get('batch_size', 1),
    validate     = True
)
test_medium_loader, _ = get_data_loader(
    # data
    root = config.get('test_data_root',"task1_classification/data/test"),
    # select from
    difficulties = ["medium"],
    lengths      = [3, 4, 5, 6],
    # how many
    total_images = config.get('med_test', 1_000), 
    shots        = None,      
    # misc
    transform    = data_tf,
    seed         = 123,
    batch_size   = config.get('batch_size', 1),
    validate     = True
)
test_hard_loader, _ = get_data_loader(
    # data
    root = config.get('test_data_root',"task1_classification/data/test"),
    # select from
    difficulties = ["hard"],
    lengths      = [3, 4, 5, 6],
    # how many
    total_images = config.get('hard_test', 1_000), 
    shots        = None,      
    # misc
    transform    = data_tf,
    seed         = 123,
    batch_size   = config.get('batch_size', 1),
    validate     = True
)
dataloaders = {
    "easy": test_easy_loader, 
    "medium": test_medium_loader, 
    "hard":test_hard_loader
}
print("")


# ----------------------------------------------------------------
# Helper 
# ----------------------------------------------------------------

def get_features(model, dataloader):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            features = model.encode_image(images.to(device))
            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()
        

# ----------------------------------------------------------------
# CLIP
# ----------------------------------------------------------------

# model
model, preprocess = clip.load("ViT-B/32", device=device)


# Encode with processor
if MODE == "inference":
    # prepare text input to CLIP
    txt_list = []
    for k in cls_to_idx: 
        # txt_list.append(f'The word "{k} in the image"')
        txt_list.append(f'The word "{k}"')
        # txt_list.append(f'"{k}"')
    print(f"Length of Text Input: {len(txt_list)}")
    texts = clip.tokenize(txt_list).to(device)

    # easy, medium, hard one at a time
    for difficulty, dataloader in dataloaders.items():
        print(f"\nTesting: {difficulty}")
        count = 0
        corrects = 0
        total_samples = 0
        for inputs, labels in tqdm(dataloader, desc="Processing Images", unit="imgs"):
            # get images
            inputs, labels = inputs.to(device), labels.to(device)
            total_samples += labels.size(0)
            
            # input tensors -> PIL -> CLIP
            inputs = inputs.squeeze(0).cpu()
            pil_images = to_pil(inputs)
            images = preprocess(pil_images).unsqueeze(0).to(device)
            
            # inference
            with torch.no_grad():
                # single image and single text (bz must be 1)                
                logits_per_image, logits_per_text = model(images, texts)
                probs = logits_per_image.softmax(dim=-1)

                # preds
                # print(f"Probabilities: {probs.shape}")
                # print(f"Probabilities: {probs.shape}")
                # print(f"Max prob: {probs.max()}")
                # print(f"Mean prob: {probs.mean()}")
                
            # accuracy
            _, preds = torch.max(probs, 1)
            corrects += torch.sum(preds == labels)
            
        print(f"Accuracy ({difficulty}): {corrects.double()/total_samples*100}%\n")
        
        
# linear probe
elif MODE == "probe":
    # NOTE 
    # each difficulty (easy, med, hard) has 5000 samples 
    # vary train (easy+med) samples across (total): 1_000, 3_000, 6_000, 10_000
    
    # train loader
    train_data_list = [10_000] #[1000, 3000, 6000, 10_000]
    for count in train_data_list: 
        print("---------------------------------------------------")
        # amount of training data (easy+med)
        config['imgs_train'] = count
        train_loader, _ = get_data_loader(
            # data
            root = config.get('train_data_root',"task1_classification/data/train"),
            # select from
            difficulties = ["medium", "hard"],
            lengths      = [3, 4, 5, 6],
            # how many
            total_images = config.get('imgs_train', 1_000),   # overrides shots
            shots        = None,                              # if total_images = None 
            # misc
            transform    = data_tf,
            seed         = 123,
            batch_size   = config.get('batch_size', 1),
            validate     = False
        )
        
        # Calculate the image features
        C = 1
        print(f"Training (C={C}): medium+hard ({config['imgs_train']} imgs total)")
        train_features, train_labels = get_features(model, train_loader)
        
        # easy, medium & hard test data
        for difficulty, dataloader in dataloaders.items():
            print(f"\nTesting: {difficulty}")
            test_features, test_labels = get_features(model, dataloader)

            # Perform logistic regression
            # C = inverse regularization constant 
            # larger C = weaker reg. more "fitting" and vice versa 
            classifier = LogisticRegression(random_state=0, C=C, max_iter=1000, verbose=0)
            classifier.fit(train_features, train_labels)
            
            # Get parameters
            n_weights = classifier.coef_.size
            n_biases = classifier.intercept_.size
            total_params = n_weights + n_biases
            print("Weights:", n_weights)
            print("Biases:", n_biases)
            print("Total parameters:", total_params)

            # Evaluate using the logistic regression classifier
            predictions = classifier.predict(test_features)
            accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
            print(f"Accuracy: {accuracy:.3f}%")
        

# Train size = 3000 (easy + med)
# C = 0.316 
#   |-> med  = ~91%
# C = 0.616 
#   |-> med  = ~92%
# C = 6.160 and 20
#   |-> med  = ~93.1%
#   |-> hard = ~59.4% 

# Train size = 6000 (easy + med)
# C = 1 and 10
#   |-> med  = ~95%
#   |-> hard = ~61.8%

# C = 1 (IDEAL)
   