# general
from PIL import Image
from tqdm import tqdm

# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
from torchvision.models import ResNet50_Weights

# module
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# custom
from task2_generation.train_cnn import build_model
from task2_generation.dataset_character import extract_chars_as_tensor_dict


def breaking_bad(
    image_path:str,
    bonus_model_path:str,
    characters_model_path:str,
) -> str : 
    """
    Input Image -> Per Character Breakdown and Recognition -> Reverse o/p or not 
    """
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     
    # Data transforms used during training for consistency
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    data_tf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    
    # Character mapping
    char_mapping = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29, 'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35, 'a': 36, 'b': 37, 'c': 38, 'd': 39, 'e': 40, 'f': 41, 'g': 42, 'h': 43, 'i': 44, 'j': 45, 'k': 46, 'l': 47, 'm': 48, 'n': 49, 'o': 50, 'p': 51, 'q': 52, 'r': 53, 's': 54, 't': 55, 'u': 56, 'v': 57, 'w': 58, 'x': 59, 'y': 60, 'z': 61}
    
    # Build CNN 
    model_characters = build_model(model_name='simple', num_classes=62).to(device)
    model_bonus = build_model(model_name='simple', num_classes=2).to(device)
    # Load saved weights
    model_characters.load_state_dict(torch.load(characters_model_path))
    model_bonus.load_state_dict(torch.load(bonus_model_path))
    print(f"Build and loaded weights into the Bonus and Character recognition model")

    # Inference character classification model
    answer = ""
    # check for valid bounding box extraction 
    character_imgs = extract_chars_as_tensor_dict(image_path, stop=True)
    if character_imgs == 0: return
    # loop throught each character
    for char, img_tensor in character_imgs.items(): 
        # predict a character 
        img_tensor = normalize(img_tensor)
        preds  = model_characters(img_tensor.unsqueeze(0).to(device)).cpu()
        _, pred_char = torch.max(preds, 1)
        # check if it is valid
        if char_mapping[pred_char.item()] != char:
            print(f"\nMismatched character prediction. Aborting\n") 
            return "INVALID"
        else: 
            answer += char
    
    # Inference bonus model (reverse or not)
    captcha_image = Image.open(image_path).convert('RGB')
    captcha_tensor = data_tf(captcha_image).unsqueeze(0).to(device)
    preds = model_bonus(captcha_tensor)
    _, reverse = torch.max(preds, 1)
    # Reverse ? (Class to Index mapping: {'green': 0, 'red': 1})
    if reverse.item() == 1: answer = answer[::-1]     
    return answer
    

if __name__ == '__main__': 
    
    # Image path
    image_path = ""
    
    # Model paths 
    # NOTE bonus_model (256 x 256 input required)
    # NOTE char_model (any resolution input required)
    bonus_model_path = ""
    characters_model_path = ""
    
    # Break the Captcha
    answer = breaking_bad(image_path, bonus_model_path, characters_model_path)
    if answer != "INVALID": print(f"\nResult: {answer}\n")