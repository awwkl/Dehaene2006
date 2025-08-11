"""
python /ccn2/u/khaiaw/Code/Dehaene2006/generate_cwm_predictions.py \
    --model_name CWM170M_RGB_Babyview_200k/model_00200000.pt \
    --model_name CWM1B_RGB_Babyview_200k/model_00200000.pt \
    --model_name CWM170M_RGB_BigVideo_200k/model_00200000.pt \
    --model_name CWM1B_RGB_BigVideo_200k/model_00200000.pt \
"""

import os
import argparse
import importlib
import random
import numpy as np
import matplotlib.pyplot as plt
import glob
import json
from PIL import Image
from tqdm import tqdm
from ccwm.cwm.cwm_predictor import CWMPredictor
import torch
import torchvision
import torch.nn.functional as F


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='CWM170M_RGB_BigVideo_200k/model_00200000.pt', help='CWM Model Name (from gcloud)')
    parser.add_argument('--stimuli_dir', default='/ccn2/u/khaiaw/Code/Dehaene2006/downloads/GT-concepts/stimuli', help='path to stimuli directory')
    parser.add_argument('--out_dir', type=str, default='/ccn2/u/khaiaw/Code/Dehaene2006/predictions', help='path to output directory')
    return parser.parse_args()

@torch.no_grad()
def compute_similarity(X1_np, X2_np):
    # X: [1024, 1280] = [tokens, dim]
    X1 = torch.from_numpy(X1_np).float()
    X2 = torch.from_numpy(X2_np).float()

    # unit-norm each token first
    X1 = F.normalize(X1, dim=-1, eps=1e-8)
    X2 = F.normalize(X2, dim=-1, eps=1e-8)

    # remove per-image token mean (kills background bias)
    X1 = X1 - X1.mean(dim=0, keepdim=True)
    X2 = X2 - X2.mean(dim=0, keepdim=True)

    mean1 = X1.mean(dim=0)
    mean2 = X2.mean(dim=0)

    v1 = F.normalize(mean1, dim=0)   # [1280]
    v2 = F.normalize(mean2, dim=0)
    return torch.dot(v1, v2).item()

def generate_predictions(args):
    cwm_predictor = CWMPredictor(model_name=args.model_name)
    
    stimuli_png_names = os.listdir(args.stimuli_dir)
    stimuli_png_names = [f for f in stimuli_png_names if f.endswith('.png')]
    stimuli_group_num_list = range(1, 46)

    prediction_dict = {}
    for stimuli_group_num in tqdm(stimuli_group_num_list):
        stimuli_group_png_names = [name for name in stimuli_png_names if name.split('_')[0] == str(stimuli_group_num)]
        stimuli_group_png_names.sort()
        # ['1_1.png', '1_2.png', '1_3.png', '1_4.png', '1_5.png', '1_6.png']
        
        stimuli_activations_dict = {}
        for png_name in stimuli_group_png_names:
            png_path = os.path.join(args.stimuli_dir, png_name)

            frame0 = Image.open(png_path).convert("RGB")
            results = cwm_predictor.single_image_forward(frame0, frame_gap=-1, return_activations=True)
            activations = results['activations']
            stimuli_activations_dict[png_name] = activations

        num_layers = stimuli_activations_dict[png_name].shape[0]
        
        # stimuli_activations_dict: {<png_name>: <activations>}.
        # Compute pairwise cosine similarities between the 6 images, and then for each image we computed its average cosine similarity to the other 5 images.
        # The image with the lowest average cosine similarity was selected as the odd-one-out.
        # use torch functional F cosine similarity
        for layer_num in range(num_layers):
            if layer_num not in prediction_dict:
                prediction_dict[layer_num] = {}

            stimuli_feat_dict = {name: activations[layer_num] for name, activations in stimuli_activations_dict.items()}
            stimuli_cosine_similarity_to_other_images = {}
            for png_name in stimuli_group_png_names:
                total_similarity = 0
                count = 0
                for other_png_name in stimuli_group_png_names:
                    if other_png_name != png_name:
                        # x1 = torch.from_numpy(stimuli_feat_dict[png_name]).float()
                        # x2 = torch.from_numpy(stimuli_feat_dict[other_png_name]).float()
                        # breakpoint()
                        # similarity = F.cosine_similarity(x1, x2, dim=-1)
                        # similarity = similarity.mean()
                        similarity = compute_similarity(stimuli_feat_dict[png_name], stimuli_feat_dict[other_png_name])
                        total_similarity += similarity
                        count += 1
                avg_similarity = total_similarity / count if count > 0 else 0
                stimuli_cosine_similarity_to_other_images[png_name] = avg_similarity

            odd_one_out = min(stimuli_group_png_names, key=lambda x: stimuli_cosine_similarity_to_other_images.get(x, 0))
            prediction_dict[layer_num][stimuli_group_num] = odd_one_out

    with open(os.path.join(args.out_dir, 'predictions.json'), 'w') as f:
        json.dump(prediction_dict, f)

if __name__ == "__main__":
    args = get_args()
    
    args.out_dir = os.path.join(args.out_dir, args.model_name)
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(args)
    generate_predictions(args)