"""
Run this file from the vjepa2 code repo, not from this repo (Dehaene dataset).

conda activate vjepa2-312
cd /ccn2/u/khaiaw/Code/baselines/vjepa2/

python /ccn2/u/khaiaw/Code/Dehaene2006/generate_vjepa2_predictions.py \
    --vjepa2_config_path evals/optical_flow/config.yaml \
    --model_name downloads/vith.pt \
    --model_name anneal/32.8.vitl16-256px-16f/babyview_bs3072_e60/e40.pt \
    --model_size vit_large \

"""

import sys
sys.path.append("/ccn2/u/khaiaw/Code/baselines/vjepa2/")

import copy
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
import yaml
import torch
import torchvision
from torchvision import transforms
import torch.nn.functional as F
from app.vjepa.utils import init_opt, init_video_model, load_checkpoint


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='downloads/vitl.pt', help='CWM Model Name (from gcloud)')
    parser.add_argument('--vjepa2_config_path', type=str, default='evals/optical_flow/config.yaml')
    parser.add_argument('--model_size', type=str, choices=['vit_large', 'vit_huge', 'vit_giant_xformers'], default='vit_large', help='Size of the model (from timm)')
    parser.add_argument('--stimuli_dir', default='/ccn2/u/khaiaw/Code/Dehaene2006/downloads/GT-concepts/stimuli', help='path to stimuli directory')
    parser.add_argument('--out_dir', type=str, default='/ccn2/u/khaiaw/Code/Dehaene2006/predictions', help='path to output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on')
    return parser.parse_args()

@torch.no_grad()
def compute_similarity(X1_np, X2_np):
    # X: [1024, 1280] = [tokens, dim]
    X1 = torch.from_numpy(X1_np).float()
    X2 = torch.from_numpy(X2_np).float()

    # unit-norm each token first
    # X1 = F.normalize(X1, dim=-1, eps=1e-8)
    # X2 = F.normalize(X2, dim=-1, eps=1e-8)

    # remove per-image token mean (kills background bias)
    # X1 = X1 - X1.mean(dim=0, keepdim=True)
    # X2 = X2 - X2.mean(dim=0, keepdim=True)

    mean1 = X1.mean(dim=0)
    mean2 = X2.mean(dim=0)

    v1 = F.normalize(mean1, dim=0)   # [1280]
    v2 = F.normalize(mean2, dim=0)
    
    # v1 = X1
    # v2 = X2
    # similarity = F.cosine_similarity(v1, v2, dim=-1).mean()
    # return similarity.item()
    
    return torch.dot(v1, v2).item()

def load_target_encoder(args):
    config_params = None
    with open(args.vjepa2_config_path, "r") as y_file:
        config_params = yaml.load(y_file, Loader=yaml.FullLoader)
    print(config_params)
    print(args)
    
    cfgs_model = config_params.get("model")
    cfgs_model['model_name'] = args.model_size
    cfgs_mask = config_params.get("mask")
    cfgs_data = config_params.get("data")
    cfgs_meta = config_params.get("meta")
    cfgs_opt = config_params.get("optimization")
    
    # === Initialize model (without weights) ===
    encoder, predictor = init_video_model(
        uniform_power=cfgs_model.get("uniform_power", False),
        use_mask_tokens=cfgs_model.get("use_mask_tokens", False),
        num_mask_tokens=int(len(cfgs_mask) * len(cfgs_data.get("dataset_fpcs"))),
        zero_init_mask_tokens=cfgs_model.get("zero_init_mask_tokens", True),
        device=args.device,
        patch_size=cfgs_data.get("patch_size"),
        max_num_frames=max(cfgs_data.get("dataset_fpcs")),
        tubelet_size= cfgs_data.get("tubelet_size"),
        model_name=cfgs_model.get("model_name"),
        crop_size=256,
        pred_depth=cfgs_model.get("pred_depth"),
        pred_num_heads=cfgs_model.get("pred_num_heads", None),
        pred_embed_dim=cfgs_model.get("pred_embed_dim"),
        use_sdpa=cfgs_meta.get("use_sdpa", False),
        use_silu=cfgs_model.get("use_silu", False),
        use_pred_silu=cfgs_model.get("use_pred_silu", False),
        wide_silu=cfgs_model.get("wide_silu", True),
        use_rope=cfgs_model.get("use_rope", False),
        use_activation_checkpointing=cfgs_model.get("use_activation_checkpointing", False),
    )
    target_encoder = copy.deepcopy(encoder)

    # === Load model weights from checkpoint ===
    checkpoint = torch.load(args.model_name, map_location=torch.device("cpu"))
    def load_state(model, key):
        state = checkpoint[key]
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)

    del encoder, predictor
    load_state(target_encoder, "target_encoder")
    return target_encoder

def generate_predictions(args):
    target_encoder = load_target_encoder(args)

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
            
            img_tensor = torchvision.io.read_image(path=png_path, mode=torchvision.io.ImageReadMode.RGB)
            img_tensor = torchvision.transforms.functional.resize(img_tensor, [256, 256]) # [3, H, W]
            img_tensor = img_tensor.float() / 255.0
            img_tensor = torch.stack([img_tensor, img_tensor], dim=0) # [T, 3, H, W] # this needs to be doubled because vjepa2 takes tubelet size of 2, kernel size (2 x 16 x 16)

            clips = img_tensor.unsqueeze(0) # [1, T, 3, H, W]
            clips = clips.permute(0, 2, 1, 3, 4) # [1, 3, T, H, W]
            clips = clips.unsqueeze(0) # [1, 1, 3, T, 256, 256]
            clips = clips.to(args.device)
            
            feat = target_encoder(clips)
            feat = [F.layer_norm(hi, (hi.size(-1),)) for hi in feat]
            feat = feat[0]# [1, 256, 1024]
            
            feat = feat.detach().cpu().numpy() # [1, 256, 1024]
            stimuli_activations_dict[png_name] = feat

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
    
    args.out_dir = os.path.join(args.out_dir, 'vjepa2', args.model_name)
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(args)
    generate_predictions(args)