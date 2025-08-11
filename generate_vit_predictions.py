"""
python /ccn2/u/khaiaw/Code/Dehaene2006/generate_vit_predictions.py \
    --hf_model_id google/vit-base-patch16-224 \
    --hf_model_id google/vit-large-patch16-224 \

"""

# generate_vit_predictions.py
import os
import argparse
import json
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
import numpy as np

from transformers import AutoImageProcessor, ViTModel


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--hf_model_id',
        type=str,
        default='google/vit-base-patch16-224',
        help='Hugging Face ViT model id (e.g., google/vit-base-patch16-224)'
    )
    parser.add_argument(
        '--stimuli_dir',
        default='/ccn2/u/khaiaw/Code/Dehaene2006/downloads/GT-concepts/stimuli',
        help='path to stimuli directory'
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default='/ccn2/u/khaiaw/Code/Dehaene2006/predictions',
        help='path to output directory'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cuda', 'cpu'],
        help='device to run the model on'
    )
    return parser.parse_args()


@torch.no_grad()
def compute_similarity(X1_np, X2_np):
    
    # X1_np: [196, 768]
    
    # X: [tokens, dim]
    X1 = torch.from_numpy(X1_np).float()
    X2 = torch.from_numpy(X2_np).float()

    # unit-norm each token first
    # X1 = F.normalize(X1, dim=-1, eps=1e-8)
    # X2 = F.normalize(X2, dim=-1, eps=1e-8)

    # remove per-image token mean (kills background bias)
    # X1 = X1 - X1.mean(dim=0, keepdim=True)
    # X2 = X2 - X2.mean(dim=0, keepdim=True)

    # average tokens to a single vector, then normalize and dot
    v1 = F.normalize(X1.mean(dim=0), dim=0)
    v2 = F.normalize(X2.mean(dim=0), dim=0)
    return torch.dot(v1, v2).item()


class ViTPredictor:
    def __init__(self, hf_model_id: str, device: str = 'cuda'):
        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(hf_model_id)
        # Ask the model for hidden states from every layer
        self.model = ViTModel.from_pretrained(hf_model_id, output_hidden_states=True)
        self.model.eval().to(self.device)

    @torch.no_grad()
    def single_image_forward(self, pil_img: Image.Image, return_activations: bool = True):
        # Preprocess (resize/center-crop to model's expected size)
        inputs = self.processor(images=pil_img, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
        # outputs.hidden_states is a tuple: (embeddings, layer1, layer2, ..., layerN)
        # We exclude the embeddings (index 0)
        per_layer = []
        for h in outputs.hidden_states[1:]:           # skip embeddings
            tokens = h[:, :, :]                      # keep CLS token
            per_layer.append(tokens.squeeze(0).detach().cpu().numpy())  # [tokens, dim]

        activations = np.stack(per_layer, axis=0)     # [num_layers, tokens, dim]
        return {'activations': activations} if return_activations else {}



def generate_predictions(args):
    vit = ViTPredictor(
        hf_model_id=args.hf_model_id,
        device=args.device,
    )

    stimuli_png_names = [f for f in os.listdir(args.stimuli_dir) if f.endswith('.png')]
    stimuli_group_nums = range(1, 46)

    prediction_dict = {}
    for stimuli_group_num in tqdm(stimuli_group_nums, desc='Groups'):
        group_pngs = sorted([n for n in stimuli_png_names if n.split('_')[0] == str(stimuli_group_num)])
        # Expect like: ['1_1.png', '1_2.png', '1_3.png', '1_4.png', '1_5.png', '1_6.png']

        # Cache activations per image
        stimuli_activations = {}
        for png_name in group_pngs:
            path = os.path.join(args.stimuli_dir, png_name)
            img = Image.open(path).convert('RGB')
            out = vit.single_image_forward(img, return_activations=True)
            stimuli_activations[png_name] = out['activations']  # [1, tokens, dim]

        # num_layers = 1 (last layer only)
        num_layers = stimuli_activations[group_pngs[0]].shape[0]

        for layer_idx in range(num_layers):  # effectively just 0
            if layer_idx not in prediction_dict:
                prediction_dict[layer_idx] = {}

            # Pull this layer’s tokens for each image
            layer_feats = {name: acts[layer_idx] for name, acts in stimuli_activations.items()}

            # Average cosine similarity to others per image
            avg_sims = {}
            for name in group_pngs:
                total = 0.0
                cnt = 0
                for other in group_pngs:
                    if other == name:
                        continue
                    sim = compute_similarity(layer_feats[name], layer_feats[other])
                    total += sim
                    cnt += 1
                avg_sims[name] = (total / cnt) if cnt > 0 else 0.0

            odd_one_out = min(group_pngs, key=lambda x: avg_sims.get(x, 0.0))
            prediction_dict[layer_idx][stimuli_group_num] = odd_one_out

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    # Put in a subfolder under out_dir named after the HF model id
    safe_model_name = args.hf_model_id.replace('/', '__')
    out_model_dir = os.path.join(args.out_dir, safe_model_name)
    os.makedirs(out_model_dir, exist_ok=True)

    with open(os.path.join(out_model_dir, 'predictions.json'), 'w') as f:
        json.dump(prediction_dict, f, indent=2)

    print(f'Wrote predictions to {os.path.join(out_model_dir, "predictions.json")}')


if __name__ == '__main__':
    args = get_args()
    print(args)
    generate_predictions(args)
