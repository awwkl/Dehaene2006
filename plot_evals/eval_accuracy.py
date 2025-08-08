"""
python /ccn2/u/khaiaw/Code/Dehaene2006/plot_evals/eval_accuracy.py \
    --plot_dir /ccn2/u/khaiaw/Code/Dehaene2006/plot_evals/plots \
    --model_name CWM170M_RGB_BigVideo_200k/model_00200000.pt \
    --model_name CWM1B_RGB_BigVideo_200k/model_00200000.pt \
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='CWM170M_RGB_BigVideo_200k/model_00200000.pt', help='CWM Model Name (from gcloud)')
    parser.add_argument('--predictions_dir', default='/ccn2/u/khaiaw/Code/Dehaene2006/predictions', help='path to predictions directory')
    parser.add_argument('--stimuli_info_json_path', default='/ccn2/u/khaiaw/Code/Dehaene2006/downloads/GT-concepts/stimuli/stiminfo.json')
    parser.add_argument('--plot_dir')
    return parser.parse_args()

def get_ground_truth_dict(stimuli_info):
    """
    returns: dict of {group_number: correct_image}
    """
    ground_truth_dict = {}
    for group in stimuli_info:
        group_number = group["group_number"]
        correct_image = group["correct_image"]
        ground_truth_dict[group_number] = correct_image
    return ground_truth_dict

def plot_accuracy(args):
    group_nums = range(1, 46)
    
    stimuli_info_path = args.stimuli_info_json_path
    with open(stimuli_info_path, 'r') as f:
        stimuli_info = json.load(f)
    ground_truth_dict = get_ground_truth_dict(stimuli_info)

    predictions_path = os.path.join(args.predictions_dir, 'predictions.json')
    with open(predictions_path, 'r') as f:
        predictions = json.load(f)

    layer_num_accuracy_dict = {}
    for layer_num, layer_prediction_dict in predictions.items():
        layer_correct_wrong_list = []
        for group_num, predicted_image in layer_prediction_dict.items():
            ground_truth_image = ground_truth_dict.get(int(group_num))
            if ground_truth_image == predicted_image:
                layer_correct_wrong_list.append(1)
            else:
                layer_correct_wrong_list.append(0)
        layer_num_accuracy_dict[layer_num] = 100 * sum(layer_correct_wrong_list) / len(layer_correct_wrong_list)

    # Plot the accuracy
    model_name = args.model_name.replace('/', '_')
    plt.figure(figsize=(10, 5))
    line, = plt.plot(list(layer_num_accuracy_dict.keys()), list(layer_num_accuracy_dict.values()), marker='o')
    for x, y in zip(layer_num_accuracy_dict.keys(), layer_num_accuracy_dict.values()):
        plt.text(x, y + 0.02, f'{y:.1f}', ha='center')
    plt.plot(layer_num_accuracy_dict.keys(), layer_num_accuracy_dict.values(), marker='o')
    plt.title('Layer-wise Accuracy')
    plt.xlabel('Layer Number')
    plt.ylabel('Accuracy')
    plt.ylim(0, 100)
    plt.grid()
    plt.xticks(list(layer_num_accuracy_dict.keys()), rotation=45)
    plt.savefig(os.path.join(args.plot_dir, f'accuracy_{model_name}.png'))
    plt.close()
    print('Plot saved to:', os.path.join(args.plot_dir, f'accuracy_{model_name}.png'))

if __name__ == "__main__":
    args = get_args()
    args.predictions_dir = os.path.join(args.predictions_dir, args.model_name)
    print(args)
    
    plot_accuracy(args)