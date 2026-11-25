
import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM, LlamaConfig
from copy import deepcopy

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch.nn.functional as F
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

def load_only_score_model(weight_path, hidden_size=5120):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(weight_path, map_location=device)

    score_model = nn.Sequential(
        nn.Linear(hidden_size, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1)
    )

    score_model.load_state_dict(state_dict)
    score_model.to(device).eval()
    print(f"✅ 已成功从 {weight_path} 加载 Reward 线性层")
    return score_model, device

jobs = [
    dict(
        name="gpt2",
        weights="/evaluation/collect_hidden/train_hidden_states_md/gpt2__dp_2000/checkpoints/best_score_model.pth",
        features="/evaluation/collect_hidden/collect_hidden_states_md/gpt2__2000/train_hidden_states.pt",
        save_dir="/evaluation/collect_hidden/extract_hidden_direction_md/gpt2__2000",
        hidden_size=1024,
    ),
    dict(
        name="mistral",
        weights="/evaluation/collect_hidden/train_hidden_states_md/mistral__dp_2000/checkpoints/best_score_model.pth",
        features="/evaluation/collect_hidden/collect_hidden_states_md/mistral__2000/train_hidden_states.pt",
        save_dir="/evaluation/collect_hidden/extract_hidden_direction_md/mistral__2000",
        hidden_size=4096,
    ),
    dict(
        name="mistral-sft",
        weights="/evaluation/collect_hidden/train_hidden_states_md/mistral-sft__dp_2000/checkpoints/best_score_model.pth",
        features="/evaluation/collect_hidden/collect_hidden_states_md/mistral-sft__2000/train_hidden_states.pt",
        save_dir="/evaluation/collect_hidden/extract_hidden_direction_md/mistral-sft__2000",
        hidden_size=4096,
    ),
    dict(
        name="llama-13b-hf",
        weights="/evaluation/collect_hidden/train_hidden_states_md/llama-13b-hf__dp_2000/checkpoints/best_score_model.pth",
        features="/evaluation/collect_hidden/collect_hidden_states_md/llama-13b-hf__2000/train_hidden_states.pt",
        save_dir="/evaluation/collect_hidden/extract_hidden_direction_md/llama-13b-hf__2000",
        hidden_size=5120,
    ),
    dict(
        name="llama-30b-hf",
        weights="/evaluation/collect_hidden/train_hidden_states_md/llama-30b-hf__dp_2000/checkpoints/best_score_model.pth",
        features="/evaluation/collect_hidden/collect_hidden_states_md/llama-30b-hf__2000/train_hidden_states.pt",
        save_dir="/evaluation/collect_hidden/extract_hidden_direction_md/llama-30b-hf__2000",
        hidden_size=6656,
    ),
    dict(
        name="llama-7b-sft",
        weights="/evaluation/collect_hidden/train_hidden_states_md/llama-7b-sft__dp_2000/checkpoints/best_score_model.pth",
        features="/evaluation/collect_hidden/collect_hidden_states_md/llama-7b-sft__2000/train_hidden_states.pt",
        save_dir="/evaluation/collect_hidden/extract_hidden_direction_md/llama-7b-sft__2000",
        hidden_size=4096,
    ),
    dict(
        name="llama-7b-hf",
        weights="/evaluation/collect_hidden/train_hidden_states_md/llama-7b-hf__dp_2000/checkpoints/best_score_model.pth",
        features="/evaluation/collect_hidden/collect_hidden_states_md/llama-7b-hf__2000/train_hidden_states.pt",
        save_dir="/evaluation/collect_hidden/extract_hidden_direction_md/llama-7b-hf__2000",
        hidden_size=4096,
    ),
    dict(
        name="opt",
        weights="/evaluation/collect_hidden/train_hidden_states_md/opt__dp_2000/checkpoints/best_score_model.pth",
        features="/evaluation/collect_hidden/collect_hidden_states_md/opt__2000/train_hidden_states.pt",
        save_dir="/evaluation/collect_hidden/extract_hidden_direction_md/opt__2000",
        hidden_size=4096,
    ),
]

for job in jobs:
    print(f"\n=== Processing {job['name']} ===")
    score_model, device = load_only_score_model(job["weights"], hidden_size=job["hidden_size"])
    all_features = torch.load(job["features"], map_location=device).float()

    sample_num, _, max_layer, hidden_size = all_features.shape

    with torch.no_grad():

        pos_all = all_features[:, 0, :, :]
        neg_all = all_features[:, 1, :, :]
        pos_flat = pos_all.reshape(-1, hidden_size)
        neg_flat = neg_all.reshape(-1, hidden_size)

        pos_scores_flat = score_model(pos_flat)
        neg_scores_flat = score_model(neg_flat)

        pos_scores = pos_scores_flat.view(sample_num, max_layer)
        neg_scores = neg_scores_flat.view(sample_num, max_layer)

        avg_pos_per_layer = pos_scores.mean(dim=0)
        avg_neg_per_layer = neg_scores.mean(dim=0)


        q_list = torch.tensor([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])

        pos_q = torch.quantile(pos_scores, q_list, dim=0)
        neg_q = torch.quantile(neg_scores, q_list, dim=0)


        save_data = {
            "pos_layer": avg_pos_per_layer.cpu(),
            "neg_layer": avg_neg_per_layer.cpu(),

            "pos_q": pos_q.cpu(),
            "neg_q": neg_q.cpu(),


        }

        save_path = os.path.join(job["save_dir"], "reward_layer.pt")
        torch.save(save_data, save_path)

       