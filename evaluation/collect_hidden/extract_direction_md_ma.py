import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from tqdm import tqdm
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from pca import PCA

def generate_steering_atlas(features_path, save_file, save_name, n_clusters=4):

    print(f"Loading features from: {features_path}")

    data = torch.load(features_path, map_location='cpu')

    pos_data = data[:, 0, :, :].float()
    neg_data = data[:, 1, :, :].float()
    all_diffs = pos_data - neg_data

    n_samples, n_layers, n_hidden = all_diffs.shape

    fine_grained_vectors = torch.zeros((n_layers, n_clusters, n_hidden))

    layer_angle_ratios = torch.zeros(n_layers)

    fine_grained_stabilities = torch.zeros((n_layers, n_clusters))

    print(f"Processing {n_layers} layers with geometric decomposition...")

    prev_aligned_vectors = None

    for l in tqdm(range(n_layers)):
        layer_pos = pos_data[:, l, :]
        layer_neg = neg_data[:, l, :]
        layer_diff = all_diffs[:, l, :]

        d_sq = torch.norm(layer_diff, dim=-1)**2
        m_sq = (torch.norm(layer_pos, dim=-1) - torch.norm(layer_neg, dim=-1))**2
        a_sq = d_sq - m_sq

        layer_angle_ratios[l] = (a_sq / (d_sq + 1e-8)).mean()

        norm_centered = F.normalize(layer_diff, p=2, dim=1)
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = km.fit_predict(norm_centered.cpu().numpy())

        current_layer_vectors = torch.zeros((n_clusters, n_hidden))

        for k in range(n_clusters):
            mask = (labels == k)
            cluster_samples = layer_diff[mask]

            if cluster_samples.size(0) > 1:
                pca_model = PCA(n_components=1)
                pca_model.fit(cluster_samples)
                cluster_direction = pca_model.components_[0, 0, :]

                if torch.dot(cluster_direction, cluster_samples.mean(dim=0)) < 0:
                    cluster_direction = -cluster_direction

                current_layer_vectors[k] = cluster_direction
            elif cluster_samples.size(0) == 1:
                current_layer_vectors[k] = cluster_samples[0]

        if prev_aligned_vectors is not None:
            sim_matrix = torch.zeros((n_clusters, n_clusters))
            for i in range(n_clusters):
                for j in range(n_clusters):
                    sim_matrix[i, j] = F.cosine_similarity(prev_aligned_vectors[i].unsqueeze(0),
                                                           current_layer_vectors[j].unsqueeze(0))

            row_ind, col_ind = linear_sum_assignment(-sim_matrix.numpy())

            aligned_vectors = torch.zeros_like(current_layer_vectors)

            for i, j in zip(row_ind, col_ind):
                aligned_vectors[i] = current_layer_vectors[j]
                fine_grained_stabilities[l, i] = sim_matrix[i, j]

            fine_grained_vectors[l] = aligned_vectors
            prev_aligned_vectors = aligned_vectors
        else:
            fine_grained_vectors[l] = current_layer_vectors
            fine_grained_stabilities[l] = 1.0
            prev_aligned_vectors = current_layer_vectors

    results = {
        "layer_multi_direction": fine_grained_vectors,
        "layer_multi_stabilities": fine_grained_stabilities
    }

    save_path = os.path.join(save_file, save_name)
    torch.save(results, save_path)
    print(f"\nExtraction Complete! Saved to {save_path}")
    return results

models = [
    ("gpt2",        "/evaluation/collect_hidden/collect_hidden_states_md/gpt2__2000/train_hidden_states.pt",
                    "/evaluation/collect_hidden/extract_hidden_direction_md/gpt2__2000"),
    ("mistral",     "/evaluation/collect_hidden/collect_hidden_states_md/mistral__2000/train_hidden_states.pt",
                    "/evaluation/collect_hidden/extract_hidden_direction_md/mistral__2000"),
    ("mistral-sft", "/evaluation/collect_hidden/collect_hidden_states_md/mistral-sft__2000/train_hidden_states.pt",
                    "/evaluation/collect_hidden/extract_hidden_direction_md/mistral-sft__2000"),
    ("llama-7b-hf", "/evaluation/collect_hidden/collect_hidden_states_md/llama-7b-hf__2000/train_hidden_states.pt",
                    "/evaluation/collect_hidden/extract_hidden_direction_md/llama-7b-hf__2000"),
    ("llama-7b-sft","/evaluation/collect_hidden/collect_hidden_states_md/llama-7b-sft__2000/train_hidden_states.pt",
                    "/evaluation/collect_hidden/extract_hidden_direction_md/llama-7b-sft__2000"),
    ("llama-13b-hf","/evaluation/collect_hidden/collect_hidden_states_md/llama-13b-hf__2000/train_hidden_states.pt",
                    "/evaluation/collect_hidden/extract_hidden_direction_md/llama-13b-hf__2000"),
    ("llama-30b-hf","/evaluation/collect_hidden/collect_hidden_states_md/llama-30b-hf__2000/train_hidden_states.pt",
                    "/evaluation/collect_hidden/extract_hidden_direction_md/llama-30b-hf__2000"),
    ("opt",         "/evaluation/collect_hidden/collect_hidden_states_md/opt__2000/train_hidden_states.pt",
                    "/evaluation/collect_hidden/extract_hidden_direction_md/opt__2000"),
]

save_name = "layer_direction_pca_mean_geometric.pt"
n_clusters = 4
for name, features_path, save_dir in models:
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n=== Processing {name} ===")
    generate_steering_atlas(features_path, save_dir, save_name, n_clusters)