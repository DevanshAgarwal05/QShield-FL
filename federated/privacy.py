import torch

def add_dp_noise(update_dict, sigma=0.02):
    noisy = {}
    for k, v in update_dict.items():
        noise = torch.randn_like(v) * sigma
        noisy[k] = v + noise
    return noisy
