import torch

checkpoint = torch.load("./trained_models/GPT2_M_compress/e2e/model.2000.pt")

print(checkpoint.keys())