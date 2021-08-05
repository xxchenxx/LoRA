import torch
import numpy as np

checkpoint = torch.load("./trained_models/GPT2_M_slimming/e2e/model.52575.pt", map_location="cpu")
print(checkpoint.keys())
