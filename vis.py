import torch
import numpy as np
init = torch.load("./pretrained_checkpoints/gpt2-medium-pytorch_model.bin", map_location="cpu")
trained = torch.load("./trained_models/GPT2_M_original/e2e/model.105155.pt",  map_location="cpu")['model_state_dict']

#diff = []
for key in init:
    #diff.append((init[key] - trained[key]).numpy())
    print(np.abs((init[key] - trained[key]).numpy()).mean())





