import torch
import numpy as np
import sys
checkpoint = torch.load(f"./trained_models/GPT2_M/e2e/model.{sys.argv[1]}.pt", map_location="cpu")['model_state_dict']

print(checkpoint.keys())

extracted = {}
for key in checkpoint:
    if ('S_Q' in key) or ('S_V' in key):
        extracted[key] = checkpoint[key]
        print(checkpoint[key].shape)
        print(checkpoint[key].abs().mean())

        if torch.isnan(checkpoint[key].abs().mean()):
            mask = [1] * 128
            mask.extend([0] * (1024 * 1024 - 128))
            mask = np.array(mask)
            mask = torch.from_numpy(np.random.permutation(mask))
            mask = mask.view(*checkpoint[key].shape)
        else:
            threshold, _ = torch.kthvalue(checkpoint[key].view(-1).detach().cpu(), 1024 * 1024 - 128)
            mask = (checkpoint[key] >= threshold).long()

        extracted[key] = mask

torch.save(extracted, "extracted_S.pth.tar")