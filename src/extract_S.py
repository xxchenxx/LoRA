import torch

checkpoint = torch.load("./trained_models/GPT2_M_compress/e2e/model.2000.pt", map_location="cpu")['model_state_dict']

print(checkpoint.keys())

extracted = {}
for key in checkpoint:
    if 'S_Q' or 'S_V' in key:
        extracted[key] = checkpoint[key]

torch.save(extracted, "extracted_S.pth.ptar")