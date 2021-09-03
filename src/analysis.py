import torch
a = torch.load("./trained_models/GPT2_M_bilateral_rank1_1/e2e/model.105155.pt", map_location='cpu')['model_state_dict']
b = torch.load("./trained_models/GPT2_M_bilateral_rank1_10/e2e/model.105155.pt", map_location='cpu')['model_state_dict']

for key in a:
    print(key)
    if 'S_V' in key:
        print(key)
        #print(a[key])
        #print(b[key])
        print(a[key].abs().sum())
        print(b[key].abs().sum())
        print((a[key] * b[key]).abs().sum())