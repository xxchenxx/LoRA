import torch

a = torch.load('./trained_models/GPT2_M_bilateral_rank1_1000/e2e/model.105155.pt',map_location='cpu')['model_state_dict']
b = torch.load('./trained_models/GPT2_M_bilateral_rank1_500/e2e/model.105155.pt',map_location='cpu')['model_state_dict']

for key in a:
    if ('S_Q' in key or 'S_V' in key) and ('embedding' not in key):
        print(key)
        print(a[key].sum())
        print(b[key].sum())
        print((a[key] * b[key]).sum())
        print('----')
        