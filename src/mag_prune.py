import torch
import sys

saved = torch.load(sys.argv[1], map_location="cpu")
nonzero = 0
total = 0
state_dict = saved['model_state_dict']
for n in state_dict:
    if 'adapter' in n:
    #print("state_dict[n]= device=", state_dict[n].device, " saved bert params device=", saved["bert_params"][n][0])
        extra = state_dict[n] - saved["gpt2_params"][n][0]
        nonzero += torch.sum(extra != 0).item()
        total += extra.numel()
print("=====> Initial sparsity: %.4lf" % (1.0 * nonzero / total))
print("total=", total, "nonzero=", nonzero)

def get_sparsity(threshold):
    nonzero = 0
    total = 0
    state_dict = saved['model_state_dict']
    for n in state_dict:
        if 'adapter' in n:
            extra = state_dict[n] - saved["gpt2_params"][n][0]
            nonzero += torch.sum((torch.abs(extra) > threshold)).item()
            total += extra.numel()
    return 1.0 * nonzero / total

l = 0
r = 1.0
target_sparsity = 0.25
print("get_sparsity(r)=", get_sparsity(r), " target=", target_sparsity)
#assert get_sparsity(r) <= target_sparsity and get_sparsity(l) >= target_sparsity
while l + 1e-6 < r:
    mid = (l + r) / 2.0
    if get_sparsity(mid) <= target_sparsity:
        r = mid
    else:
        l = mid

threshold = r
print("threshold=", threshold, " sparsity=", get_sparsity(threshold))

import copy
new_state_dict = copy.deepcopy(state_dict)
for n in saved['model_state_dict']:
    if 'adatper' in n:
        extra = saved['model_state_dict'][n] - saved["gpt2_params"][n][0]
        extra[torch.abs(extra) < threshold] = 0
        assert extra.size() == saved["bert_params"][n][0].size()
        new_state_dict[n] = saved["bert_params"][n][0] + extra

torch.save(saved, sys.argv[1] + "_pruned_0.25")