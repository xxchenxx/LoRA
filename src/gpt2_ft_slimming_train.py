import argparse
import time
import math
import os, sys

import torch
torch.set_printoptions(threshold=100000)

import numpy as np

from gpu import add_gpu_params, parse_gpu, distributed_opt, distributed_gather, distributed_sync, cleanup
from optimizer import create_adam_optimizer, create_optimizer_scheduler, add_optimizer_params, create_adam_optimizer_from_args


from data_utils import FT_Dataset # BinCorpus, BinLMOrderedIterator
from model_prune_head import GPT2Config, GPT2LMModel, MLP
from exp_utils import create_exp_dir

import itertools

from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description='PyTorch GPT2 evaluation script.')

add_gpu_params(parser)
add_optimizer_params(parser)

parser.add_argument('--train_data', type=str, default='../data/wikitext-103',
                    help='location of training data corpus')

parser.add_argument('--valid_data', type=str, default='../data/wikitext-103',
                    help='location of validation data corpus')

parser.add_argument('--train_batch_size', type=int, default=8, help='training batch size')

parser.add_argument('--valid_batch_size', type=int, default=4, help='validation batch size')

parser.add_argument('--grad_acc', type=int, default=1, help='gradient accumlate')

parser.add_argument('--clip', type=float, default=0.0, help='gradient clip')

parser.add_argument('--seq_len', type=int, default=512, help='number of tokens to predict.')

parser.add_argument('--model_card', default='gpt2.sm', choices=['gpt2.sm', 'gpt2.md', 'gpt2.lg'], help='model names.')

parser.add_argument('--init_checkpoint', default=None, type=str, help='initial checkpoint.')

parser.add_argument('--fp16', action='store_true', help='model fp16.')

parser.add_argument('--log_interval', type=int, default=100, help='log interval.')

parser.add_argument('--eval_interval', type=int, default=2000, help='eval interval.')

parser.add_argument('--save_interval', type=int, default=500, help='save interval.')

parser.add_argument('--work_dir', type=str, default=os.getenv('PT_OUTPUT_DIR', 'gpt2_model'), help='working folder.')

parser.add_argument('--lora_dim', type=int, default=0, help='lora attn dimension.')

parser.add_argument('--lora_alpha', type=int, default=128, help='lora attn alpha.')

parser.add_argument('--obj', default='clm', choices=['jlm', 'clm'], help='language model training objective.')

parser.add_argument('--lora_dropout', default=0.0, type=float, help='dropout probability for lora layers.')

parser.add_argument('--label_smooth', default=0.0, type=float, help='label smoothing.')

parser.add_argument('--prefix_len', default=0, type=int, help='prefix length.')

parser.add_argument('--infix_len', default=0, type=int, help='infix length.')

parser.add_argument('--roll_interval', type=int, default=-1, help='rolling interval.')

parser.add_argument('--roll_lr', type=float, default=0.00001, help='rolling learning rate.')

parser.add_argument('--roll_step', type=int, default=100, help='rolling step.')

parser.add_argument('--eval_epoch', type=int, default=1, help='eval per number of epochs.')

# influence model, calculate the influence score between two samples.
def print_args(args):
  if args.rank == 0:
    print('=' * 100)
    for k, v in args.__dict__.items():
      print('    - {} : {}'.format(k, v))
    print('=' * 100)

class AverageMeter(object):
  """Computes and stores the average and current value
     Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
  """
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

def optimizer_step(_loss, _optimizer, _model, _schedule, args, is_update = True):
  if args.fp16:
    with amp.scale_loss(_loss, _optimizer) as _scaled_loss:
      _scaled_loss.backward()
  else:
    _loss.backward()

  if is_update:
    if args.clip > 0:
      if args.fp16:
        torch.nn.utils.clip_grad_norm_(amp.master_params(_optimizer), args.clip)
      else:
        torch.nn.utils.clip_grad_norm_(_model.parameters(), args.clip)

    _optimizer.step()    
    _optimizer.zero_grad()

  if _schedule is not None:
    _schedule.step()

def evaluate(model, valid_loader, args):
  model.eval()
  total_loss = 0.
  start_time = time.time()

  avg_lm_loss = AverageMeter()

  with torch.no_grad():
    for idx, data in enumerate(valid_loader):
      data = {key: value for key, value in data.items()}

      _input = data['input'].to(args.device)
      _target = data['target'].to(args.device)
      _msk = data['mask'].to(args.device)

      _lm_logits, _loss = model(_input, lm_labels=_target, lm_mask=_msk) 
      loss = _loss.mean() 
      
      avg_lm_loss.update(loss.item())

      if idx % 100 == 0:
        print('eval samples:', idx, 'loss:', loss.float())

    total_time = time.time() - start_time
    print('average loss', avg_lm_loss.avg)
  return avg_lm_loss.avg, math.exp(avg_lm_loss.avg)


def train_validate(model, optimizer, scheduler, train_loader, valid_loader, args, train_step = 0, epoch = 0):
  model.train()
  avg_lm_loss = AverageMeter()
  print('start to train the model................', epoch)
  log_start_time = time.time()
  best_val_ppl = None

  train_loader.sampler.set_epoch(epoch)
  
  for idx, data in enumerate(train_loader):
    data = {key: value for key, value in data.items()}

    _input = data['input'].to(args.device)
    _target = data['target'].to(args.device)
    _msk = data['mask'].to(args.device)

    _lm_logits, _lm_loss = model(_input, lm_labels=_target, lm_mask=_msk, label_smooth=args.label_smooth) 

    _lm_loss = _lm_loss.mean() 
    idx_layer = 0

    train_step += 1
    is_update = True if train_step % args.grad_acc == 0 else False
    avg_lm_loss.update(_lm_loss.item())
    optimizer_step(_lm_loss/(args.grad_acc), optimizer, model, scheduler, args, is_update=is_update)
    
    if train_step % args.log_interval == 0: 
      elapsed = time.time() - log_start_time

      log_str = '| epoch {:3d} step {:>8d} | {:>6d} batches | lr {:.3g}' \
                '| ms/batch {:5.2f} | loss {:5.2f} | avg loss {:5.2f} | ppl {:5.2f}'.format(
                epoch, train_step, idx + 1, optimizer.param_groups[0]['lr'], 
                elapsed * 1000 / args.log_interval, avg_lm_loss.val, avg_lm_loss.avg, math.exp(avg_lm_loss.avg)) 

      if args.rank == 0: 
        print(log_str)
      log_start_time = time.time()
      avg_lm_loss.reset()
    
    if train_step % args.save_interval == 0: 
      if args.rank == 0:
        model_path = os.path.join(args.work_dir, 'model.'+str(train_step)+'.pt')
        print('saving checkpoint', model_path)
        torch.save({'model_state_dict': model.state_dict()}, model_path)
      distributed_sync(args)

    # evaluation interval
    attention_modules = []
    # slimming_coefs = []
    '''
    for m in model.modules():
        if isinstance(m, Attention):
            attention_modules.append(m)
                    # slimming_coefs.append(m.self.slimming_coef.detach().cpu().numpy().reshape(-1))
            # slimming_coefs = np.array(slimming_coefs)
        #if training_args.slimming_coef_step > 0:
        if False:
            slimming_coefs = np.load(training_args.self_slimming_coef_file)[:, training_args.slimming_coef_step-1, :]
        else:
            # random pruning
            # get internal state of the random generator first
            rand_state = np.random.get_state()
            # set random seed
            np.random.seed(1)
            slimming_coefs = np.random.rand(len(attention_modules), 24)
            # reset internal state
            np.random.set_state(rand_state)
        #quantile_axis = -1 if training_args.self_pruning_method == 'layerwise' else None
        quantile_axis = -1
        #training_args.self_pruning_ratio = 0.5
        threshold = np.quantile(slimming_coefs, 0.5, axis=quantile_axis, keepdims=True)
        layers_masks = slimming_coefs > threshold
        for m, mask in zip(attention_modules, layers_masks):
            pruned_heads = [i for i in range(len(attention_modules)) if mask[i] == 0]
            #logger.info('pruned heads:', pruned_heads)
            m.prune_heads(pruned_heads)
    '''
    if train_step % args.eval_interval == 0:
      eval_start_time = time.time()

      valid_loss, valid_ppl = evaluate(model, valid_loader, args)

      if best_val_ppl is None or valid_ppl < best_val_ppl:
        best_val_ppl = valid_ppl
        
      log_str = '| Eval {:3d} at step {:>8d} | time: {:5.2f}s ' \
                '| valid loss {:5.2f} | valid ppl {:5.2f} | best ppl {:5.2f} '.format(
                train_step // args.eval_interval, train_step,
                (time.time() - eval_start_time), valid_loss, valid_ppl, best_val_ppl)

      if args.rank == 0:
        print('-' * 100)
        print(log_str)
        print('-' * 100)

      model.train()
      distributed_sync(args)

    if train_step == args.max_step:
      break

  if args.rank == 0:
    model_path = os.path.join(args.work_dir, 'model.'+str(train_step)+'.pt')
    print('saving checkpoint', model_path)
    torch.save({'model_state_dict': model.state_dict()}, model_path) 
  distributed_sync(args)
  return train_step
if __name__ == '__main__':
  args = parser.parse_args()
  parse_gpu(args)
  print_args(args)
  
  if args.rank == 0:
    args.logging = create_exp_dir(args.work_dir)

  train_data =  FT_Dataset(args.train_data, args.train_batch_size, args.seq_len, joint_lm=args.obj=='jlm', prefix_len=args.prefix_len, infix_len=args.infix_len)   
  
  valid_data = FT_Dataset(args.valid_data, args.valid_batch_size, args.seq_len, prefix_len=args.prefix_len, infix_len=args.infix_len)

  train_loader = DataLoader(train_data, batch_size=args.train_batch_size, num_workers=0, shuffle=False, pin_memory=False, drop_last=True,
                            sampler=torch.utils.data.distributed.DistributedSampler(train_data, seed=args.random_seed))
  
  valid_loader = DataLoader(valid_data, batch_size=args.valid_batch_size, num_workers=0, shuffle=False, pin_memory=False, drop_last=False,
                           sampler=torch.utils.data.distributed.DistributedSampler(valid_data, seed=args.random_seed))

  if args.model_card == 'gpt2.sm':
    config = GPT2Config(n_embd=768, n_layer=12, n_head=12, lora_attn_dim=args.lora_dim, lora_attn_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                        prefix_len=args.prefix_len, infix_len=args.infix_len)
  elif args.model_card == 'gpt2.md':
    config = GPT2Config(n_embd=1024, n_layer=24, n_head=16, lora_attn_dim=args.lora_dim, lora_attn_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                        prefix_len=args.prefix_len, infix_len=args.infix_len)
  elif args.model_card == 'gpt2.lg':
    config = GPT2Config(n_embd=1280, n_layer=36, n_head=20, lora_attn_dim=args.lora_dim, lora_attn_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                        prefix_len=args.prefix_len, infix_len=args.infix_len)

  lm_net = GPT2LMModel(config)
  if args.init_checkpoint is not None:
    print('loading model pretrained weight.')
    lm_net.load_weight(torch.load(args.init_checkpoint))  

  #lm_net = lm_net.cuda()

  optimizer = create_adam_optimizer_from_args(lm_net, args)
    # create_adam_optimizer(lm_net, args.lr, args.weight_decay, correct_bias=True, adam_epislon=1.0e-6, no_decay_bias=args.no_decay_bias)
  

  if args.max_step is None:
    args.max_step = (args.max_epoch * train_data.num_batches + args.world_size - 1) // args.world_size
    print('set max_step:', args.max_step)

  scheduler = create_optimizer_scheduler(optimizer, args)
  if args.fp16:
    lm_net, optimizer = amp.initialize(lm_net, optimizer, opt_level="O1")
  #lm_net, optimizer = distributed_opt(args, lm_net, optimizer, grad_acc=args.grad_acc, find_unused_parameters=True)

  try:
    train_step = 0
    block_modules = []
    checkpoint = torch.load(args.init_checkpoint, map_location="cpu")['model_state_dict']
    
    slimming_coefs = []
    count = 0
    print(checkpoint.keys())
    for k in checkpoint:
        if 'slimming_coef' in k:
            slimming_coefs.append(checkpoint[k].detach().view(-1).numpy())
            count += 1

    print(count)
    slimming_coefs = np.stack(slimming_coefs)
    print(slimming_coefs.shape)
    for m in lm_net.modules():
        if isinstance(m, MLP):
            block_modules.append(m)
    quantile_axis = -1
    threshold = np.quantile(slimming_coefs, 0.5, axis=quantile_axis, keepdims=True)
    layers_masks = slimming_coefs > threshold
    layers_masks = [layers_masks[i] for i in range(layers_masks.shape[0])]
    for m, mask in zip(block_modules, layers_masks):
        pruned_heads = [i for i in range(len(mask)) if mask[i] == 0]
        #print()
        m.prune_inter_neurons(pruned_heads)
        #m.inter_slimming = False
    for epoch in itertools.count(start=1):
      #def train_validate(model, optimizer, scheduler, train_data_iter, train_corpus, valid_data_iter, valid_corpus, args, train_step = 0, epoch = 0):
      train_step = train_validate(lm_net, optimizer, scheduler, train_loader, valid_loader, args, train_step=train_step, epoch = epoch)
      
      if train_step >= args.max_step or (args.max_epoch is not None and epoch >= args.max_epoch):
        if args.rank == 0:
          print('-' * 100)
          print('End of training')
        break
  except KeyboardInterrupt:
    if args.rank == 0:
      print('-' * 100)
      print('Exiting from training early')

  distributed_sync(args)
  print('cleanup dist ...')
  cleanup(args)

