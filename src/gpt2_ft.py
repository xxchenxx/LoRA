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
from model import GPT2Config, GPT2LMModel
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

def concrete_stretched(alpha, l=0., r = 1.):
	u = torch.zeros_like(alpha).uniform_().clamp_(0.0001, 0.9999)
	s = (torch.sigmoid(u.log() - (1-u).log() + alpha)).detach()
	u = s*(r-l) + l
	t = u.clamp(0, 1000)
	z = t.clamp(-1000, 1)
	dz_dt = (t < 1).float().to(alpha.device).detach()
	dt_du = (u > 0).float().to(alpha.device).detach()
	du_ds = r - l
	ds_dalpha = (s*(1-s)).detach()
	dz_dalpha = dz_dt*dt_du*du_ds*ds_dalpha
	return z.detach(), dz_dalpha.detach()

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


def train_validate(model, optimizer, alpha_optimizer, scheduler, alpha_scheduler, train_loader, valid_loader, args, gpt2_params, per_params_alpha, sparsity_pen, l0_pen, train_step = 0, epoch = 0):
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

		train_step += 1
		is_update = True if train_step % args.grad_acc == 0 else False
		avg_lm_loss.update(_lm_loss.item())
		_loss = _lm_loss/(args.grad_acc)
		if args.fp16:
			with amp.scale_loss(_loss, optimizer) as _scaled_loss:
				_scaled_loss.backward()
		else:
			_loss.backward()
		if args.per_layer_alpha == 1:
			per_layer_alpha.grad.zero_()

		for n, p in model.named_parameters():
			if p.grad is None:
				continue
			if "classifier" in n:
				gpt2_params[n][1].grad.copy_(p.grad.data)
			else:
				gpt2_params[n][1].grad.copy_(p.grad.data * grad_params[n][1].data)
				gpt2_params[n][2].grad.copy_(p.grad.data * grad_params[n][0].data *
											grad_params[n][2].data)
			
				if args.per_params_alpha == 1:
					per_params_alpha[n].grad.copy_(torch.sum(p.grad.data * grad_params[n][3].data * 
										per_params_z_grad[n].data))
				if args.per_layer_alpha == 1:
					per_layer_alpha.grad[get_layer_ind(n)] += torch.sum(p.grad.data * grad_params[n][3].data *
					per_layer_z_grad[ind].data)
		sum_l0_pen = 0
		for i in range(24):
			if l0_pen[i] != 0:
				sum_l0_pen += (sparsity_pen[i] * l0_pen[i]).sum()
		sum_l0_pen.sum().backward()			
		if is_update:
			if args.clip > 0:
				if args.fp16:
					torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.clip)
				else:
					torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
					torch.nn.utils.clip_grad_norm_(finetune_params, args.max_grad_norm)
					torch.nn.utils.clip_grad_norm_(alpha_params, args.max_grad_norm)

			optimizer.step()    
			optimizer.zero_grad()
			alpha_optimizer.step()
			alpha_optimizer.zero()

		if scheduler is not None:
			scheduler.step()
			alpha_scheduler.step()

		model.zero_grad()
		params_norm = [0, 0, 0, 0, 0, 0]
		exp_z = 0
		for n, p in gpt2_params.items():
			params_norm[0] += p[2].sum().item()
			params_norm[1] += p[2].norm().item()**2
			params_norm[2] += p[2].grad.norm().item()**2
			params_norm[3] += torch.sigmoid(p[2]).sum().item()
			params_norm[4] += p[2].numel()
			# params_norm[5] += (grad_params[n][1] > 0).float().sum().item()
			if args.per_params_alpha == 1:
				exp_z += (torch.sigmoid(p[2]).sum() * torch.sigmoid(per_params_alpha[n])).item()
			else:
				exp_z += torch.sigmoid(p[2]).sum().item()

			p[1].grad.zero_()
			p[2].grad.zero_()

		mean_exp_z = exp_z / params_norm[4]

			
		if args.per_layer_alpha == 1:
			per_layer_alpha.grad.zero_()
		if args.per_params_alpha == 1:
			for n,p in per_params_alpha.items():
				p.grad.zero_()

		
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
				torch.save({'model_state_dict': model.state_dict(), 'gpt2_params': gpt2_params}, model_path)
			distributed_sync(args)

		# evaluation interval
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

	lm_net = lm_net.cuda()

	gpt2_params = {}
	finetune_params = []
	alpha_params = []
	for n,p in lm_net.named_parameters():
		p0 = torch.zeros_like(p.data).copy_(p) #original BERT
		p1 = torch.zeros_like(p.data) #params to be fine-tuned
		p1.requires_grad = True

		p1.grad = torch.zeros_like(p.data)
		alpha = torch.zeros_like(p.data) + args.alpha_init
		alpha.requires_grad = True
		alpha.grad = torch.zeros_like(p.data)
		if args.local_rank != -1 or args.n_gpu > 1:
				name = "module." + n
		else:
				name = n
		gpt2_params[name] = [p0, p1, alpha]
		finetune_params.append(gpt2_params[name][1])
		alpha_params.append(gpt2_params[name][2])
	model_device = list(lm_net.named_parameters())[0][1].device
	if args.per_params_alpha == 1:
		per_params_alpha = {}
		for n, p in lm_net.named_parameters(): 
			alpha = torch.zeros((1)).to(model_device) + args.alpha_init
			alpha.requires_grad=True
			alpha.grad = torch.zeros_like(alpha)
			if args.local_rank != -1 or args.n_gpu > 1:
					name = "module." + n
			else:
					name = n
			per_params_alpha[name] = alpha
			alpha_params.append(alpha)
	if args.lora_dim == 0:
		no_decay = ["bias", "layer_norm.weight"]
		optimizer_grouped_parameters = [
				{
						"params": [p[1] for n, p in gpt2_params.items() if not any(nd in n for nd in no_decay) and p[1].requires_grad is True],
						"weight_decay": args.weight_decay,
				},
				{"params": [p[1] for n, p in gpt2_params.items() if any(nd in n for nd in no_decay) and p[1].requires_grad is True], "weight_decay": 0.0},
		]
		optimizer = create_adam_optimizer_from_args(lm_net, args, grouped_parameters=optimizer_grouped_parameters)
		alpha_optimizer = torch.optim.AdamW(alpha_params, lr = 0.1, eps=args.adam_epsilon)

		# create_adam_optimizer(lm_net, args.lr, args.weight_decay, correct_bias=True, adam_epislon=1.0e-6, no_decay_bias=args.no_decay_bias)
	else:
		for n, p in lm_net.named_parameters():
			if 'adapter' in n:
				print(f'{n}, shape: {p.shape}')
			else:
				p.requires_grad = False

		optimizer_grouped_parameters = [
				{
						"params": [p for n, p in lm_net.named_parameters() if 'adapter' in n],
				}
		]
		optimizer = create_adam_optimizer_from_args(None, args, grouped_parameters=optimizer_grouped_parameters)
		#None, args.lr, args.weight_decay, optimizer_grouped_parameters=optimizer_grouped_parameters, correct_bias=True, adam_epislon=1.0e-6)

	if args.max_step is None:
		args.max_step = (args.max_epoch * train_data.num_batches + args.world_size - 1) // args.world_size
		print('set max_step:', args.max_step)

	scheduler = create_optimizer_scheduler(optimizer, args)
	alpha_scheduler= create_optimizer_scheduler(alpha_optimizer, args)

	if args.fp16:
		lm_net, optimizer = amp.initialize(lm_net, optimizer, opt_level="O1")
	lm_net, optimizer = distributed_opt(args, lm_net, optimizer, grad_acc=args.grad_acc)



	total_layers = 24
	if args.sparsity_penalty_per_layer is None:
		sparsity_pen = [args.sparsity_pen] * total_layers  # NB(anon)
	else:
		sparsity_pen = args.sparsity_penalty_per_layer
		assert len(sparsity_pen) == total_layers,  "invalid sparsity penalty per layer: # of layers mismatch"
	def get_layer_ind(n):
		if "transformer" in n:
			ind = int(n.split(".")[2])
		else:
			ind = total_layers - 1
		return ind
			
	try:
		train_step = 0
		for epoch in itertools.count(start=1):
			nonzero_params = 0
			grad_params = {}
			if args.concrete_lower == 0:
				log_ratio = 0
			else:
				log_ratio = np.log(-args.concrete_lower / args.concrete_upper)
			l0_pen = [0] * total_layers
			l0_pen_sum = 0
			if args.per_params_alpha:
				per_params_z = {}
				per_params_z_grad = {}

			for n, p in lm_net.named_parameters():
				if n not in gpt2_params:
					print(" n not in gpt2_params")
				assert(n in gpt2_params)

				if "classifier" in n:
					nonzero_params += p.numel()
					p.data.copy_(gpt2_params[n][0].data + gpt2_params[n][1].data)
				else:
					if args.per_params_alpha == 1:
						params_z, params_z_grad = concrete_stretched(per_params_alpha[n], args.concrete_lower,
								args.concrete_upper)
						per_params_z[n] = params_z
						per_params_z_grad[n] = params_z_grad

					z, z_grad = concrete_stretched(gpt2_params[n][2], args.concrete_lower,
												   args.concrete_upper)
					# z, z_grad = concrete(gpt2_params[n][2], args.temp, discrete=False)
					
					ind = get_layer_ind(n)
					l0_pen[ind] += torch.sigmoid(gpt2_params[n][2] - log_ratio).sum()
					l0_pen_sum += torch.sigmoid(gpt2_params[n][2] - log_ratio).sum()
					
					if args.per_layer_alpha == 1:
						z2 = per_layer_z[ind]
					elif args.per_params_alpha == 1:
						z2 =  per_params_z[n]
					else:
						z2 = 1

					grad_params[n] = [gpt2_params[n][1] * z2, z * z2, z_grad, gpt2_params[n][1] * z]

					if args.per_params_alpha == 1:
						l0_pen[ind] += torch.sigmoid(per_params_alpha[n] - log_ratio).sum()
				
					p.data.copy_(gpt2_params[n][0].data + (z2*z).data*gpt2_params[n][1].data)
					nonzero_params += ((z2*z)>0).float().detach().sum().item()
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


