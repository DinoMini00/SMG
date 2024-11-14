import torch
import numpy as np
import os
import glob
import json
import random
import augmentations
import subprocess
from datetime import datetime


class eval_mode(object):
	def __init__(self, *models):
		self.models = models

	def __enter__(self):
		self.prev_states = []
		for model in self.models:
			self.prev_states.append(model.training)
			model.train(False)

	def __exit__(self, *args):
		for model, state in zip(self.models, self.prev_states):
			model.train(state)
		return False


def soft_update_params(net, target_net, tau):
	for param, target_param in zip(net.parameters(), target_net.parameters()):
		target_param.data.copy_(
			tau * param.data + (1 - tau) * target_param.data
		)


def cat(x, y, axis=0):
	return torch.cat([x, y], axis=0)


def set_seed_everywhere(seed):
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)


def write_info(args, fp):
	data = {
		'timestamp': str(datetime.now()),
		'git': subprocess.check_output(["git", "describe", "--always"]).strip().decode(),
		'args': vars(args)
	}
	with open(fp, 'w') as f:
		json.dump(data, f, indent=4, separators=(',', ': '))


def load_config(key=None):
	path = os.path.join('setup', 'config.cfg')
	with open(path) as f:
		data = json.load(f)
	if key is not None:
		return data[key]
	return data


def make_dir(dir_path):
	try:
		os.makedirs(dir_path)
	except OSError:
		pass
	return dir_path


def listdir(dir_path, filetype='jpg', sort=True):
	fpath = os.path.join(dir_path, f'*.{filetype}')
	fpaths = glob.glob(fpath, recursive=True)
	if sort:
		return sorted(fpaths)
	return fpaths


def prefill_memory(obses, capacity, obs_shape):
	"""Reserves memory for replay buffer"""
	c,h,w = obs_shape
	for _ in range(capacity):
		frame = np.ones((3,h,w), dtype=np.uint8)
		obses.append(frame)
	return obses


class ReplayBuffer(object):
	"""Buffer to store environment transitions"""
	def __init__(self, obs_shape, action_shape, capacity, batch_size, prefill=True):
		self.capacity = capacity
		self.batch_size = batch_size

		self._obses = []
		if prefill:
			self._obses = prefill_memory(self._obses, capacity, obs_shape)
		self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
		self.rewards = np.empty((capacity, 1), dtype=np.float32)
		self.not_dones = np.empty((capacity, 1), dtype=np.float32)

		self.idx = 0
		self.full = False

	def add(self, obs, action, reward, next_obs, done):
		obses = (obs, next_obs)
		if self.idx >= len(self._obses):
			self._obses.append(obses)
		else:
			self._obses[self.idx] = (obses)
		np.copyto(self.actions[self.idx], action)
		np.copyto(self.rewards[self.idx], reward)
		np.copyto(self.not_dones[self.idx], not done)

		self.idx = (self.idx + 1) % self.capacity
		self.full = self.full or self.idx == 0

	def _get_idxs(self, n=None):
		if n is None:
			n = self.batch_size
		return np.random.randint(
			0, self.capacity if self.full else self.idx, size=n
		)

	def _encode_obses(self, idxs):
		obses, next_obses = [], []
		for i in idxs:
			obs, next_obs = self._obses[i]
			obses.append(np.array(obs, copy=False))
			next_obses.append(np.array(next_obs, copy=False))
		return np.array(obses), np.array(next_obses)

	def sample_soda(self, n=None):
		idxs = self._get_idxs(n)
		obs, _ = self._encode_obses(idxs)
		return torch.as_tensor(obs).cuda().float()

	def sample_curl(self, n=None):
		idxs = self._get_idxs(n)

		obs, next_obs = self._encode_obses(idxs)
		obs = torch.as_tensor(obs).cuda().float()
		next_obs = torch.as_tensor(next_obs).cuda().float()
		actions = torch.as_tensor(self.actions[idxs]).cuda()
		rewards = torch.as_tensor(self.rewards[idxs]).cuda()
		not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()

		pos = augmentations.random_crop(obs.clone())
		obs = augmentations.random_crop(obs)
		next_obs = augmentations.random_crop(next_obs)

		return obs, actions, rewards, next_obs, not_dones, pos

	def sample_drq(self, n=None, pad=4):
		idxs = self._get_idxs(n)

		obs, next_obs = self._encode_obses(idxs)
		obs = torch.as_tensor(obs).cuda().float()
		next_obs = torch.as_tensor(next_obs).cuda().float()
		actions = torch.as_tensor(self.actions[idxs]).cuda()
		rewards = torch.as_tensor(self.rewards[idxs]).cuda()
		not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()

		obs = augmentations.random_shift(obs, pad)
		next_obs = augmentations.random_shift(next_obs, pad)

		return obs, actions, rewards, next_obs, not_dones
	
	def sample_sacai(self, n=None, pad=4):
		idxs = self._get_idxs(n)

		obs, next_obs = self._encode_obses(idxs)
		obs = torch.as_tensor(obs).cuda().float()
		next_obs = torch.as_tensor(next_obs).cuda().float()
		actions = torch.as_tensor(self.actions[idxs]).cuda()
		rewards = torch.as_tensor(self.rewards[idxs]).cuda()
		not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()


		return obs, actions, rewards, next_obs, not_dones

	def sample(self, n=None):
		idxs = self._get_idxs(n)

		obs, next_obs = self._encode_obses(idxs)
		obs = torch.as_tensor(obs).cuda().float()
		next_obs = torch.as_tensor(next_obs).cuda().float()
		actions = torch.as_tensor(self.actions[idxs]).cuda()
		rewards = torch.as_tensor(self.rewards[idxs]).cuda()
		not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()

		obs = augmentations.random_crop(obs)
		next_obs = augmentations.random_crop(next_obs)

		return obs, actions, rewards, next_obs, not_dones


class LazyFrames(object):
	def __init__(self, frames, extremely_lazy=True):
		self._frames = frames
		self._extremely_lazy = extremely_lazy
		self._out = None

	@property
	def frames(self):
		return self._frames

	def _force(self):
		if self._extremely_lazy:
			return np.concatenate(self._frames, axis=0)
		if self._out is None:
			self._out = np.concatenate(self._frames, axis=0)
			self._frames = None
		return self._out

	def __array__(self, dtype=None):
		out = self._force()
		if dtype is not None:
			out = out.astype(dtype)
		return out

	def __len__(self):
		if self._extremely_lazy:
			return len(self._frames)
		return len(self._force())

	def __getitem__(self, i):
		return self._force()[i]

	def count(self):
		if self.extremely_lazy:
			return len(self._frames)
		frames = self._force()
		return frames.shape[0]//3

	def frame(self, i):
		return self._force()[i*3:(i+1)*3]


def count_parameters(net, as_int=False):
	"""Returns total number of params in a network"""
	count = sum(p.numel() for p in net.parameters())
	if as_int:
		return count
	return f'{count:,}'



#SRM algorithm
def random_mask_freq_v1(x):
	p = random.uniform(0, 1)
	if p > 0.5:
		return x
	# need to adjust r1 r2 and delta for best performance
	r1=random.uniform(0,0.5)
	delta_r=random.uniform(0,0.035)
	r2=np.min((r1+delta_r,0.5))
	# print(r2)
	# generate Mask M
	B,C,H,W = x.shape
	center = (int(H/2), int(W/2))
	diagonal_lenth = max(H,W) # np.sqrt(H**2+W**2) is also ok, use a smaller r1
	r1_pix = diagonal_lenth * r1
	r2_pix = diagonal_lenth * r2
	Y_coord, X_coord = np.ogrid[:H, :W]
	dist_from_center = np.sqrt((Y_coord - center[0])**2 + (X_coord - center[1])**2)
	M = dist_from_center <= r2_pix
	M = M * (dist_from_center >= r1_pix)
	M = ~M

	# mask Fourier spectrum
	M = torch.from_numpy(M).float().to(x.device)
	srm_out = torch.zeros_like(x)
	for i in range(C):
		x_c = x[:,i,:,:]
		x_spectrum = torch.fft.fftn(x_c, dim=(-2,-1))
		x_spectrum = torch.fft.fftshift(x_spectrum, dim=(-2,-1))
		out_spectrum = x_spectrum * M
		out_spectrum = torch.fft.ifftshift(out_spectrum, dim=(-2,-1))
		srm_out[:,i,:,:] = torch.fft.ifftn(out_spectrum, dim=(-2,-1)).float()
	return srm_out


def random_mask_freq_v2(x):
    p = random.uniform(0, 1)
    if p > 0.5:
        return False,x,x,x

    # dynamicly select freq range to erase
    A = 0
    B = 0.5
    a = random.uniform(A, B)
    C = 2
    freq_limit_low = round(a, C)

    A = 0
    B = 0.05
    a = random.uniform(A, B)
    C = 2
    diff = round(a, C)
    freq_limit_hi = freq_limit_low + diff

    # b, 9, h, w
    b, c, h, w = x.shape
    x0, x1, x2 = torch.chunk(x, 3, dim=1)
    # b, 3, 3, h, w
    x = torch.cat((x0.unsqueeze(1), x1.unsqueeze(1), x2.unsqueeze(1)), dim=1)

    pass1 = torch.abs(torch.fft.fftfreq(x.shape[-1], device=x.device)) < freq_limit_hi
    pass2 = torch.abs(torch.fft.fftfreq(x.shape[-2], device=x.device)) < freq_limit_hi
    kernel1 = torch.outer(pass2, pass1)  # freq_limit_hi square is true

    pass1 = torch.abs(torch.fft.fftfreq(x.shape[-1], device=x.device)) < freq_limit_low
    pass2 = torch.abs(torch.fft.fftfreq(x.shape[-2], device=x.device)) < freq_limit_low
    kernel2 = torch.outer(pass2, pass1)  # freq_limit_low square is true

    kernel = kernel1 * (~kernel2)  # a square ring is true
    fft_1 = torch.fft.fftn(x, dim=(2, 3, 4))
	
    fft_image = fft_1.view(b, 9, h, w)	#fft image
    imgs = torch.fft.ifftn(fft_1 * (~kernel), dim=(2, 3, 4)).float()

    srm_image = imgs.view(b,9,h,w)	#srm_image
    x0, x1, x2 = torch.chunk(imgs, 3, dim=1)
    imgs = torch.cat((x0.squeeze(1), x1.squeeze(1), x2.squeeze(1)), dim=1)

    return True,fft_image,srm_image,imgs
