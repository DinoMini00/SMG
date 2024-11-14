import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import utils
from utils import random_mask_freq_v2, random_mask_freq_v1
import algorithms.modules as m
from algorithms.drq import DrQ

from .rl_utils import (
    compute_attribution,
    compute_attribution_mask,
    make_attribution_pred_grid,
    make_obs_grid,
    make_obs_grad_grid,
)

class SRM(DrQ):
    def __init__(self, obs_shape, action_shape, args, writer):
        super().__init__(obs_shape, action_shape, args, writer)
        self.writer = writer

    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample_drq()

        # SRM data agument
        is_srm,fft_obs,srm_obs,final_obs = random_mask_freq_v2(obs)
        fft_obs_grad = make_obs_grid(fft_obs)
        srm_obs_grad = make_obs_grid(srm_obs)
        final_obs_grad = make_obs_grid(final_obs)
        
        self.update_critic(final_obs, action, reward, next_obs, not_done, L, step)

        if step % 10000 == 0 and is_srm == True:
            self.writer.add_image("/fft_obs", fft_obs_grad, global_step=step)
            self.writer.add_image("/srm_obs", srm_obs_grad, global_step=step)
            self.writer.add_image("/final_obs", final_obs_grad, global_step=step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(final_obs, L, step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()