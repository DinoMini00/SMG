import os
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import utils
import copy
import augmentations
import algorithms.modules as m
from algorithms.sac import SAC

from .rl_utils import (
    compute_attribution,
    compute_attribution_mask,
    make_attribution_pred_grid,
    make_obs_grid,
    make_obs_grad_grid,
)
import random


class SMG(SAC):
    def __init__(self, obs_shape, action_shape, args, writer):
        super().__init__(obs_shape, action_shape, args, writer)

        self.writer = writer
        self.aux_update_freq = args.aux_update_freq
        self.consistency = args.consistency
        self.consistency_scale = args.consistency_scale
        self.frame_stack = args.frame_stack
        self.back_scale = args.back_scale
        self.recon_scale = args.recon_scale
        self.mask_scale = args.mask_scale
        self.percentage = args.mask_percentage
        self.fore_scale = args.fore_scale
        self.action_scale = args.action_scale
        self.grad_clip_norm = args.grad_clip_norm
        self.image_loss = torch.nn.L1Loss()

        # ! models - aux_head
        emb_dim = self.critic.encoder.out_dim
        fore_head = m.AuxHead(emb_dim)

        # ! models - fore
        self.fore_encoder = m.AuxEncoder(self.critic.encoder, fore_head)

        # ! models - back
        self.back_encoder = copy.deepcopy(self.fore_encoder)

        # ! models - full
        self.predictor = m.CoPredictor(self.fore_encoder, self.back_encoder, action_shape, batch_size=64, emb_dim=emb_dim).cuda()
        self.model_params = list(self.predictor.parameters())

        # ! optimizers
        self.aux_optimizer = torch.optim.Adam(
            self.model_params,
            lr=1e-3,
            betas=(args.aux_beta, 0.999),
        )


    def update_critic(self, obs, action, reward, next_obs, not_done, consistency_aug, L=None, step=None):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        current_Q1, current_Q2 = self.critic(obs, action)

        critic_loss = F.mse_loss(current_Q1, target_Q) + \
                      F.mse_loss(current_Q2, target_Q)
        
        if self.consistency_scale:
            aug_Q1, aug_Q2 = self.critic(consistency_aug, action)
            consistency_loss =  F.mse_loss(aug_Q1, current_Q1.detach()) + \
                                F.mse_loss(aug_Q2, current_Q2.detach())
            critic_loss += self.consistency_scale*consistency_loss

        if L is not None:
            if self.consistency_scale:
                L.log("train_critic/consistency_loss", consistency_loss, step)
            L.log("train_critic/loss", critic_loss, step)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


    def update_aux(self, obs, action, next_obs, attrib_aug, gt_back, consistency_aug, step=None, L=None):
        loss = 0

        # ! recon
        if self.recon_scale != 0:
            fore, back, recon_mask, recon_img, recon_loss \
                = self.image_recon_loss(obs, obs)
            loss += recon_loss * self.recon_scale
    
        # ! mask
        if self.mask_scale != 0:
            mask_loss = self.mask_loss(recon_mask, self.percentage)
            if step > 1000:
                loss += mask_loss * self.mask_scale
        
        # ! back
        if self.back_scale != 0:
            pred_back, back_recon_loss \
                = self.back_recon_loss(attrib_aug, gt_back)
            loss += back_recon_loss * self.back_scale
        
        # ! inverse dynamic
        if self.action_scale != 0:
            action_loss = self.action_loss(obs, next_obs, action)
            loss += action_loss * self.action_scale
        
        # ! fore
        if self.fore_scale != 0:
            pred_fore, fore_loss \
                = self.fore_loss(consistency_aug, (fore*recon_mask).detach())
            loss += fore_loss * self.fore_scale

        self.aux_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model_params, self.grad_clip_norm)
        self.aux_optimizer.step()

        if L is not None:
            if self.back_scale != 0:
                L.log("train/back_recon_loss", back_recon_loss, step)
            if self.recon_scale != 0:
                L.log("train/recon_loss", recon_loss, step)
            if self.mask_scale != 0:
                L.log("train/mask_loss", mask_loss, step)
            if self.fore_scale != 0:
                L.log("train/fore_loss", fore_loss, step)
            if self.action_scale != 0:
                L.log("train/action_loss", action_loss, step)
            L.log("train/aux_loss", loss, step)

        if step % 1000 == 0:
            self.log_image(obs, step, "obs/original")
            self.log_image(attrib_aug, step, "obs/attrib_aug")
            self.log_image(consistency_aug, step, "obs/consistency_aug")

            if self.back_scale != 0:
                self.log_image(gt_back, step, "background/ground_truth/")
                self.log_image(pred_back, step, "background/prediction/")
            
            if self.fore_scale != 0:
                self.log_image(fore*recon_mask, step, "foreground/ground_truth")
                self.log_image(pred_fore, step, "foreground/prediction")

            if self.recon_scale != 0:
                self.log_image(fore, step, "reconstruction/foreground")
                self.log_image(back, step, "reconstruction/background")
                self.log_image(recon_mask*255, step, "reconstruction/mask")
                self.log_image(recon_img, step, "reconstruction/full")


    def back_recon_loss(self, obs, gt):
        pred_back = self.predictor.back(obs)
        back_recon_loss = self.image_loss(pred_back, gt)

        return pred_back, back_recon_loss
    

    def action_loss(self, obs, next_obs, action):
        pred_action = self.predictor.action(obs, next_obs)
        action_loss = F.mse_loss(pred_action, action)
        return action_loss
    
 
    def image_recon_loss(self, obs, gt):
        fore, back, recon_mask, recon_img = self.predictor(obs)
        recon_loss = self.image_loss(recon_img, gt)
        return fore, back, recon_mask, recon_img, recon_loss
    

    def fore_loss(self, obs, gt):
        pred_fore = self.predictor.fore(obs)
        fore_loss = self.image_loss(pred_fore, gt)

        return pred_fore, fore_loss
    

    def mask_loss(self, mask, gt):
        quantile = torch.sum(mask, dim=(-1, -2))/(84*84)
        mask_loss = torch.mean(torch.square(quantile - gt))
        return mask_loss


    def log_image(self, img, step, title):
        grid = make_obs_grid(img, self.frame_stack)
        self.writer.add_image(title, grid, global_step=step)
    

    def consistency_aug(self, obs, attrib_obs):
        rd = random.randint(0, 3)
        if rd == 0:
            return attrib_obs
        else:
            return augmentations.random_overlay(obs)
    
    def load(self, model_dir, step):
        self.actor.load_state_dict(torch.load(os.path.join(model_dir, f"actor_{step}.pt")))
        self.critic.load_state_dict(torch.load(os.path.join(model_dir, f"critic_{step}.pt")))
        self.predictor.load_state_dict(torch.load(os.path.join(model_dir, f"predictor_{step}.pt")))

    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample_drq()

        mask = self.predictor.mask(obs).detach()
        attrib_aug, gt_fore, gt_back = augmentations.attribution_augmentation(obs, mask)
        consistency_aug = self.consistency_aug(obs, attrib_aug)

        self.update_critic(obs, action, reward, next_obs, not_done, consistency_aug, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()

        if step % self.aux_update_freq == 0:
            self.update_aux(obs, action, next_obs, attrib_aug, gt_back, consistency_aug, step, L)