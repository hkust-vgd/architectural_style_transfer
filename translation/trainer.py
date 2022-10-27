"""
Copyright (C) 2022  HKUST VGD Group
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os

import torch
import torch.nn as nn
from torch.autograd import Variable

from networks import AdaINGen, MsImageDis, ContentEncoder
from utils import weights_init, get_model_list, get_scheduler
import torch.nn.functional as F


class DOT_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(DOT_Trainer, self).__init__()
        # Initiate the networks
        mid_downsample = hyperparameters['gen'].get('mid_downsample', 1)
        self.content_enc = ContentEncoder(hyperparameters['gen']['n_downsample'],
                                        mid_downsample,
                                        hyperparameters['gen']['n_res'],
                                        hyperparameters['input_dim_a'],
                                        hyperparameters['gen']['dim'],
                                        'in',
                                        hyperparameters['gen']['activ'],
                                        pad_type=hyperparameters['gen']['pad_type'])

        self.style_dim = hyperparameters['gen']['style_dim']
        self.gen_a = AdaINGen(hyperparameters['input_dim_a'], self.content_enc, 'a', hyperparameters['gen']) # auto-encoder for domain a
        self.gen_b = AdaINGen(hyperparameters['input_dim_b'], self.content_enc, 'b', hyperparameters['gen'])  # auto-encoder for domain b
        self.dis_a = MsImageDis(hyperparameters['input_dim_a'], self.content_enc.output_dim, hyperparameters['dis'])  # discriminator for domain a
        self.dis_b = MsImageDis(hyperparameters['input_dim_b'], self.content_enc.output_dim, hyperparameters['dis'])  # discriminator for domain b

    def build_optimizer(self, hyperparameters):
        # Setup the optimizers
        lr = hyperparameters['lr']
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))


    def recon_criterion(self, input, target):
        '''
            Construction loss: 
        '''
        return torch.mean(torch.abs(input - target))

    # Input and target have different dimensions, 1D mean subtraction
    def mean_recon_criterion(self, input, target):
        '''
            Mean Construction loss: 
        '''
        return torch.abs(torch.mean(input) - torch.mean(target))

    # Luminance Gradient Loss
    def gradient_loss(self, gen_frames, gt_frames, alpha=1): 
        '''
            Luminance Gradient Loss:
            Paper Eq.(1)
            Here applies horizontal and vertial gradients
            More complex gradient algorithms should works
        '''

        def gradient(x):
            ''' x [B,C,H,W], float32 or float64
                dx, dy: [B,C,H,W] 
                return average loss [0,2]
            ''' 
            x =  x[:,0:1,:,:] # only calculate Luminance channel
            # gradient step=1
            left = x
            right = F.pad(x, [0, 1, 0, 0])[ :,:, :, 1:]
            top = x
            bottom = F.pad(x, [0, 0, 0, 1])[ :,:, 1:, :]

            # dx, dy = torch.abs(right - left), torch.abs(bottom - top)
            dx, dy = right - left, bottom - top 
            # dx will always have zeros in the last column, right-left
            # dy will always have zeros in the last row,    bottom-top
            dx[:, :, :, -1] = 0
            dy[:, :, -1, :] = 0

            return dx, dy

        # gradient
        gen_dx, gen_dy = gradient(gen_frames)
        gt_dx, gt_dy = gradient(gt_frames)
        
        # loss
        grad_diff_x = torch.abs(gt_dx - gen_dx)
        grad_diff_y = torch.abs(gt_dy - gen_dy)

        # condense into one tensor and avg
        return torch.mean(grad_diff_x ** alpha + grad_diff_y ** alpha)

    # Luminance KL Loss
    def compute_lum_kl_loss(self, input_L, target_L):
        '''
            Luminance KL Loss:
            Paper Eq.(3)
            INPUTS: luminance
                input_L, target_L: [B, 1, H, W]
        '''
        num_batch = input_L.size()[0]
        input_L = input_L.reshape(num_batch, -1)
        target_L = target_L.reshape(num_batch, -1)

        # Get "probability" in Eq.(3)
        # This is a bit diffrent from description in Paper Eq.(3)
        # We do not do normalization but exponential normalization
        # It is found direct softmax operation can already get good performance
        # Softmax operation makes sum of probability 1 for each bach
        input_x = F.log_softmax(input_L, 1)
        target_y = F.softmax(target_L, 1)         

        kl_loss = F.kl_div(input_x, target_y, reduction='batchmean')

        return kl_loss

    # content geometry gradient loss, computation only involves Luminance channel
    def geometry_grad_criterion(self, predict, target, geo_grad_w):
        grad_loss = self.gradient_loss(predict, target) if geo_grad_w > 0 else 0
        return geo_grad_w * grad_loss 
        
    # content geometry kl loss, computation only involves Luminance channel
    def geometry_kl_criterion(self, predict, target, geo_kl_w):
        lum_kl_loss = self.compute_lum_kl_loss(predict[:,0:1,:,:], target[:,0:1,:,:]) if geo_kl_w > 0 else 0
        return geo_kl_w * lum_kl_loss

    def gen_update(self, x_a, x_b, hyperparameters, iterations):
        self.gen_opt.zero_grad()
        self.gen_backward_cc(x_a, x_b, hyperparameters)
       
        self.gen_backward_latent(x_a, x_b, hyperparameters)
        self.gen_opt.step()

        return self.loss_gen_total

    # latent code reconstruction
    def gen_backward_latent(self, x_a, x_b, hyperparameters):
        # random sample style vector and multimodal training
        s_a_random = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b_random = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        # decode
        x_ba_random = self.gen_a.decode(self.c_b, self.da_b, s_a_random)
        x_ab_random = self.gen_b.decode(self.c_a, self.db_a, s_b_random)

        c_b_random_recon, _, _, s_a_random_recon = self.gen_a.encode(x_ba_random)
        c_a_random_recon, _, _, s_b_random_recon = self.gen_b.encode(x_ab_random)

        # style reconstruction loss
        self.loss_gen_recon_s_a = self.recon_criterion(s_a_random, s_a_random_recon)
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_random, s_b_random_recon)
        loss_gen_recon_c_a = self.recon_criterion(self.c_a, c_a_random_recon)
        loss_gen_recon_c_b = self.recon_criterion(self.c_b, c_b_random_recon)
        loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba_random, x_a)
        loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab_random, x_b)

        loss_gen_total = hyperparameters['gan_w'] * loss_gen_adv_a + \
                          hyperparameters['gan_w'] * loss_gen_adv_b + \
                          hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a + \
                          hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b + \
                          hyperparameters['recon_c_w'] * loss_gen_recon_c_a + \
                          hyperparameters['recon_c_w'] * loss_gen_recon_c_b 

        self.loss_gen_total += loss_gen_total
        self.loss_gen_total.backward()

        self.loss_gen_total = self.loss_gen_total + loss_gen_total
        self.loss_gen_adv_a = self.loss_gen_adv_a + loss_gen_adv_a
        self.loss_gen_adv_b = self.loss_gen_adv_b + loss_gen_adv_b
        self.loss_gen_recon_c_a = self.loss_gen_recon_c_a + loss_gen_recon_c_a
        self.loss_gen_recon_c_b =  self.loss_gen_recon_c_b + loss_gen_recon_c_b

    # content and style reconstruction, geometry losses
    def gen_backward_cc(self, x_a, x_b, hyperparameters):
        # encode images, get content and style codes
        pre_c_a, self.c_a, c_domain_a, self.db_a, self.s_a_prime = self.gen_a.encode(x_a, training=True, flag=True)
        pre_c_b, self.c_b, c_domain_b, self.da_b, self.s_b_prime = self.gen_b.encode(x_b, training=True, flag=True)
       
        self.da_a = self.gen_b.domain_mapping(self.c_a, pre_c_a) # to content specific domain A
        self.db_b = self.gen_a.domain_mapping(self.c_b, pre_c_b) # to content specific domain B
      
        # decode (within domain): A->A, B->B
        x_a_recon = self.gen_a.decode(self.c_a, self.da_a, self.s_a_prime)
        x_b_recon = self.gen_b.decode(self.c_b, self.db_b, self.s_b_prime)

        # decode (cross domain): B->A, A->B
        x_ba = self.gen_a.decode(self.c_b, self.da_b, self.s_a_prime)
        x_ab = self.gen_b.decode(self.c_a, self.db_a, self.s_b_prime)

        ## B->A->B, A->B->A
        c_b_recon, _, self.db_b_recon, s_a_recon = self.gen_a.encode(x_ba, training=True)
        c_a_recon, _, self.da_a_recon, s_b_recon = self.gen_b.encode(x_ab, training=True)
        # decode again (for cycle-consistent loss)
        x_aba = self.gen_a.decode(c_a_recon, self.da_a_recon, s_a_recon )
        x_bab = self.gen_b.decode(c_b_recon, self.db_b_recon, s_b_recon )


        # image reconstruction loss (Xa->a, Xb->b)
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)


        # geometry loss (preserve geometry for foreground only)
        self.loss_gen_geo_x_a = self.geometry_grad_criterion(x_ab, x_a, hyperparameters['geo_grad_w'])
        self.loss_gen_geo_x_a = self.loss_gen_geo_x_a + self.geometry_kl_criterion(x_ab, x_a, hyperparameters['geo_kl_w']) 
        self.loss_gen_geo_x_b = self.geometry_grad_criterion(x_ba, x_b, hyperparameters['geo_grad_w'])
        self.loss_gen_geo_x_b = self.loss_gen_geo_x_b + self.geometry_kl_criterion(x_ba, x_b, hyperparameters['geo_kl_w']) 
        
        if hyperparameters['self_geo']: # Optional. If you want a sharper reconstruction output
            self.loss_gen_geo_x_a = self.loss_gen_geo_x_a + self.geometry_grad_criterion(x_a_recon, x_a, hyperparameters['geo_grad_w'])
            self.loss_gen_geo_x_a = self.loss_gen_geo_x_a + self.geometry_kl_criterion(x_a_recon, x_a, hyperparameters['geo_kl_w'])
            self.loss_gen_geo_x_b = self.loss_gen_geo_x_b + self.geometry_grad_criterion(x_b_recon, x_b, hyperparameters['geo_grad_w'])
            self.loss_gen_geo_x_b = self.loss_gen_geo_x_b + self.geometry_kl_criterion(x_b_recon, x_b, hyperparameters['geo_kl_w'])
              
        # domain-invariant content reconstruction loss (Ca->a, Cb->b, latent code loss)
        self.loss_gen_recon_c_a = self.recon_criterion(c_a_recon, self.c_a)
        self.loss_gen_recon_c_b = self.recon_criterion(c_b_recon, self.c_b)
        # domain-specific content reconstruction loss (Ca->a, Cb->b, latent code loss)
        self.loss_gen_recon_d_a = self.recon_criterion(c_domain_a, self.da_a) if hyperparameters['recon_d_w'] > 0 else 0
        self.loss_gen_recon_d_b = self.recon_criterion(c_domain_b, self.db_b) if hyperparameters['recon_d_w'] > 0 else 0

        # cycle consistenct loss (Xa->b->a, Xb->a->b)
        self.loss_gen_cycrecon_x_a = self.recon_criterion(x_aba, x_a)
        self.loss_gen_cycrecon_x_b = self.recon_criterion(x_bab, x_b)

        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba, x_a)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab, x_b)

        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_a + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_d_w'] * self.loss_gen_recon_d_a + \
                              hyperparameters['recon_d_w'] * self.loss_gen_recon_d_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b + \
                              self.loss_gen_geo_x_a + \
                              self.loss_gen_geo_x_b 

        # self.loss_gen_total.backward(retain_graph=True)


    def dis_update(self, x_a, x_b, hyperparameters, iterations):
        self.dis_opt.zero_grad()
        s_a_random = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b_random = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())

        # encode
        _, c_a, _, db_a, s_a = self.gen_a.encode(x_a, training=True, flag=True)
        _, c_b, _, da_b, s_b = self.gen_b.encode(x_b, training=True, flag=True)

        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, da_b, s_a)
        x_ab = self.gen_b.decode(c_a, db_a, s_b)
        # decode (cross domain)
        x_ba_random = self.gen_a.decode(c_b, da_b, s_a_random)
        x_ab_random = self.gen_b.decode(c_a, db_a, s_b_random)

        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a) + self.dis_a.calc_dis_loss(x_ba_random.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b) + self.dis_b.calc_dis_loss(x_ab_random.detach(), x_b)

        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + \
                              hyperparameters['gan_w'] * self.loss_dis_b
        self.loss_dis_total.backward()
        self.dis_opt.step()

        return self.loss_dis_total

    def sample(self, x_a, x_b):
        self.eval()
        x_a_recon, x_b_recon, x_ab, x_ba = [], [], [], []
        for i in range(x_a.size(0)):
            pre_c_a, c_a, _, db_a, s_a_fake = self.gen_a.encode(x_a[i].unsqueeze(0), flag=True)
            pre_c_b, c_b, _, da_b, s_b_fake = self.gen_b.encode(x_b[i].unsqueeze(0), flag=True)
            da_a = self.gen_b.domain_mapping(c_a, pre_c_a)
            db_b = self.gen_a.domain_mapping(c_b, pre_c_b)

            x_a_recon.append(self.gen_a.decode(c_a, da_a, s_a_fake))
            x_b_recon.append(self.gen_b.decode(c_b, db_b, s_b_fake))
            x_ba.append(self.gen_a.decode(c_b, da_b, s_a_fake))
            x_ab.append(self.gen_b.decode(c_a, db_a, s_b_fake))

        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ab, x_ba = torch.cat(x_ab), torch.cat(x_ba)

        self.train()

        return x_a, x_a_recon, x_ab, x_b, x_b_recon, x_ba

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()
    
    # Please follow the same naming convention, such that iteration information can be correctly read
    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_b.load_state_dict(state_dict['b'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    # Please follow the same naming convention, such that iteration information can be correctly read 
    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)

    def get_z_random(self, image):
        # random sample style vector 
        z_random = Variable(torch.randn(image.size(0), self.style_dim, 1, 1).cuda())

        return z_random
