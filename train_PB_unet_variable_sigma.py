import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps
import pytorch_lightning as pl
import torch
from scipy.ndimage import gaussian_filter
from sklearn.metrics import mean_squared_error
import os
import pandas as pd
import skimage.measure as measure
import scipy
from skimage.morphology import disk
import torchvision

import torch.nn as nn

import ufl
import dolfin as dlf
import dolfin_adjoint as d_ad
import pyadjoint as pyad

from torchvision import transforms

import warnings
import time
warnings.filterwarnings('ignore')
from tqdm.autonotebook import tqdm  


from utils.nn import ConvNextCell, DownsampleLayer
from utils.PB_utils_vary_sigma import Dataset, PB_Unet
import utils.proc as proc


pl.seed_everything(0)
np.random.seed(0)


from argparse import ArgumentParser


if __name__=='__main__':
    ap = ArgumentParser()
    ap.add_argument('--train_cells', type=str, help='comma delimited string of cell names')
    ap.add_argument('--data_root', type=str, default="/project/vitelli/cell_stress/MeshDatasets/imrescale-4_maxcellvol-20/")
    ap.add_argument('--data_max_include', type=int, default=160)
    
    ap.add_argument('--pde_epochs', type=int, default=301)
    ap.add_argument('--epochs', type=int, default=301)
    ap.add_argument('--batch_size', type=int, default=0)
    ap.add_argument('--logdir', type=str, default=0)
    
    ap.add_argument('--N_layers', type=int, default=0)
    ap.add_argument('--kernel', type=int, default=15)
    ap.add_argument('--LR', type=float, default=1e-3)
    ap.add_argument('--PDE_LR', type=float, default=1e-3)
    ap.add_argument('--gamma', type=float, default=0.995)
    ap.add_argument('--hidden_channels', type=int, default=32)
    
    args = ap.parse_args()
    print(args)
    

    dataset = Dataset(root=args.data_root, 
                      cells_to_include=args.train_cells.split(','), 
                      frames_to_include = np.arange(0,120,3),#[0,39,79,119], # np.arange(2) 
                      n_include=args.data_max_include, indices=args.data_max_include)
    
    print("DF cells:", dataset.df.cell.unique(), "\t DF len:", len(dataset.df))
    
    
    # Seed before making model for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    
    model = PB_Unet( 
                     kernel_size=args.kernel,
                     LR=args.LR,
                     N_layers=args.N_layers,
                     N_latent_layers=args.N_layers,
                     hidden_channels=args.hidden_channels,
                     gamma=args.gamma,
                     regularization=None,
                     batch_size=args.batch_size,
                     logdir=args.logdir)

    model.to(torch.device('cuda:0'))

    ### PRETRAIN ## 
    sig_init = 1e-1
    a_init = 1e-1
    d_ad.set_working_tape(d_ad.Tape())
    #d_ad.get_working_tape().get_blocks()[:] = []
    #print(len(d_ad.get_working_tape().get_blocks()))

    t0 = time.time()
    for i in range(args.epochs):
        for s in range(len(dataset)):
            m = i*len(dataset) + s

            sample = dataset[s]
            sample['zyx'] = sample['zyx'].to(torch.device('cuda:0'))
            sample['Y_init'] = sample['Y_init'].to(torch.device('cuda:0'))
            sample['sig_init'] = sample['sig_init'].to(torch.device('cuda:0'))
            sample['mask_bool'] = sample['mask_bool'].to(torch.device('cuda:0'))

            loss, y_im, sig_im, alpha = model.pretrain(sample['zyx'], 1e2*sample['Y_init'], sample['sig_init'], a_init, mask=sample['mask_bool'])

            if torch.isnan(loss):
                raise ValueError(f"Nans encountered, iter {m}")

            model.logger.add_scalar('Loss', loss, global_step=m) 
            model.logger.add_scalar('LR', model.opt.param_groups[0]['lr'], global_step=m) 
            
        #model.logger.add_scalar('Sig', sig_a, global_step=i) 
        model.logger.add_scalar('Alpha', alpha, global_step=i) 
        model.plot_Y(i, sample['zyx'].detach().cpu().numpy().squeeze(), y_im.squeeze()) 
        model.plot_sig(i, sample['zyx'].detach().cpu().numpy().squeeze(), sig_im.squeeze()) 

        
    #### 
    for g in range(len(model.opt.param_groups)):
        model.opt.param_groups[g]['lr'] = args.PDE_LR

    for i in range(args.pde_epochs):
        for s in range(len(dataset)):
            m = i*len(dataset) + s
            sample = dataset[s]
            sample['zyx'] = sample['zyx'].to(torch.device('cuda:0'))

            constants, y_im, _, sig_im, _, J, _ = model.step(sample['zyx'], sample['Jhat'], len(sample['control_arr']), sample['Y_FctSpace'])

            model.logger.add_scalar('PDE_Loss', J, global_step=m) 
            model.logger.add_scalar('PDE_LR', model.opt.param_groups[0]['lr'], global_step=m) 

        #model.logger.add_scalar('Sig', constants[0].detach().cpu().numpy(), global_step=i+args.epochs) 
        model.logger.add_scalar('Alpha', constants[0].detach().cpu().numpy(), global_step=i+args.epochs) 
        model.plot_Y(i+args.epochs, sample['zyx'].detach().cpu().numpy().squeeze(), y_im.detach().cpu().numpy().squeeze()) 
        model.plot_sig(i+args.epochs, sample['zyx'].detach().cpu().numpy().squeeze(), sig_im.detach().cpu().numpy().squeeze()) 
        
        torch.save(model.state_dict(), os.path.join(args.logdir, './model.pt'))

                
