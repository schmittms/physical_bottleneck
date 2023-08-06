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
from utils.PB_utilsfixed_Uscale import Dataset, PB_Unet
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


    ap.add_argument('--log_Y_init', type=float, default=1e-3)
    ap.add_argument('--log_sig_init', type=float, default=1e-3)
    ap.add_argument('--E_init', type=float, default=1e-3)
    ap.add_argument('--poisson_init', type=float, default=1e-3)
    
    args = ap.parse_args()
    print(args)
    

    dataset = Dataset(root=args.data_root, 
                      Y_init_scale=np.exp(args.log_Y_init),
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
    sig_init = np.exp(args.log_sig_init)
    E_init = args.E_init
    poisson_init = args.poisson_init
    
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
            sample['mask_bool'] = sample['mask_bool'].to(torch.device('cuda:0'))

            loss, y_im, sig_a, E, poisson = model.pretrain(sample['zyx'], sample['Y_init'], sig_init, E_init, poisson_init, mask=sample['mask_bool'])

            if torch.isnan(loss):
                raise ValueError(f"Nans encountered, iter {m}")

            model.logger.add_scalar('Loss', loss, global_step=m) 
            model.logger.add_scalar('LR', model.opt.param_groups[0]['lr'], global_step=m) 
            
        model.logger.add_scalar('Sig', np.exp(sig_a), global_step=i) 
        model.logger.add_scalar('E_cell', np.exp(E), global_step=i) 
        model.logger.add_scalar('poisson', 0.75*np.tanh(poisson)-0.25, global_step=i) 
        model.plot_Y(i, sample['zyx'].detach().cpu().numpy().squeeze(), y_im.squeeze()) 

        
    #### 
    for g in range(len(model.opt.param_groups)):
        model.opt.param_groups[g]['lr'] = args.PDE_LR

    for i in range(args.pde_epochs):
        for s in range(len(dataset)):
            m = i*len(dataset) + s
            sample = dataset[s]
            sample['zyx'] = sample['zyx'].to(torch.device('cuda:0'))

            constants, y_im, y_mesh_vals, J, _ = model.step(sample['zyx'], sample['Jhat'], len(sample['control_arr']), sample['Y_FctSpace'])

            model.logger.add_scalar('PDE_Loss', J, global_step=m) 
            model.logger.add_scalar('PDE_LR', model.opt.param_groups[0]['lr'], global_step=m) 

        mesh = sample['Y_FctSpace'].mesh()
        sigma_a = d_ad.Constant(constants[0].exp().detach().cpu().numpy(), name = "sigma")
        E = d_ad.Constant(constants[1].exp().detach().cpu().numpy(), name = "E")
        poisson = d_ad.Constant(0.75*np.tanh(constants[2].detach().cpu().numpy()) - 0.25, name = "poisson")

        #### Generate Y, F fields on mesh ####
        Y_mesh = d_ad.Function(sample['Y_FctSpace'])
        F_mesh = proc.vector_img_to_mesh(sample['F'].squeeze().detach().numpy(), dlf.VectorFunctionSpace(mesh, 'CG', 1, dim=2), mesh_inv_scale=1/(0.68*1e-6))
        #F_mesh = proc.vector_img_to_mesh(F_img*1000, F_FctSpace, mesh_inv_scale=1/(0.68*1e-6)) # Now in units of N/m^2

        Y_mesh.vector()[:] = y_mesh_vals.detach().cpu().numpy()
        u,_,_ = sample['pde_forward'](Y_mesh, sigma_a, E, poisson, mesh) 

        model.logger.add_scalar('Sig', constants[0].detach().exp().cpu().numpy(), global_step=i+args.epochs) 
        model.logger.add_scalar('E_cell', constants[1].detach().exp().cpu().numpy(), global_step=i+args.epochs) 
        model.logger.add_scalar('poisson', 0.75*np.tanh(constants[2].detach().cpu().numpy())-0.25, global_step=i+args.epochs) 
        model.plot_Y(i+args.epochs, sample['zyx'].detach().cpu().numpy().squeeze(), y_im.detach().cpu().numpy().squeeze()) 
        model.plot_F(i+args.epochs, F_mesh, Y_mesh, u, mesh) 
        
        torch.save(model.state_dict(), os.path.join(args.logdir, './model.pt'))

                
