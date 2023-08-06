import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import scipy

import ufl
import dolfin as dlf
import dolfin_adjoint as d_ad
import pyadjoint as pyad

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

import warnings
import time
warnings.filterwarnings('ignore')

from utils.nn import ConvNextCell, DownsampleLayer
import utils.proc as proc

pl.seed_everything(1)
np.random.seed(1)

from torch.utils.tensorboard import SummaryWriter



class Dataset(torch.utils.data.Dataset):
    def __init__(self, 
                 root,
                 indices=[0],
                 Y_init_scale=1e-3,
                 use_F_NN_or_Exp='NN',
                 n_include = None,
                 cells_to_include = None,
                 frames_to_include = None,
                 transform=None,
                 device='cpu'):
        
        self.use_F_NN_or_Exp = use_F_NN_or_Exp
        self.Y_init_scale = Y_init_scale

        self.generate_dataframe(root, cells_to_include=cells_to_include, n_include=n_include, frames_to_include=frames_to_include)
        self.prune_dataframe(cells_to_include=cells_to_include, frames_to_include=frames_to_include)
        self.form_reduced_dataset(indices=indices)

        #self.keys = list(self.data.keys())

        #self.transform = transforms.Compose(transform_list)
        
        #self.device=device
        #self.data_file = data_file
        
        
    def generate_dataframe(self, root, n_include=None, cells_to_include=None, frames_to_include=np.arange(200)):
        files = os.listdir(root)
        cells = np.unique([f.split('-')[0] for f in files])
        self.all_cells = cells
        cells = [cell for cell in cells if cell in cells_to_include]
        frames = {cell: np.unique([cell+'-'+str(int(f.split('-')[-1].split('.')[0])) for f in files 
                                   if cell in f and int(f.split('-')[-1].split('.')[0]) in frames_to_include])
                  for cell in cells}

        df = pd.DataFrame(columns=['cell', 'frame', 'mesh_file', 'npy_file', 'mesh_len'])
        idx = 0
        for cell in cells:
            for frame in frames[cell]:
                frame = int(frame.split('-')[-1])
                df.loc[idx] = [cell, 
                                frame, 
                                os.path.join(root, f'{cell}-{frame}.xml'),
                                os.path.join(root, f'{cell}-{frame}.npy'),
                                dlf.Mesh(os.path.join(root, f'{cell}-{frame}.xml')).coordinates().shape[0]]
                assert os.path.exists(df.loc[idx, 'mesh_file'])
                assert os.path.exists(df.loc[idx, 'npy_file'])

                idx += 1
        self.df = df[df.mesh_len >= 500].reset_index()
        return
        #self.loader = self.get_loader()
        
    def prune_dataframe(self, cells_to_include=None, frames_to_include=None):
        if cells_to_include is not None: self.df = self.df[self.df.cell.isin(cells_to_include) == True]
        if frames_to_include is not None: self.df = self.df[self.df.frame.isin(frames_to_include) == True]
        return
        
    def get_loader(self, indices, batch_size, num_workers, pin_memory=True):        
        sampler = SubsetRandomSampler(indices)
        loader = torch.utils.data.DataLoader(self, 
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            pin_memory=pin_memory)
        return loader
    
    def form_reduced_dataset(self, indices):
        if indices is None: indices = np.arange(len(self.df))
        elif isinstance(indices, int): indices = np.random.choice(np.arange(len(self.df)), 
                                                                np.minimum(indices,len(self.df)) , replace=False)
        
        self.red_dataset = []
        for idx in indices:
            x = np.load(self.df.loc[idx, 'npy_file'])
            mesh = d_ad.Mesh( self.df.loc[idx, 'mesh_file'])
            
            frame = self.df.loc[idx, 'frame']
            
            mask = x[0]
            mask_grad = x[1]
            F_NN = x[[2,3]]
            F_exp= np.stack([x[4]*np.cos(x[5]), x[4]*np.sin(x[5])]) # shape (2, H, W)
            if self.use_F_NN_or_Exp=='NN':
                F = F_NN
            elif self.use_F_NN_or_Exp=='exp':
                F = F_exp
            else:
                raise ValueError(f'use_F_NN_or_Exp is {self.use_F_NN_or_Exp}, should be in in [NN, exp]')
            zyx = x[6]

            F[:, mask==0] = 0 #np.nan
            Y_init = np.linalg.norm(F, axis=0)
            Y_init = Y_init*self.Y_init_scale/Y_init[~np.isnan(Y_init)].max() + 0.0001 # Careful. Terrible abuse of notation Y_init=const, Yinit=image


            F_torch = torch.tensor(F[None,:,:,:], device='cpu', dtype=torch.float)
            zyx_torch = torch.tensor(zyx[None,None,:,:], device='cpu', dtype=torch.float)
            Y_init_torch = torch.tensor(Y_init[None,:,:], device='cpu', dtype=torch.float)
            mask_bool_torch = torch.tensor(mask[None,:,:], device='cpu', dtype=torch.bool)

            Jhat_np, control_arr, pde_forward, Y_FctSpace, Jhat, J = self.init_dolfin(Y_init, F, mesh, element_degree=1)
            self.red_dataset.append({'zyx': zyx_torch, 
                       'F': F_torch, 
                       'Y_init': Y_init_torch, 
                       'mask_bool': mask_bool_torch,
                       'Jhat': Jhat_np,
                       'control_arr': control_arr,
                       'pde_forward': pde_forward,
                       'Y_FctSpace': Y_FctSpace,
                       'base_RF': Jhat,
                        'J': J,
                        'frame': frame})
            
        return
            
            
            
    def init_dolfin(self, Y_init, F_img, mesh, element_degree=1):
        """
        Generates the reduced functional which can be used to calculate gradients dJdY
        """
        #mesh = dlf.Mesh(meshfile)
        #mesh = d_ad.UnitSquareMesh(10, 10)

        def epsilon(u):
            return 0.5*(ufl.grad(u) + ufl.grad(u).T)

        def sigma(u, alpha):
            return alpha*ufl.div(u)*ufl.Identity(2) + epsilon(u)

        el = ufl.VectorElement("CG", mesh.ufl_cell(), element_degree)
        V = dlf.FunctionSpace(mesh, el)

        def forward(Y, sigma_a, alpha, mesh):
            """Solve the forward problem for a given material distribution a(x)."""

            # Define variational problem
            u = dlf.TrialFunction(V)
            v = dlf.TestFunction(V)
            n = dlf.FacetNormal(mesh)

            a = ufl.inner(sigma(u, alpha), epsilon(v))*ufl.dx + ufl.dot(Y*u,v)*ufl.dx 
            L = ufl.dot(-sigma_a*n, v)*ufl.ds 

            # Compute solution
            u = d_ad.Function(V, name="Sol")
            d_ad.solve(a == L, u)

            return u, a, L

        #self.pde_forward = forward

        sigma_a = d_ad.Constant(1., name = "sigma")
        alpha = d_ad.Constant(0.1, name = "alpha")
        
        Y_FctSpace = dlf.FunctionSpace(mesh, "CG", 1)    
        F_FctSpace = dlf.VectorFunctionSpace(mesh, 'CG', 1, dim=2)

        #### Generate Y, F fields on mesh ####
        Y_mesh = proc.scalar_img_to_mesh(Y_init, Y_FctSpace)
        F_mesh = proc.vector_img_to_mesh(F_img, F_FctSpace)

        u, a, L = forward(Y_mesh, sigma_a, alpha, mesh)   
        
        J = d_ad.assemble(ufl.dot(Y_mesh*u - F_mesh, Y_mesh*u - F_mesh)*ufl.dx)
                             
        controlY = d_ad.Control(Y_mesh) # Control has functions self.update(value), and get_derivative, which returns self.block_variable.adj_value
        controlsig = d_ad.Control(sigma_a)
        controla = d_ad.Control(alpha)

        Jhat_np = pyad.ReducedFunctionalNumPy(J, [controlY, controlsig, controla])
        Jhat = pyad.ReducedFunctional(J, [controlY, controlsig, controla])

        control_arr = [p.data() for p in Jhat_np.controls]
        control_arr = Jhat_np.obj_to_array(control_arr) # Array of all controls, flattened
        
        return Jhat_np, control_arr, forward, Y_FctSpace, Jhat, J # pass objects as well as forward method

    def __getitem__(self, idx):
        if torch.is_tensor(idx):   idx = idx.tolist()
        
        x = self.red_dataset[idx]
        
        #x = {k: for k in x if isinstance(x[k], torch.Tensor)}
        return x
    
    
    def __len__(self):
        return len(self.red_dataset)
    


class PB_model(nn.Module):
    def __init__(self, 
                 kernel_size,
                 N_layers,
                 N_latent_layers,
                 hidden_channels=64,
                 batch_size=8,
                 LR=1e-3,
                 gamma=0.95,
                 regularization=None,
                 logdir=None,
                 exp=False):
        
        super(PB_model, self).__init__()
        
        if logdir is not None:
            self.logger = SummaryWriter( logdir ) 
            self.logger.add_text('Name', f'K-{kernel_size}_Nlayer-{N_layers}_hidden_channels-{hidden_channels}_batch_size-{batch_size}_gamma-{gamma}', global_step=0) 
        
        self._device = torch.device('cpu')#$torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        
        self.kernel_size = kernel_size
        
        self.convNextBlock = nn.ModuleList()
        self.convNextBlock.append(nn.Conv2d(1, hidden_channels, 5, padding=2))
        self.convNextBlock.append(nn.BatchNorm2d(hidden_channels))
        self.convNextBlock.append(nn.ReLU())
        
        for _ in range(N_layers):
            self.convNextBlock.append(ConvNextCell(in_out_channel=hidden_channels, kernel=kernel_size, activation='gelu', batchnorm=False, dropout_rate=0.0, 
                                          inv_bottleneck_factor=4, verbose=False, bias=True))

        # Latent blocks takes input and gives output of same shape
        self.latentBlocks = nn.ModuleList()
        self.latentBlocks.append(DownsampleLayer(hidden_channels, 2*hidden_channels, kernel=4, activation='gelu'))
        for _ in range(N_latent_layers):
            self.latentBlocks.append(ConvNextCell(in_out_channel=2*hidden_channels, kernel=kernel_size, activation='gelu', batchnorm=False, dropout_rate=0.0, 
                                          inv_bottleneck_factor=4, verbose=False, bias=True))
            
        self.latentBlocks.append(nn.Upsample(scale_factor=4))
        
        self.convNextBlock.append(nn.Conv2d(3*hidden_channels, 1, 5, padding=2)) # 3*hidden because it takes 2*hidden from the latent space

        self.make_scalar_net() # Predicts sigma, alpha
        
        self.first_epoch = True
        
        self.blur = transforms.GaussianBlur(9, sigma=(2., 4.))
        self.opt = torch.optim.Adam( [{"params": self.sig_alpha_net.parameters(), "lr": LR, "name": 'const_net'},
                                      {"params": self.convNextBlock.parameters(), "lr": LR, "name": 'convNext'},
                                      {"params": self.latentBlocks.parameters(), "lr": LR, "name": 'latent'}
                                     ], 
                                    lr=LR)
        self.sch = torch.optim.lr_scheduler.ExponentialLR(self.opt, gamma=gamma)
        self.batch_counter = 0
        self.batch_size = batch_size
        
    def make_scalar_net(self):
        self.sig_alpha_net = nn.ModuleList()
        self.sig_alpha_net.append(nn.Conv2d(1, 16, 16, stride=16))
        self.sig_alpha_net.append(nn.BatchNorm2d(16))
        self.sig_alpha_net.append(nn.ReLU())
        self.sig_alpha_net.append(nn.Flatten())
        self.sig_alpha_net.append(nn.LazyLinear(32))
        self.sig_alpha_net.append(nn.ReLU())
        self.sig_alpha_net.append(nn.Linear(32, 2))
        return
        
        
    def pretrain(self, x, Y_init, sigma_a_init, alpha_init, mask=None):
        self.batch_counter += 1
        if self.batch_counter == 1: self.opt.zero_grad()
        
        y_im, _, constants = self.forward(x, None)
        
        if mask is None:
            loss = nn.MSELoss()(y_im, Y_init) + self.imag_regularization*imag_part.abs().sum()
        else:
            loss = nn.MSELoss()(y_im[mask!=0], Y_init[mask!=0]) #+ imag_part.abs().sum()

        loss += (constants[0] - sigma_a_init)**2 + (constants[1] - alpha_init)**2
        loss.backward()
        
        if self.batch_counter == self.batch_size:
            #print(f"Batch counter {self.batch_counter}. Stepping grad")
            self.batch_counter = 0
            self.opt.step()
            self.sch.step()
        
        return loss, y_im.detach().cpu().numpy(), constants[0].detach().cpu().numpy(), constants[1].detach().cpu().numpy()
    
    def step(self, x, Jhat, len_control_arr, Y_FctSpace):
        self.batch_counter += 1
        if self.batch_counter == 1: self.opt.zero_grad()
        
        y_im, y_mesh_vals, constants = self.forward(x, Y_FctSpace)
        
        assert len(y_mesh_vals) + len(constants.squeeze()) == len_control_arr
        
        control_arr = np.zeros(len_control_arr)
        
        control_arr[:len(y_mesh_vals)] = y_mesh_vals.detach().cpu().numpy()
        control_arr[-2] = constants[0] # sig
        control_arr[-1] = constants[1] # alpha

        J = Jhat(control_arr) # Tape gets two z
        dJdy = Jhat.derivative(control_arr, forget=True, project=False)

        y_mesh_vals.backward(gradient=torch.tensor(dJdy[:-2], device=torch.device('cuda:0'))/self.batch_size) #+ self.y_regularization*y_mesh_vals.abs())
        constants[0].backward(gradient=torch.tensor(dJdy[-2], device=torch.device('cuda:0'))/self.batch_size, retain_graph=True)
        constants[1].backward(gradient=torch.tensor(dJdy[-1], device=torch.device('cuda:0'))/self.batch_size)        
        
        if self.batch_counter == self.batch_size:
            self.batch_counter = 0
            self.opt.step()
            self.sch.step()
        return constants, y_im, y_mesh_vals, J, dJdy #, pvals.detach().abs().mean().numpy(), pgrad.detach().abs().mean().numpy()#, xgrad, yimgrad
    
    
    def forward(self, x, Y_FctSpace, const_grad=False):
        if const_grad: constants = 1.*x.clone()
        else: constants = 1.*x.detach().clone()
        
        for l, layer in enumerate(self.convNextBlock):
            #print(l, len(self.convNextBlock))
            #print(layer)
            #print(x.shape)
            x = layer(x)
            if l==0: # after passing through first layer of net, pass to latent space
                latent = 1.*x
                for latent_layer in self.latentBlocks: latent = latent_layer(latent)
            if l==len(self.convNextBlock)-2: # prior to last layer, concatenate them
                x = torch.cat([x, latent], axis=1) 
                #print("Concatenating")
            #print(x.shape)
            #print(latent.shape)
        
        for layer in self.sig_alpha_net:
            constants = layer(constants) # shape [B, 2]
            
        constants=constants[0].exp()
        y_im = self.blur(x[0].exp()) # should have shape [B, H, W]
        
        if Y_FctSpace is not None: 
            y_mesh_vals = proc.scalar_img_to_mesh(y_im.squeeze(), Y_FctSpace, vals_only=True)

            return y_im, y_mesh_vals, constants
        else: 
            return y_im, None, constants
    

    def plot_Y(self, epoch, zyx, Y):
        fig, ax = plt.subplots(1, 4, figsize=(2*4, 2*1), dpi=144)

        ax[0].imshow(zyx/zyx.max(), origin='lower', vmax=0.3, cmap='gray')
        ax[1].imshow(Y, origin='lower', vmax=None, vmin=0, cmap='Greens')
        ax[2].imshow(Y, origin='lower', vmax=1e-3, vmin=0, cmap='Greens')
        ax[3].imshow(Y, origin='lower', vmax=1e-5, vmin=0, cmap='Greens')
        
    
        for a in ax.flat: a.axis('off')

        ax[0].set_title('Zyx')
        ax[1].set_title('Y')
        ax[2].set_title('Y vmax 1e-3')
        ax[3].set_title('Y vmax 1e-5')

        fig.subplots_adjust(wspace=0.01, hspace=0.01)
        self.logger.add_figure("Y/model", fig, close=True, global_step=epoch)
        return
