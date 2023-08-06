from dolfin import *
from dolfin_adjoint import *

import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

import subprocess
import time
from argparse import ArgumentParser
from skimage.morphology import disk
import scipy.ndimage
import skimage.measure as measure

from utils.UNeXt import UNet
from utils.utils_data_processing import CellDataset



def downsample_mask(mask, im_rescale=1):    
    mask_max = mask.max()
    mask = mask.astype(float)
    
    reduce_factor = len(mask.shape)*[1,]
    reduce_factor[-2:] = [im_rescale, im_rescale]
    mask = measure.block_reduce(mask, tuple(reduce_factor), np.mean)
    
    mask = mask>mask.max()/2
    mask = mask.astype(int)

    mask = scipy.ndimage.binary_dilation(mask, structure=disk(21//im_rescale), iterations=1)
    mask = scipy.ndimage.binary_fill_holes(mask)

    g = scipy.ndimage.morphology.distance_transform_edt((mask!=0)*1.)


    return mask, g



def mask_to_mesh(mask, matlab_file_loc='/home/schmittms/cell_stress/fenics_testing/matlab_utils/ContourToXMLMesh', exists=False):
    """
    Takes mask.
    First gets contour points, saves to file contour.txt
    Then gets mesh via matlab, and saves to xml file
    Then reads xml to dolfin
    """
    matlabdir=os.path.dirname(matlab_file_loc)
    matlabfile=os.path.basename(matlab_file_loc)

    n = np.random.randint(1e9,1e10)

    contourfile = os.path.join(matlabdir, 'temp', f'contour_{n}.txt')
    outfile = os.path.join(matlabdir, f'temp/contour2xml_{n}.xml')

    print("MESH: ", contourfile, '\t', outfile)

    if not exists:
        mask = mask.astype(np.uint8).squeeze()

        contours, _ = cv.findContours(mask, cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
        contours = np.asarray(contours)

        assert len(contours)==1
        contour = np.squeeze(contours)

        contour = contour[:, [1,0]] # swap x,y, because matlab expects first dim to be x, but it's really y
        
        np.savetxt(contourfile, contour.astype(int), fmt='%u', delimiter=',')

        cmd = f'module load matlab/2022a; cd {matlabdir}; matlab -noopengl -r \" {matlabfile}(\'{contourfile}\',\'{outfile}\'); quit;\"'

        #!{cmd}
        ret = subprocess.run(cmd, capture_output=False, shell=True)
        #shellout = ret.stdout.decode()
    
    
        print("outfile", outfile)
        
        print('file exists?\t', os.path.exists(outfile))

    try:
        mesh = Mesh(outfile)
    except:
        print(shellout)
        mesh = Mesh(outfile)

    os.remove(outfile) 
    
    return mesh


def refine_mesh(mesh, max_volume):
    it = 0
    all_small = False
    while all_small == False:
        cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
        cell_markers.set_all(False)

        all_small = True
        for cell in cells(mesh):
            vol = cell.volume()
            if vol>max_volume: 
                all_small = False
                cell_markers[cell] = True
        it += 1

        if not all_small: mesh = refine(mesh, cell_markers)
        
    print(f"Mesh refined {it} times")
    return mesh

def scalar_img_to_mesh(img, FctSpace, mesh_inv_scale=None, vals_only=False):
    dof_coords = FctSpace.tabulate_dof_coordinates().copy() # Shape (Nnodes, 2)
    
    fct_vals = img[dof_coords[:,1].astype(int), dof_coords[:,0].astype(int)]

    if mesh_inv_scale is not None: # mesh inv scale is factor to go back to go from [m] to [px]
        dof_coords *= mesh_inv_scale

    if vals_only:
        return fct_vals #, dof_coords
    else:
        meshfct = Function(FctSpace, name='Y')                                                                  
        meshfct.vector()[:] = img[dof_coords[:,1].astype(int), dof_coords[:,0].astype(int)]
        return meshfct

def vector_img_to_mesh(img, VecFctSpace, mesh_inv_scale=None):
    meshfct = Function(VecFctSpace)        
    xc, yc = meshfct.split()              

    dof_coords = VecFctSpace.tabulate_dof_coordinates().copy() # Shape (2*Nnodes, 2). Doubled b/c vector space
    if mesh_inv_scale is not None: # mesh inv scale is factor to go back to go from [m] to [px]
        dof_coords *= mesh_inv_scale
    
    xcoords = img[1, dof_coords[:,1].astype(int), dof_coords[:,0].astype(int)] # Make full vector, including the doubling
    dof_coords = dof_coords[::2,:] # Get reduced coordinates (equal to dof_coords[1::2, :]
    ycoords_red = img[0, dof_coords[:,1].astype(int), dof_coords[:,0].astype(int)] # Get y
    xcoords[::2] = ycoords_red # Replace the duplicated coordinates with y coords
    
    xc.vector().set_local( xcoords )
    
    return meshfct


def downsample(im, downsample_factor):
    reduce_factor = len(im.shape)*[1,]
    reduce_factor[-2:] = [downsample_factor, downsample_factor]
    return measure.block_reduce(im, tuple(reduce_factor), np.mean)


def process_mask_generate_mesh(mask, im, im_rescale=1, boundary_factor=10., maxcellvol=100, make_mesh=True):
    assert mask.shape[-2:]==im.shape[-2:], 'Shapes not equal: Mask: \t%s\tF\t%s'%(str(mask.shape), str(F.shape))
    
    mask_max = mask.max()
    mask = downsample(mask.astype(float), im_rescale)
    im_dwnsample = downsample(im.astype(float), im_rescale)

    mask = mask>mask.max()/2
    mask = mask.astype(int)

    mask = scipy.ndimage.binary_dilation(mask, structure=disk(21//im_rescale), iterations=1)
    mask = scipy.ndimage.binary_fill_holes(mask)

    g = scipy.ndimage.morphology.distance_transform_edt((mask!=0)*1.)
    boundary = (g!=0)*(g<boundary_factor*np.median(g[g!=0])) 

    im_dwnsample[:,~boundary] = 0

    if make_mesh:
        mesh = mask_to_mesh(mask, exists=False)
        mesh = refine_mesh(mesh, max_volume=maxcellvol)
    else:
        mesh=None

    return mask, im_dwnsample, mesh, g, boundary
    
def meshfctvalues_to_im(meshfct, mask, mesh, smooth=1.):
    V = FunctionSpace(mesh, "CG", 1)                                                
    Y = Function(V)
    Y.vector()[:] = meshfct
    
    coords = Y.function_space().tabulate_dof_coordinates()

    frbf = scipy.interpolate.Rbf(coords[:,0], coords[:,1], Y.vector().get_local(), smooth=smooth)#, fill_value=np.nan)

    imY = np.zeros(mask.shape)
    nzar =  np.asarray(np.nonzero(mask)).T

    for pt in nzar:
        imY[pt[0],pt[1]] = frbf(pt[1],pt[0])   # interpolated values

    return imY

def get_nn_forces(modelpath, cell, frame, return_zyxin=False):
    #
    modelinfo = torch.load(modelpath,  map_location=torch.device('cpu'));

    dataset_kwargs = modelinfo['dataset_kwargs']
    test_cells = dataset_kwargs['test_cells']

    modelinfo['dataset_kwargs']['root'] = '/project/vitelli/cell_stress/TractionData_All_16kpa_new/'
    # Cells:  'myo_cell_7', 'pax_cell_5', 'pax_cell_2', 'myo_cell_6', 'myo_cell_1', '17_cell_3', '08_cell_1', 'myo_cell_0', '08_cell_3', '17_cell_2', '11_cell_1', 'myo_cell_4', 'myo_cell_9', '11_cell_0', '17_cell_0', '17_cell_1', '10_cell_2', 'myo_cell_5', '08_cell_4', '10_cell_3', 'pax_cell_4', 'myo_cell_8', '11_cell_4', '10_cell_4', '11_cell_2', 'myo_cell_2', '10_cell_1', 'pax_cell_1', 'myo_cell_3', '17_cell_4', '08_cell_2',
    
    modelinfo['dataset_kwargs']['transform_kwargs']['crop_size'] = 960
    modelinfo['dataset_kwargs']['transform_kwargs']['rotate'] = False

    model = UNet(**modelinfo['model_kwargs'], model_idx = 0)

    model.load_state_dict(modelinfo['model'])

    dataset = CellDataset(**modelinfo['dataset_kwargs'])

    filenames = dataset.info.loc[dataset.info['folder']==cell]['filename']
    idx = filenames.index.values[frame]

    sample = dataset[idx]
    for k in sample:
        sample[k] = sample[k].unsqueeze(0)
        
    prediction = model(model.select_inputs(model.input_type, sample)).detach().cpu().numpy().squeeze()
    
    fx,fy = prediction[0]*np.cos(prediction[1]), prediction[0]*np.sin(prediction[1])
    
    if return_zyxin:
        return sample['mask'].detach().cpu().numpy().squeeze(), np.stack([fx, fy]), sample['zyxin'].detach().cpu().numpy().squeeze(), sample['output'].detach().cpu().numpy().squeeze()
    else:
        return sample['mask'].detach().cpu().numpy().squeeze(), np.stack([fx, fy])
    

def get_weight_fct(weighting, F, g): 
    g = g/g.max()
    
    weight_dict = {s.split(',')[0]: float(s.split(',')[1]) if s.split(',')[1][-1].isdigit() else s.split(',')[1] for s in weighting.split('/')}
    #print("WEIGHT DICT: ", weight_dict)

    #weight = [np.ones_like(np.linalg.norm(F, axis=0)), np.ones_like(np.linalg.norm(F, axis=0))]
    if weight_dict['YN']:
        if weight_dict['WeightBy']=='boundary':
            weightBy = g 
        elif weight_dict['WeightBy']=='force':
            weightBy = np.linalg.norm(F, axis=0)
            weightBy = np.minimum(weightBy, weight_dict['WeightParam3'])
        elif weight_dict['WeightBy']=='boundaryMinusForce':
            weightBy = g/g.max() - np.linalg.norm(F, axis=0)/np.max(np.linalg.norm(F, axis=0))
            #weightBy = np.minimum(weightBy, weight_dict['WeightParam3'])

        if weight_dict['WeightFct']=='exp':
            weight_fct = np.exp( weight_dict['WeightParam2']*weightBy)
        elif weight_dict['WeightFct']=='lin':
            weight_fct = weight_dict['WeightParam1'] +  weight_dict['WeightParam2']*weightBy
        

        weight = weight_dict['WeightParam1']*weight_fct/weight_fct.max()

    else:
        weight = np.ones_like(np.linalg.norm(F, axis=0))
    
    return weight, weight_dict
