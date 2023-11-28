# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy

#----------------------------------------------------------------------------

def getRandomSamplesInNSphere(N , R , numberOfSamples):
    # Return 'numberOfSamples' samples of vectors of dimension N 
    # with an uniform distribution inside the N-Sphere of radius R.
    # RATIONALE: https://math.stackexchange.com/q/87238
    
    randomnessGenerator = np.random.default_rng()
    
    X = randomnessGenerator.normal(size=(numberOfSamples , N))
    U = randomnessGenerator.random((numberOfSamples , 1)) 
    
    return R * U**(1/N) / np.sqrt(np.sum(X**2, 1, keepdims=True)) * X

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=int, help='Number of random seeds')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def generate_images(
    ctx: click.Context,
    network_pkl: str,
    seeds: int,
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    class_idx: Optional[int]
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate class conditional images
    python generate_class_ind.py --outdir=../storage/out_ind --seeds=10000 --class=4 \
        --network=training-runs/00006-faces80k256-cond-paper256-batch256-ada-target0.5-resumecustom/network-snapshot-001433.pkl

    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)
    
    os.makedirs(os.path.join(outdir, 'class'+str(class_idx)), exist_ok=True)

    if seeds is None:
        ctx.fail('--seeds option is required')

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            ctx.fail('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')

    # Generate images.
    z_array = getRandomSamplesInNSphere(G.z_dim, 10, seeds)
    z_array = np.add(z_array, np.random.RandomState(999).randn(G.z_dim))
    
    for seed in range(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed+1, seed+1, seeds))
        z = z_array[seed].reshape(1, G.z_dim)  
        z = torch.from_numpy(z).to(device)
        
        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/class{class_idx}/seed{seed:04d}.png')
    
    # Save the seed z's
    np.savetxt(f'{outdir}/z_array.csv', z_array, delimiter = ",", header = ','.join(['X' + str(i) for i in range(seeds)]))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
