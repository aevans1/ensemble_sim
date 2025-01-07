from typing import Any, Tuple
from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
import equinox as eqx
import equinox.internal as eqxi

import cryojax.simulator as cxs
from cryojax.io import read_atoms_from_pdb
from cryojax.image.operators import FourierGaussian
from cryojax.rotations import SO3

import logging
import os

from jax.lib import xla_bridge
from jax.extend import backend
print("Device : ", backend.get_backend().platform)

# enable 16 bit precision for jax
from jax import config
config.update("jax_enable_x64", True)

def simulate_image(key, args):

    ## extract arguments
    instrument_config, potentials, potential_integrator, weights = args

    ## Key generation for imaging parameters
    # Rotation
    key, subkey = jax.random.split(key)  
    rotation = SO3.sample_uniform(subkey)

    # Translation
    key, subkey = jax.random.split(key)
    ny, nx = instrument_config.shape
    in_plane_offset_in_angstroms = (
        jax.random.uniform(subkey, (2,), minval=0, maxval=0) # maximum change is in a radius of 0.2 * box length
        * jnp.asarray((nx, ny))
        * instrument_config.pixel_size
    )
   
    # Convert 2D in-plane translation to 3D, set out-ou-plane translation to 0
    offset_in_angstroms = jnp.pad(in_plane_offset_in_angstroms, ((0, 1),))

    # Build the pose
    pose = cxs.EulerAnglePose.from_rotation_and_translation(
        rotation, offset_in_angstroms
    )

    # defocus
    key, subkey = jax.random.split(key)
    defocus_in_angstroms = jax.random.uniform(
        subkey,
        (),
        minval=10000, # change to prefered values
        maxval=15000,
    )
    
    # astigmatism
    key, subkey = jax.random.split(key)
    astigmatism_in_angstroms = jax.random.uniform(
        subkey,
        (),
        minval=0,
        maxval=100,
    )
    key, subkey = jax.random.split(key)
    astigmatism_angle = jax.random.uniform(
        subkey,
        (),
        minval=0,
        maxval=jnp.pi,
    )
    
    # phase shift
    key, subkey = jax.random.split(key)
    phase_shift = jax.random.uniform(
        subkey, 
        (), 
        minval=0, 
        maxval=jnp.pi
    )
    
    # b_factor
    key, subkey = jax.random.split(key)
    b_factor = jax.random.uniform(
        subkey,
        (),
        minval=0, # For now, no b-factor
        maxval=0
    )

    # Various other ctf-y things
    spherical_aberration_in_mm = 2.7
    amplitude_contrast_ratio = 0.1
    ctf_scale_factor = 1.0

    ## Build the CTF
    transfer_theory = cxs.ContrastTransferTheory(
        ctf=cxs.ContrastTransferFunction(
            defocus_in_angstroms=defocus_in_angstroms,
            astigmatism_in_angstroms=astigmatism_in_angstroms,
            astigmatism_angle=astigmatism_angle,
            spherical_aberration_in_mm=spherical_aberration_in_mm,
            amplitude_contrast_ratio=amplitude_contrast_ratio,
            phase_shift=phase_shift,
        ),
        envelope=FourierGaussian(b_factor=b_factor, amplitude=ctf_scale_factor),
    )

    ## Sample from potentials
    key, subkey = jax.random.split(key)
    structure_id = jax.random.choice(subkey, weights.shape[0], p=weights)
    
    structural_ensemble = cxs.DiscreteStructuralEnsemble(
        potentials,
        pose,
        cxs.DiscreteConformationalVariable(structure_id),
    )

    scattering_theory = cxs.WeakPhaseScatteringTheory(
        structural_ensemble,
        potential_integrator,
        transfer_theory,
    )

    imaging_pipeline = cxs.ContrastImagingPipeline(
        instrument_config, scattering_theory
    )

    return imaging_pipeline.render(), structure_id 

# NOTE: From https://github.com/DSilva27/cryo_MD/blob/before_cryojax/src/cryo_md/_simulator/noise_utils.py 
def add_noise_(image, random_key, noise_grid, noise_radius_mask, noise_snr):
    radii_for_mask = noise_grid[None, :] ** 2 + noise_grid[:, None] ** 2
    mask = radii_for_mask < noise_radius_mask**2

    signal_power = jnp.sqrt(jnp.sum((image * mask) ** 2) / jnp.sum(mask))

    noise_power = signal_power / jnp.sqrt(noise_snr)
    image = image + jax.random.normal(random_key, shape=image.shape) * noise_power
    return image, noise_power**2

# NOTE: for now, unsure of exactly how to extract structure_id or conformation from the simulation objects, so just returning it alongside images for now
def simulate_dataset(config: dict, weights=None):

    box_size = config["box_size"]
    pixel_size = config["pixel_size"]
    pdb_fnames = config["models_fnames"]
    path_to_models = config["path_to_models"]

    if weights is None:
        weights = jnp.array(config['weights_models'])
    rng_seed = config["rng_seed"] 
    number_of_images = config["number_of_images"]

    logging.info("Generating potentials...")
    potential_integrator = cxs.GaussianMixtureProjection()
    potentials = []
    for i in range(len(pdb_fnames)):
        filename = path_to_models + "/" + pdb_fnames[i]
        atom_positions, atom_identities, b_factors = read_atoms_from_pdb(
            filename, assemble=False, get_b_factors=True
        )
        atomic_potential = cxs.PengAtomicPotential(atom_positions, atom_identities, b_factors)
        potentials.append(atomic_potential)
    potentials = tuple(potentials)
    logging.info("...Potentials generated")

    instrument_config = cxs.InstrumentConfig(
        shape=(box_size, box_size),
        pixel_size=pixel_size,
        voltage_in_kilovolts=300.0,
        pad_scale=1.0,
    )
    noise_grid = jnp.linspace(
          -0.5 * (box_size - 1),
          0.5 * (box_size - 1),
          box_size,
      )
    args = (instrument_config, potentials, potential_integrator, weights)
    
    key = jax.random.PRNGKey(rng_seed)
    key, *subkeys = jax.random.split(key, number_of_images + 1)
    subkeys = jnp.array(subkeys)
    
    logging.info("Generating clean images...")
    images, structure_id  = jax.lax.map(lambda x: simulate_image(x, args), xs=subkeys)
    logging.info("...clean images generated")

    noise_radius = 0.5*box_size - 1 # For now, using a disc that is radius of the image for SNR calculations
    noise_args = noise_grid, noise_radius, config["noise_snr"] 
    key, *subkeys = jax.random.split(jax.random.key(rng_seed), number_of_images+1)
    subkeys = jnp.array(subkeys)
    
    logging.info("Adding noise to images...")
    noised_images, noise_power_sq = jax.lax.map(lambda x: add_noise_(*x, *noise_args), xs=(images, subkeys))
    logging.info("...images are noised")

    logging.info("Returning images and structure labels all at once, until I figure out how to do it smarter!")
    return noised_images, structure_id

