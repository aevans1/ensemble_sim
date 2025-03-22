from typing import Any, Tuple
from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray, PyTree
import equinox as eqx
import equinox.internal as eqxi

import cryojax as cx
import cryojax.simulator as cxs
from cryojax.io import read_atoms_from_pdb
from cryojax.image.operators import FourierGaussian
from cryojax.rotations import SO3
from cryojax import get_filter_spec

import logging

from jax.extend import backend
print("Device : ", backend.get_backend().platform)

# enable 16 bit precision for jax
from jax import config
config.update("jax_enable_x64", True)

def build_image_formation_stuff(config):
    box_size = config["box_size"]
    pixel_size = config["pixel_size"]
    pdb_fnames = config["models_fnames"]
    path_to_models = config["path_to_models"]
    weights = jnp.array(config['weights_models'])
    
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
    args = {}
    args["instrument_config"] = instrument_config
    args["potentials"] = potentials
    args["potential_integrator"] = potential_integrator
    args["weights"] = weights
    return  args


@partial(eqx.filter_vmap, in_axes=(0, None), out_axes=(0, None))
def make_imaging_pipeline(key, args):
    ## extract arguments
    instrument_config = args["instrument_config"]
    potentials = args["potentials"]
    potential_integrator = args["potential_integrator"]
    weights = args["weights"]

    ## Key generation for imaging parameters
    # Rotation
    key, subkey = jax.random.split(key)  
    rotation = SO3.sample_uniform(subkey)

    # Translation
    key, subkey = jax.random.split(key)
    ny, nx = instrument_config.shape
    in_plane_offset_in_angstroms = (
        jax.random.uniform(subkey, (2,), minval=0, maxval=0.1)
        * jnp.asarray((nx, ny))
        * instrument_config.pixel_size
    )
   
    # Convert 2D in-plane translation to 3D, set out-of-plane translation to 0
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
        minval=10000,
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

    filter_spec = _get_imaging_pipeline_filter_spec(imaging_pipeline)
    imaging_pipeline_vmap, imaging_pipeline_novmap = eqx.partition(
        imaging_pipeline, filter_spec)
    
    return imaging_pipeline_vmap, imaging_pipeline_novmap


def compute_image_stack_with_noise(key, config, imaging_pipeline, noise_args):
    # Define what we vmap over
    filter_spec = _get_imaging_pipeline_filter_spec(imaging_pipeline)

    # Compute clean images, with fancy vmapping
    @partial(cx.filter_vmap_with_spec, filter_spec=filter_spec)
    def compute_image_stack(imaging_pipeline): 
        return imaging_pipeline.render()
    images = compute_image_stack(imaging_pipeline) 

    # Add noise to images
    key, *subkeys = jax.random.split(key, config["number_of_images"] + 1)
    subkeys = jnp.array(subkeys)
    noised_images, noise_power_sq = add_noise_(images, subkeys, noise_args)
    
    return noised_images, noise_power_sq


@partial(jax.vmap, in_axes=(0, 0, None), out_axes=0)
def add_noise_(image, key, noise_args):
    noise_grid, noise_radius_mask, noise_snr = noise_args
    key, subkey = jax.random.split(key)
    radii_for_mask = noise_grid[None, :] ** 2 + noise_grid[:, None] ** 2
    mask = radii_for_mask < noise_radius_mask**2

    signal_power = jnp.sqrt(jnp.sum((image * mask) ** 2) / jnp.sum(mask))

    noise_power = signal_power / jnp.sqrt(noise_snr)
    image = image + jax.random.normal(subkey, shape=image.shape) * noise_power
    return image, noise_power**2

def _pointer_to_vmapped_parameters(imaging_pipeline):
    output = (
        imaging_pipeline.scattering_theory.transfer_theory.ctf.defocus_in_angstroms,
        imaging_pipeline.scattering_theory.transfer_theory.ctf.astigmatism_in_angstroms,
        imaging_pipeline.scattering_theory.transfer_theory.ctf.astigmatism_angle,
        imaging_pipeline.scattering_theory.transfer_theory.ctf.phase_shift,
        imaging_pipeline.scattering_theory.transfer_theory.envelope.b_factor,
        imaging_pipeline.scattering_theory.structural_ensemble.pose.offset_x_in_angstroms,
        imaging_pipeline.scattering_theory.structural_ensemble.pose.offset_y_in_angstroms,
        imaging_pipeline.scattering_theory.structural_ensemble.pose.view_phi,
        imaging_pipeline.scattering_theory.structural_ensemble.pose.view_theta,
        imaging_pipeline.scattering_theory.structural_ensemble.pose.view_psi,
        imaging_pipeline.scattering_theory.structural_ensemble.conformation
    )
    return output


def _get_imaging_pipeline_filter_spec(imaging_pipeline):
    return get_filter_spec(imaging_pipeline, _pointer_to_vmapped_parameters)
