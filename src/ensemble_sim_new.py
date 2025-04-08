from functools import partial

from typing import Any

from cryojax.inference import distributions as dist
from cryojax.image.operators import CircularCosineMask


import jax
import jax.numpy as jnp
import equinox as eqx
import equinox.internal as eqxi
from jaxtyping import PRNGKeyArray

import cryojax as cx
import cryojax.simulator as cxs
from cryojax.io import read_atoms_from_pdb
from cryojax.image.operators import FourierGaussian
from cryojax.rotations import SO3
from cryojax.utils import get_filter_spec
from cryojax.utils._filtered_transformations import filter_vmap_with_spec

from cryojax.constants import get_tabulated_scattering_factor_parameters
from cryojax.io import read_atoms_from_pdb
from cryojax.image import operators as op

from cryojax.data import RelionParticleStack, RelionParticleParameters,AbstractParticleStack, RelionParticleDataset, AbstractDataset

import logging

from jax.extend import backend
print("Device : ", backend.get_backend().platform)

# enable 16 bit precision for jax
from jax import config
config.update("jax_enable_x64", True)

def build_image_formation_stuff(config):
    pdb_fnames = config["models_fnames"]
    path_to_models = config["path_to_models"]
    weights = jnp.array(config['weights_models'])
    box_size = config["box_size"]
    voxel_size = config["pixel_size"]

    logging.info("Generating potentials...")
    potentials = []
    for i in range(len(pdb_fnames)):

        # Load atomic structure and transform into a potential
        filename = path_to_models + "/" + pdb_fnames[i]
        atom_positions, atom_identities, b_factors = read_atoms_from_pdb(
            filename, center=True, select="not element H", get_b_factors=True
        )
        atomic_potential = cxs.PengAtomicPotential(atom_positions, atom_identities, b_factors)
        
        #NOTE: for now, just using the atomic potential, voxel potential takes up too much memory 
        potentials.append(atomic_potential)

        # alternatively: Convert to a real voxel grid
        # This step is optional, you could use the atomic potential directly!
        #real_voxel_grid = atomic_potential.as_real_voxel_grid(
        #    shape=(box_size, box_size, box_size), voxel_size=voxel_size
        #)
        #potential = cxs.FourierVoxelGridPotential.from_real_voxel_grid(
        #real_voxel_grid, voxel_size, pad_scale=2
        #)
        #potentials.append(potential)
    # NOTE: here, this is the potential integrator used for the atomic potential.
    # below that is the potential integral if we had fourier voxel grids above 
    potential_integrator = cxs.GaussianMixtureProjection()
    #potential_integrator = cxs.FourierSliceExtraction(interpolation_order=1)

    logging.info("...Potentials generated")
    
    instrument_config = instrument_config_from_params(config)
    
    args = {}
    args["instrument_config"] = instrument_config
    args["potentials"] = potentials
    args["potential_integrator"] = potential_integrator
    args["weights"] = weights
    return  args

def instrument_config_from_params(config):
    box_size = config["box_size"]
    pixel_size = config["pixel_size"]

    instrument_config = cxs.InstrumentConfig(
        shape=(box_size, box_size),
        pixel_size=pixel_size,
        voltage_in_kilovolts=300.0,
        pad_scale=1.0,
    )
    return instrument_config

def build_distribution_from_particle_parameters(
    key: PRNGKeyArray,
    relion_particle_parameters_vmap: RelionParticleParameters,
    relion_particle_parameters_novmap: RelionParticleParameters,
    args: Any,
) -> cxs.ContrastImagingPipeline:

    relion_particle_parameters = eqx.combine(relion_particle_parameters_vmap, relion_particle_parameters_novmap)

    potentials, potential_integrator, weights, noise_variance = args

    key, subkey = jax.random.split(key)
    potential_id = jax.random.choice(
        subkey, weights.shape[0], p=weights
    )

    structural_ensemble = cxs.DiscreteStructuralEnsemble(
        potentials,
        relion_particle_parameters.pose,
        cxs.DiscreteConformationalVariable(potential_id),
    )

    scattering_theory = cxs.WeakPhaseScatteringTheory(
        structural_ensemble,
        potential_integrator,
        relion_particle_parameters.transfer_theory,
    )
    imaging_pipeline = cxs.ContrastImagingPipeline(
        relion_particle_parameters.instrument_config, scattering_theory
    )
    distribution = dist.IndependentGaussianPixels(
        imaging_pipeline,
        variance=noise_variance,
    )
    return distribution


@partial(eqx.filter_vmap, in_axes=(0, None), out_axes=(0, None))
def make_particle_parameters(
    key: PRNGKeyArray, instrument_config: cxs.InstrumentConfig
) -> RelionParticleParameters:
    # Generate random parameters

    # Pose
    # ... instantiate rotations

    key, subkey = jax.random.split(key)  # split the key to use for the next random number

    rotation = SO3.sample_uniform(subkey)
    key, subkey = jax.random.split(key)  # do this everytime you use a key!!

    # ... now in-plane translation
    ny, nx = instrument_config.shape
    offset_in_angstroms = (
        jax.random.uniform(subkey, (2,), minval=-0.2, maxval=0.2)
        * jnp.asarray((nx, ny))
        * instrument_config.pixel_size
    )
    # ... build the pose
    pose = cxs.EulerAnglePose.from_rotation_and_translation(rotation, offset_in_angstroms)

    # CTF Parameters
    # ... defocus
    defocus_in_angstroms = jax.random.uniform(subkey, (), minval=10000, maxval=15000)
    key, subkey = jax.random.split(key)

    astigmatism_in_angstroms = jax.random.uniform(subkey, (), minval=0, maxval=100)
    key, subkey = jax.random.split(key)

    astigmatism_angle = jax.random.uniform(subkey, (), minval=0, maxval=jnp.pi)
    key, subkey = jax.random.split(key)

    phase_shift = jax.random.uniform(subkey, (), minval=0, maxval=0)
    # no more random numbers needed

    # now generate your non-random values
    spherical_aberration_in_mm = 2.7
    amplitude_contrast_ratio = 0.1
    b_factor = 0.0
    ctf_scale_factor = 1.0

    # ... build the CTF
    transfer_theory = cxs.ContrastTransferTheory(
        ctf=cxs.ContrastTransferFunction(
            defocus_in_angstroms=defocus_in_angstroms,
            astigmatism_in_angstroms=astigmatism_in_angstroms,
            astigmatism_angle=astigmatism_angle,
            spherical_aberration_in_mm=spherical_aberration_in_mm,
            amplitude_contrast_ratio=amplitude_contrast_ratio,
            phase_shift=phase_shift,
        ),
        envelope=op.FourierGaussian(b_factor=b_factor, amplitude=ctf_scale_factor),
    )

    relion_particle_parameters = RelionParticleParameters(
        instrument_config=instrument_config,
        pose=pose,
        transfer_theory=transfer_theory,
    )

    filter_spec = _get_parameters_filter_spec(relion_particle_parameters)
    relion_particle_parameters_vmap, relion_particle_parameters_novmap = eqx.partition(relion_particle_parameters, filter_spec)

    return relion_particle_parameters_vmap, relion_particle_parameters_novmap

@eqx.filter_jit
#@partial(eqx.filter_vmap, in_axes=(0, eqx.if_array(0), None))
@partial(eqx.filter_vmap, in_axes=(0, 0, None, None))
def simulate_noiseless_images(key, particle_parameters_vmap, particle_parameters_novmap, args):
    distribution = build_distribution_from_particle_parameters(
        key, particle_parameters_vmap, particle_parameters_novmap, args
    )
    return distribution.compute_signal()


@eqx.filter_jit
def estimate_signal_variance(
    key, n_images_for_estimation, mask_radius, instrument_config, args, *, batch_size=None
):
    key, *subkeys = jax.random.split(key, n_images_for_estimation + 1)
    subkeys = jnp.array(subkeys)


    particle_parameters_vmap, particle_parameters_novmap = make_particle_parameters(subkeys, instrument_config)
    
    # set offset at 0 for simplicity
    particle_parameters_vmap = eqx.tree_at(
        lambda d: (d.pose.offset_x_in_angstroms, d.pose.offset_y_in_angstroms),
        particle_parameters_vmap,
        replace_fn=lambda x: 0.0 * x,
    )

    key, *subkeys = jax.random.split(key, n_images_for_estimation + 1)
    subkeys = jnp.array(subkeys)
    
    noiseless_images = simulate_noiseless_images(subkeys, particle_parameters_vmap, particle_parameters_novmap, args)

    # define noise mask
    mask = CircularCosineMask(
        particle_parameters_novmap.instrument_config.coordinate_grid_in_pixels,
        radius_in_angstroms_or_pixels=mask_radius,
        rolloff_width_in_angstroms_or_pixels=1.0,
    )

    signal_variance = jnp.var(
        noiseless_images, axis=(1, 2), where=jnp.where(mask.array == 1.0, True, False)
    ).mean()

    return signal_variance


def compute_image_with_noise(
    key: PRNGKeyArray,
    relion_particle_parameters_vmap: RelionParticleParameters,
    relion_particle_parameters_novmap: RelionParticleParameters,
    args: Any,
):
    key_noise, key_structure = jax.random.split(key)
    distribution = build_distribution_from_particle_parameters(
        key_structure, relion_particle_parameters_vmap, relion_particle_parameters_novmap, args
    )
    return distribution.sample(key_noise)

def compute_image_stack_with_noise(key, number_of_images, parameters, filter_spec_for_vmap, args):
    
    key, *subkeys = jax.random.split(key, number_of_images + 1)
    subkeys = jnp.array(subkeys)
    parameters_vmap, parameters_novmap = eqx.partition(parameters, filter_spec_for_vmap)
    noised_images = jax.lax.map(
        lambda x: compute_image_with_noise(key=x[0], 
                                           relion_particle_parameters_vmap=x[1],
                                           relion_particle_parameters_novmap=parameters_novmap, 
                                           args=args),
        xs=(subkeys, parameters_vmap),
        batch_size=100,
    )
    imaging_stack = RelionParticleStack(parameters, noised_images)
    return imaging_stack

def _get_parameters_filter_spec(parameters):
    return get_filter_spec(parameters, _pointer_to_vmapped_parameters)

def _pointer_to_vmapped_parameters(parameters):
    output = (
        parameters.transfer_theory.ctf.defocus_in_angstroms,
        parameters.transfer_theory.ctf.astigmatism_in_angstroms,
        parameters.transfer_theory.ctf.astigmatism_angle,
        parameters.transfer_theory.ctf.phase_shift,
        parameters.transfer_theory.envelope.b_factor,
        parameters.transfer_theory.envelope.amplitude,
        parameters.pose
    )
    return output

