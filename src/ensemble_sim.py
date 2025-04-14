import jax.numpy as jnp
import jax
import equinox as eqx
from jaxtyping import PRNGKeyArray
import jax_dataloader as jdl
from typing import Any

import logging
from functools import partial

from cryojax.io import read_atoms_from_pdb
from cryojax.data import RelionParticleStackDataset, ParticleStack, RelionParticleParameters
from cryojax.rotations import SO3
from cryojax.inference import distributions as dist
from cryojax.image.operators import CircularCosineMask
from cryojax.image import operators as op
import cryojax.simulator as cxs


#### Imaging Functions
@partial(eqx.filter_vmap, in_axes=(0, None), out_axes=eqx.if_array(0))
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
        jax.random.uniform(subkey, (2,), minval=0, maxval=0.1)
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
                    ),
        envelope=op.FourierGaussian(b_factor=b_factor, amplitude=ctf_scale_factor),
        amplitude_contrast_ratio=amplitude_contrast_ratio,
        phase_shift=phase_shift,
    )

    particle_parameters = RelionParticleParameters(
        instrument_config=instrument_config,
        pose=pose,
        transfer_theory=transfer_theory,
    )

    return particle_parameters

def build_image_formation_stuff(config):
    pdb_fnames = config["models_fnames"]
    path_to_models = config["path_to_models"]
    weights = jnp.array(config['weights_models'])
    box_size = config["box_size"]
    voxel_size = config["pixel_size"]

    logging.info("Generating potentials...")
    potential_integrator = cxs.GaussianMixtureProjection()
    potentials = []
    for i in range(len(pdb_fnames)):

        # Load atomic structure and transform into a potential
        filename = path_to_models + "/" + pdb_fnames[i]
        atom_positions, atom_identities, b_factors = read_atoms_from_pdb(
            filename, loads_b_factors=True
        )
        atomic_potential = cxs.PengAtomicPotential(atom_positions, atom_identities, b_factors)
        potentials.append(atomic_potential)
    potentials = tuple(potentials)
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
    particle_parameters: RelionParticleParameters,
    args: Any,
) -> cxs.ContrastImageModel:
    potentials, potential_integrator, structural_weights, variance = args

    key, subkey = jax.random.split(key)
    potential_id = jax.random.choice(
        subkey, structural_weights.shape[0], p=structural_weights
    )

    structural_ensemble = cxs.DiscreteStructuralEnsemble(
        potentials,
        particle_parameters.pose,
        cxs.DiscreteConformationalVariable(potential_id),
    )

    scattering_theory = cxs.WeakPhaseScatteringTheory(
        structural_ensemble,
        potential_integrator,
        particle_parameters.transfer_theory,
    )
    imaging_pipeline = cxs.ContrastImageModel(
        particle_parameters.instrument_config, scattering_theory
    )
    distribution = dist.IndependentGaussianPixels(
        imaging_pipeline,
        variance=variance,
    )
    return distribution

@eqx.filter_jit
@partial(eqx.filter_vmap, in_axes=(0, eqx.if_array(0), None))
def simulate_noiseless_images(key, particle_parameters, args):
    distribution = build_distribution_from_particle_parameters(
        key, particle_parameters, args
    )

    distribution
    return distribution.compute_signal()


@eqx.filter_jit
def estimate_signal_variance(
    key, n_images_for_estimation, mask_radius, instrument_config, args, *, batch_size=None
):
    key, *subkeys = jax.random.split(key, n_images_for_estimation + 1)
    subkeys = jnp.array(subkeys)

    particle_parameters = make_particle_parameters(subkeys, instrument_config)

    # set offset at 0 for simplicity
    particle_parameters = eqx.tree_at(
        lambda d: (d.pose.offset_x_in_angstroms, d.pose.offset_y_in_angstroms),
        particle_parameters,
        replace_fn=lambda x: 0.0 * x,
    )

    key, *subkeys = jax.random.split(key, n_images_for_estimation + 1)
    subkeys = jnp.array(subkeys)
    noiseless_images = simulate_noiseless_images(subkeys, particle_parameters, args)

    # define noise mask
    mask = CircularCosineMask(
        particle_parameters.instrument_config.coordinate_grid_in_pixels,
        radius_in_angstroms_or_pixels=mask_radius,
        rolloff_width_in_angstroms_or_pixels=1.0,
    )

    signal_variance = jnp.var(
        noiseless_images, axis=(1, 2), where=jnp.where(mask.array == 1.0, True, False)
    ).mean()

    return signal_variance

def compute_image_clean(
    key: PRNGKeyArray,
    particle_parameters: RelionParticleParameters,
    args: Any,
):
    key_noise, key_structure = jax.random.split(key)
    distribution = build_distribution_from_particle_parameters(
        key_structure, particle_parameters, args
    )
    return distribution.compute_signal()

def compute_image_with_noise(
    key: PRNGKeyArray,
    particle_parameters: RelionParticleParameters,
    args: Any,
):
    key_noise, key_structure = jax.random.split(key)
    distribution = build_distribution_from_particle_parameters(
        key_structure, particle_parameters, args
    )
    return distribution.sample(key_noise)


#### Likelihood Functions
class CustomJaxDataset(jdl.Dataset):
    def __init__(self, cryojax_dataset: RelionParticleStackDataset):
        self.cryojax_dataset = cryojax_dataset

    def __getitem__(self, index) -> ParticleStack:
        return self.cryojax_dataset[index]

    def __len__(self) -> int:
        return len(self.cryojax_dataset)
    
@eqx.filter_jit
def compute_single_likelihood(
    potential_id,
    relion_particle_images_map: ParticleStack,
    relion_particle_images_nomap: ParticleStack,
    args: Any,
) -> cxs.ContrastImageModel:
    relion_particle_images = eqx.combine(
        relion_particle_images_map, relion_particle_images_nomap
    )
    potentials, potential_integrator, variance = args
    structural_ensemble = cxs.DiscreteStructuralEnsemble(
        potentials,
        relion_particle_images.parameters.pose,
        cxs.DiscreteConformationalVariable(potential_id),
    )
    scattering_theory = cxs.WeakPhaseScatteringTheory(
        structural_ensemble,
        potential_integrator,
        relion_particle_images.parameters.transfer_theory,
    )
    imaging_pipeline = cxs.ContrastImageModel(
        relion_particle_images.parameters.instrument_config, scattering_theory
    )
    distribution = dist.IndependentGaussianPixels(
        imaging_pipeline,
        variance=variance,
    )
    return distribution.log_likelihood(relion_particle_images.images)


@eqx.filter_jit
def compute_likelihood_with_map(
    potential_id, relion_particle_images, args, *, batch_size_images
):
    """
    Computes one row of the likelihood matrix (all structures, one image)
    """

    stack_map, stack_nomap = eqx.partition(relion_particle_images, eqx.is_array)

    likelihood_batch = jax.lax.map(
        lambda x: compute_single_likelihood(potential_id, x, stack_nomap, args),
        xs=stack_map,
        batch_size=batch_size_images,  # compute for this many images in parallel
    )
    return likelihood_batch


def compute_likelihood_matrix_with_lax_map(
    dataloader, args, *, batch_size_potentials=None, batch_size_images=None
):
    n_potentials = len(args[0])
    likelihood_matrix = []
    for batch in dataloader:
        batch_likelihood = jax.lax.map(
            lambda x: compute_likelihood_with_map(
                x, batch, args, batch_size_images=batch_size_images
            ),
            xs=jnp.arange(n_potentials),
            batch_size=batch_size_potentials,  # potentials to compute in parallel
        ).T
        likelihood_matrix.append(batch_likelihood)
    likelihood_matrix = jnp.concatenate(likelihood_matrix, axis=0)
    return likelihood_matrix
