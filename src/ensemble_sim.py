import jax.numpy as jnp
import jax
import equinox as eqx
from jaxtyping import PRNGKeyArray
from typing import Any

from functools import partial

from cryojax.data import RelionParticleParameters
from cryojax.rotations import SO3
from cryojax.inference import distributions as dist
from cryojax.image.operators import CircularCosineMask
from cryojax.image import operators as op
import cryojax.simulator as cxs


@partial(eqx.filter_vmap, in_axes=(0, None), out_axes=eqx.if_array(0))
def make_particle_parameters(
    key: PRNGKeyArray, instrument_config: cxs.InstrumentConfig
) -> RelionParticleParameters:
    """
    Takes an instrument_config and generates transfer theory and pose
    i.e, all things needed to simulate images and store info on them 

    From the cryojax tutorial for generating ensembles of images.

    Below, many customizable parameters are fixed for simplicity, can be adjusted 
    """
    ## Generate random parameters
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


def build_distribution_from_particle_parameters(
    key: PRNGKeyArray,
    particle_parameters: RelionParticleParameters,
    args: Any,
) -> cxs.ContrastImageModel:
    """
    Takes generated particle parameters and generates a white noise Distribution,
    from which one can generate noisy or clean images, or compute likelihoods of images 

    From the cryojax tutorial for generating ensembles of images.
    """
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
    """
    Takes generated particle parameters,generates a white noise Distribution,
    and then simulates noiseless images from them.

    From the cryojax tutorial for generating ensembles of images.
    """
    distribution = build_distribution_from_particle_parameters(
        key, particle_parameters, args
    )
    return distribution.compute_signal()


@eqx.filter_jit
def estimate_signal_variance(
    key, n_images_for_estimation, mask_radius, instrument_config, args, *, batch_size=None
):
    """
    Estimates the "signal" of signal-to-noise for images generated according input param specification
    
    Roughly, computes some sample images, takes the average per-image variance of masked images

    From the cryojax tutorial for generating ensembles of images.
    """
    
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
    """
    This is the per-image function for generating clean images, which is then vectorized via another function.
    """
    _, key_structure = jax.random.split(key)
    distribution = build_distribution_from_particle_parameters(
        key_structure, particle_parameters, args
    )
    return distribution.compute_signal()


def compute_image_with_noise(
    key: PRNGKeyArray,
    particle_parameters: RelionParticleParameters,
    args: Any,
):
    """
    This is the per-image function for generating clean images, which is then vectorized via another function.
    
    From the cryojax tutorial for generating ensembles of images.
    """
    key_noise, key_structure = jax.random.split(key)
    distribution = build_distribution_from_particle_parameters(
        key_structure, particle_parameters, args
    )
    return distribution.sample(key_noise)
