import jax.numpy as jnp
import jax
import equinox as eqx
from jaxtyping import PRNGKeyArray
from typing import Any

from functools import partial

from cryojax.rotations import SO3
from cryojax.inference.distributions import IndependentGaussianPixels
from cryojax.image.operators import CircularCosineMask
from cryojax.image import operators as op
import cryojax.simulator as cxs


@partial(eqx.filter_vmap, in_axes=(0, None))
def make_particle_parameters(
    key: PRNGKeyArray, instrument_config: cxs.InstrumentConfig
):
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

    # ... now in-plane translation
    ny, nx = instrument_config.shape
    key, subkey = jax.random.split(key)  # do this everytime you use a key!!
    offset_in_angstroms = (
        jax.random.uniform(subkey, (2,), minval=-0.1, maxval=0.1)
        * jnp.asarray((nx, ny))
        * instrument_config.pixel_size
    )
    # ... build the pose
    pose = cxs.EulerAnglePose.from_rotation_and_translation(rotation, offset_in_angstroms)

    # CTF Parameters
    # ... defocus
    key, subkey = jax.random.split(key)
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

    # ... build the CTF
    transfer_theory = cxs.ContrastTransferTheory(
        ctf=cxs.CTF(
            defocus_in_angstroms=defocus_in_angstroms,
            astigmatism_in_angstroms=astigmatism_in_angstroms,
            astigmatism_angle=astigmatism_angle,
            spherical_aberration_in_mm=spherical_aberration_in_mm,
        ),
        amplitude_contrast_ratio=amplitude_contrast_ratio,
        phase_shift=phase_shift,
    )

    particle_parameters = {
        "instrument_config": instrument_config,
        "pose": pose,
        "transfer_theory": transfer_theory,
        "metadata": {},
    } 
    
    return particle_parameters


def compute_image(
    parameters,
    constant_args,
    per_particle_args
):
 
    potentials, potential_integrator, mask, snr = constant_args
    noise_key, potential_id = per_particle_args

    structural_ensemble = cxs.DiscreteStructuralEnsemble(
        potentials,
        parameters["pose"],
        cxs.DiscreteConformationalVariable(potential_id),
    )

    scattering_theory = cxs.WeakPhaseScatteringTheory(
        structural_ensemble,
        potential_integrator,
        parameters["transfer_theory"],
    )

    image_model = cxs.ContrastImageModel(
        parameters["instrument_config"], 
        scattering_theory, 
        mask=mask
    )
    distribution = IndependentGaussianPixels(
        image_model,
        variance=1.0,
        signal_scale_factor=jnp.sqrt(snr),
        normalizes_signal=True
    )
    
    return distribution.sample(noise_key, applies_mask=False)
