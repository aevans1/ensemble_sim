import jax.numpy as jnp
import jax
import equinox as eqx
from typing import Any

from cryojax.data import ParticleStack
from cryojax.inference import distributions as dist
import cryojax.simulator as cxs

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
