import jax.numpy as jnp
import jax
import jax_dataloader as jdl
import equinox as eqx
from typing import Tuple
from jaxtyping import Float, Array

import cryojax.simulator as cxs

@eqx.filter_jit
def compute_single_likelihood(
    potential_id: int,
    particle_stack,
    args: Tuple[
        Tuple[cxs.AbstractPotentialRepresentation], cxs.AbstractPotentialIntegrator
    ],
) -> Float:
    potentials, potential_integrator = args
    structural_ensemble = cxs.DiscreteStructuralEnsemble(
        potentials,
        particle_stack["parameters"]["pose"],
        cxs.DiscreteConformationalVariable(potential_id),
    )

    scattering_theory = cxs.WeakPhaseScatteringTheory(
        structural_ensemble,
        potential_integrator,
        particle_stack["parameters"]["transfer_theory"],
    )
    image_model = cxs.ContrastImageModel(
        particle_stack["parameters"]["instrument_config"], scattering_theory
    )

    simulated_image = image_model.render()
    observed_image = particle_stack["images"]


    # NOTE: this is here because we normalized each observed image to variance 1, 
    #  and we have to incorporate that into the likelihood calc
    cc = jnp.mean(simulated_image**2)
    co = jnp.mean(observed_image * simulated_image)
    c = jnp.mean(simulated_image)
    o = jnp.mean(observed_image)

    scale = (co - c * o) / (cc - c**2)
    bias = o - scale * c

    return -jnp.sum((observed_image - scale * simulated_image - bias) ** 2) / 2.0


@eqx.filter_jit
def compute_likelihood_with_map(
    potential_id: int,
    particle_stack,
    args: Tuple[
        Tuple[cxs.AbstractPotentialRepresentation], cxs.AbstractPotentialIntegrator
    ],
    *,
    batch_size_images: int,
) -> Float[Array, " n_structures"]:
    """
    Computes one row of the likelihood matrix (all structures, one image)
    """

    stack_map, stack_nomap = eqx.partition(particle_stack, eqx.is_array)

    likelihood_batch = jax.lax.map(
        lambda x: compute_single_likelihood(
            potential_id, eqx.combine(x, stack_nomap), args
        ),
        xs=stack_map,
        batch_size=batch_size_images,  # compute for this many images in parallel
    )
    return likelihood_batch


def compute_likelihood_matrix_with_lax_map(
    dataloader: jdl.DataLoader,
    args: Tuple[
        Tuple[cxs.AbstractPotentialRepresentation], cxs.AbstractPotentialIntegrator
    ],
    *,
    batch_size_potentials: int = None,
    batch_size_images: int = None,
) -> Float[Array, " n_images n_structures"]:
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