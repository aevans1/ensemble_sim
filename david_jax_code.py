import json
import logging
from functools import partial

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import jax_dataloader as jdl
import MDAnalysis as mda
import mdtraj
import numpy as np
from tqdm import tqdm

import cryojax.simulator as cxs
from cryojax.constants import get_tabulated_scattering_factor_parameters
from cryojax.data import (
    RelionParticleDataset,
    RelionParticleMetadata,
    RelionParticleStack,
)
from cryojax.image.operators import FourierGaussian
from cryojax.io import read_atoms_from_pdb_or_cif

# from cryojax.image import rfftn, irfftn
from cryojax.utils import get_filter_spec

# from cryo_md.optimization.loss_and_gradients import compute_lklhood_matrix
from cryo_md.data import OptimizationConfig
from cryo_md.simulator._distributions import VarianceMarginalizedWhiteGaussianNoise


class CustomJaxDataset(jdl.Dataset):
    def __init__(self, cryojax_dataset: RelionParticleDataset):
        self.cryojax_dataset = cryojax_dataset

    def __getitem__(self, index) -> RelionParticleStack:
        return self.cryojax_dataset[index]

    def __len__(self) -> int:
        return len(self.cryojax_dataset)


def _get_particle_stack_filter_spec(particle_stack):
    return get_filter_spec(particle_stack, _pointer_to_vmapped_parameters)


def _pointer_to_vmapped_parameters(particle_stack):
    if isinstance(particle_stack.parameters.transfer_theory.envelope, FourierGaussian):
        output = (
            particle_stack.parameters.transfer_theory.ctf.defocus_in_angstroms,
            particle_stack.parameters.transfer_theory.ctf.astigmatism_in_angstroms,
            particle_stack.parameters.transfer_theory.ctf.astigmatism_angle,
            particle_stack.parameters.transfer_theory.ctf.phase_shift,
            particle_stack.parameters.transfer_theory.envelope.b_factor,
            particle_stack.parameters.transfer_theory.envelope.amplitude,
            particle_stack.parameters.pose.offset_x_in_angstroms,
            particle_stack.parameters.pose.offset_y_in_angstroms,
            particle_stack.parameters.pose.view_phi,
            particle_stack.parameters.pose.view_theta,
            particle_stack.parameters.pose.view_psi,
            particle_stack.image_stack,
        )
    else:
        output = (
            particle_stack.parameters.transfer_theory.ctf.defocus_in_angstroms,
            particle_stack.parameters.transfer_theory.ctf.astigmatism_in_angstroms,
            particle_stack.parameters.transfer_theory.ctf.astigmatism_angle,
            particle_stack.parameters.transfer_theory.ctf.phase_shift,
            particle_stack.parameters.pose.offset_x_in_angstroms,
            particle_stack.parameters.pose.offset_y_in_angstroms,
            particle_stack.parameters.pose.view_phi,
            particle_stack.parameters.pose.view_theta,
            particle_stack.parameters.pose.view_psi,
            particle_stack.image_stack,
        )
    return output

@eqx.filter_jit
def compute_loss_single(
    potential, relion_stack_vmap, relion_stack_novmap
):

    relion_stack = eqx.combine(relion_stack_vmap, relion_stack_novmap)

    structural_ensemble = cxs.SingleStructureEnsemble(
        potential, relion_stack.parameters.pose
    )

    scattering_theory = cxs.WeakPhaseScatteringTheory(
        structural_ensemble=structural_ensemble,
        potential_integrator=cxs.FourierSliceExtraction(interpolation_order=1),
        transfer_theory=relion_stack.parameters.transfer_theory,
    )

    imaging_pipeline = cxs.ContrastImagingPipeline(
        relion_stack.parameters.instrument_config, scattering_theory
    )
    distribution = VarianceMarginalizedWhiteGaussianNoise(imaging_pipeline)
    # distribution = WhiteGaussianNoise(imaging_pipeline, noise_variance)

    return distribution.log_likelihood(relion_stack.image_stack)

@eqx.filter_jit
@partial(eqx.filter_vmap, in_axes=(0, None, None), out_axes=0)
def compute_loss_build_pot(atom_positions, relion_stack_vmap, args):

    atom_identities, b_factors, parameter_table, _, relion_stack_novmap = (
        args
    )

    atom_potential = cxs.PengAtomicPotential(
        atom_positions,
        atom_identities,
        b_factors,
        scattering_factor_parameter_table=parameter_table,
    )

    real_voxel_grid = atom_potential.as_real_voxel_grid(
        (relion_stack_novmap.parameters.instrument_config.shape[0],) * 3,
        relion_stack_novmap.parameters.instrument_config.pixel_size,
        batch_size_for_z_planes=10,
    )

    voxel_potential = cxs.FourierVoxelGridPotential.from_real_voxel_grid(
        real_voxel_grid,
        relion_stack_novmap.parameters.instrument_config.pixel_size
    )

    likelihoods = jax.lax.map(
        lambda x: compute_loss_single(
            potential=voxel_potential, relion_stack_vmap=x, relion_stack_novmap=relion_stack_novmap
        ),
        xs=relion_stack_vmap,
        batch_size=2000,
    )

    return likelihoods

@eqx.filter_jit
def compute_loss(
    atom_positions, weights, relion_stack_vmap, args
):
    likelihood_matrix = compute_loss_build_pot(
        atom_positions=atom_positions, relion_stack_vmap=relion_stack_vmap, args=args
    ).T

    log_lklhood = jax.scipy.special.logsumexp(
        a=likelihood_matrix, b=weights[None, :], axis=1
    )
    return -jnp.mean(log_lklhood)


def compute_losses_trajs(
    trajectories,
    dataloader,
    traj_weights,
    filter_spec_for_vmap,
    atom_identities,
    b_factors,
    parameter_table,
):
    losses = np.zeros(trajectories.shape[0])

    n_batches = len(dataloader)

    for batch in tqdm(dataloader):
        relion_stack_vmap, relion_stack_novmap = eqx.partition(
            batch, filter_spec_for_vmap
        )
        logging.info("Running for Batch")

        losses += jax.lax.map(
            lambda x: compute_loss(
                atom_positions=x[0],
                weights=x[1],
                relion_stack_vmap=relion_stack_vmap,
                args=(
                    atom_identities,
                    b_factors,
                    parameter_table,
                    0.0,
                    relion_stack_novmap,
                ),
            ),
            xs=(trajectories, traj_weights),
            batch_size=10,
        )
    return losses / n_batches


def main():
    logger = logging.getLogger()
    logger_fname = "compute_loss_traj.log"
    fhandler = logging.FileHandler(filename=logger_fname, mode="a")
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)

    atom_indices = (
        mda.Universe("../unkwown_poses/Job003/model_0.pdb")
        .select_atoms("protein and not name H*")
        .indices
    )

    stride = 3
    print("Loading trajectories")
    traj1 = mdtraj.load(
        "traj_0.xtc",
        top="../unkwown_poses/Job003/model_0.pdb",
        atom_indices=atom_indices,
        stride=stride,
    )

    traj2 = mdtraj.load(
        "traj_1.xtc",
        top="../unkwown_poses/Job003/model_0.pdb",
        atom_indices=atom_indices,
        stride=stride,
    )

    ref = mdtraj.load(
        "/mnt/home/dsilvasanchez/ceph/projects/cryo_md_project/experiments/groel_cryojax/experimental_data/dataset_for_reconstruction_test_set/ref_model.pdb",
        atom_indices=atom_indices,
    )

    traj1.superpose(ref, frame=0)
    traj2.superpose(ref, frame=0)

    traj1 = traj1.xyz * 10.0
    traj2 = traj2.xyz * 10.0

    trajectories = jnp.stack([jnp.array(traj1), jnp.array(traj2)], axis=1)
    del traj1, traj2
    print("...done")

    logging.info(f"Trajectories shape: {trajectories.shape}")

    _, atom_identities, b_factors = read_atoms_from_pdb_or_cif(
        "../unkwown_poses/Job003/model_0.pdb", get_b_factors=True, atom_filter="all"
    )

    atom_identities = atom_identities[atom_indices]
    b_factors = b_factors[atom_indices]

    parameter_table = get_tabulated_scattering_factor_parameters(atom_identities)

    with h5py.File("../unkwown_poses/Job003/optimize_apo_to_holo.h5", "r") as file:
        dataset = file["trajs_weights"]
        assert isinstance(dataset, h5py.Dataset)
        traj_weights = dataset[:]
    
    traj_weights = jnp.array(traj_weights)[::stride]
    logging.info(f"Traj weights shape: {traj_weights.shape}")

    # ------------------ Train Set ------------------
    metadata = RelionParticleMetadata(
        path_to_starfile="/mnt/home/dsilvasanchez/ceph/projects/cryo_md_project/experiments/groel_cryojax/experimental_data/dataset_for_reconstruction/relion_dataset.star",
        path_to_relion_project="/mnt/home/dsilvasanchez/ceph/projects/cryo_md_project/experiments/groel_cryojax/experimental_data/dataset_for_reconstruction/",
        get_envelope_function=True,
    )

    particle_reader = RelionParticleDataset(metadata)

    filter_spec_for_vmap = _get_particle_stack_filter_spec(particle_reader[0:2])

    dataloader = jdl.DataLoader(
        CustomJaxDataset(
            particle_reader
        ),  # Can be a jdl.Dataset or pytorch or huggingface or tensorflow dataset
        backend="jax",  # Use 'jax' backend for loading data
        batch_size=2000,  # Batch size
        shuffle=False,  # Shuffle the dataloader every iteration or not
        drop_last=False,  # Drop the last batch or not
    )

    losses1 = compute_losses_trajs(
        trajectories,
        dataloader,
        traj_weights,
        filter_spec_for_vmap,
        atom_identities,
        b_factors,
        parameter_table,
    )

    jnp.save("losses1.npy", losses1)

    #- ------------------ Test Set ------------------
    metadata = RelionParticleMetadata(
        path_to_starfile="/mnt/home/dsilvasanchez/ceph/projects/cryo_md_project/experiments/groel_cryojax/experimental_data/dataset_for_reconstruction_test_set/relion_dataset.star",
        path_to_relion_project="/mnt/home/dsilvasanchez/ceph/projects/cryo_md_project/experiments/groel_cryojax/experimental_data/dataset_for_reconstruction_test_set/",
        get_envelope_function=True,
    )

    particle_reader = RelionParticleDataset(metadata)

    filter_spec_for_vmap = _get_particle_stack_filter_spec(particle_reader[0:2])

    dataloader = jdl.DataLoader(
        CustomJaxDataset(
            particle_reader
        ),  # Can be a jdl.Dataset or pytorch or huggingface or tensorflow dataset
        backend="jax",  # Use 'jax' backend for loading data
        batch_size=2000,  # Batch size
        shuffle=False,  # Shuffle the dataloader every iteration or not
        drop_last=False,  # Drop the last batch or not
    )

    losses2 = compute_losses_trajs(
        trajectories,
        dataloader,
        traj_weights,
        filter_spec_for_vmap,
        atom_identities,
        b_factors,
        parameter_table,
    )

    jnp.save("losses2.npy", losses2)

    return


if __name__ == "__main__":
    main()
