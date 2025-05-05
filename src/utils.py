import jax.numpy as jnp
import jax_dataloader as jdl

import logging

from cryojax.io import read_atoms_from_pdb
from cryojax.data import RelionParticleStackDataset, ParticleStack
from cryojax.constants import get_tabulated_scattering_factor_parameters
import cryojax.simulator as cxs


class CustomJaxDataset(jdl.Dataset):
    """
    Used to circumvent dataloader weirdness in Jax? 
    """
    def __init__(self, cryojax_dataset: RelionParticleStackDataset):
        self.cryojax_dataset = cryojax_dataset

    def __getitem__(self, index) -> ParticleStack:
        return self.cryojax_dataset[index]

    def __len__(self) -> int:
        return len(self.cryojax_dataset)


def build_image_formation_stuff(config):
    """
    This is a shortcut script for using a configuration file to build args necessary for the
    'build_distribution_for_particle_parameters' function

    e.g, something like
    config = {
    "number_of_images": 1000,
    "snr": 0.001,
    "weights_models": true_weights,
    "models_fnames": model_fnames, 
    "path_to_models": path_to_models, 
    "path_to_starfile": path_to_starfile,
    "path_to_images": path_to_images,
    "box_size": 128,
    "rng_seed": 0,
    "pixel_size": 1.6,
    }
    """
    # Load up a list of filenames to the .pdb files we want to simulate from
    pdb_fnames = config["models_fnames"]
    path_to_models = config["path_to_models"]

    # Load up the per-pdb weights we want to sample
    weights = jnp.array(config['weights_models'])

    logging.info("Generating potentials...")
    potential_integrator = cxs.FourierSliceExtraction()
    potentials = []
    voxel_size = config["pixel_size"]
    box_size = config["box_size"]
    for i in range(len(pdb_fnames)):

        # Load atomic structure and transform into a potential
        filename = path_to_models + "/" + pdb_fnames[i]
        # Load the atomic structure and transform into a potential
        
        atom_positions, atom_identities, bfactors = read_atoms_from_pdb(
            filename, center=True, select="not element H", loads_b_factors=True
        )
        scattering_factor_parameters = get_tabulated_scattering_factor_parameters(
            atom_identities 
        )
        atomic_potential = cxs.PengAtomicPotential(
            atom_positions,
            scattering_factor_a=scattering_factor_parameters["a"],
            scattering_factor_b=scattering_factor_parameters["b"],
            b_factors=bfactors
        )
        # Convert to a real voxel grid
        # This step is optional, you could use the atomic potential directly!
        real_voxel_grid = atomic_potential.as_real_voxel_grid(
            shape=(box_size, box_size, box_size), voxel_size=voxel_size
        )
        potential = cxs.FourierVoxelGridPotential.from_real_voxel_grid(
            real_voxel_grid, voxel_size, pad_scale=2
        )
        potentials.append(potential)

    potentials = tuple(potentials)
    logging.info("...Potentials generated")

    # Use shortcut script to make an instrument config
    instrument_config = instrument_config_from_params(config)

    # Define args needed  
    args = {}
    args["instrument_config"] = instrument_config
    args["potentials"] = potentials
    args["potential_integrator"] = potential_integrator
    args["weights"] = weights
    return  args


def instrument_config_from_params(config):
    """
    This is a shortcut script for creating an `instrument_config' from a user config file
    """
    box_size = config["box_size"]
    pixel_size = config["pixel_size"]

    instrument_config = cxs.InstrumentConfig(
        shape=(box_size, box_size),
        pixel_size=pixel_size,
        voltage_in_kilovolts=300.0,
        pad_scale=1.0,
    )
    return instrument_config

