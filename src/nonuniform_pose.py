import jax.numpy as jnp
from jax import random
import numpyro.distributions as dist
from dataclasses import dataclass, asdict
import starfile
import pandas as pd
from numpyro import distributions as dist
from scipy.spatial.transform import Rotation as R
import hydra
from omegaconf import OmegaConf

# Function to export to RELION starfile format
def export_starfile(euler_angles, output_path):
    """
    Save rotation matrices into a RELION-format .star file using starfile.
    """

    # Build a DataFrame
    df = pd.DataFrame({
        'rlnAngleRot': euler_angles[:,0],
        'rlnAngleTilt': euler_angles[:,1],
        'rlnAnglePsi': euler_angles[:,2],
    })
    
    # Write it using starfile
    starfile.write(df, output_path)


# Function to sample mixture of projected normal distributions
def sample_mixture_of_projected_normals(rng_key, n, concentration_top, concentration_side, prob_top, prob_side, **kwargs):
    """
    Samples from a mixture of two projected normal distributions, with centers on the unit sphere.
    
    Parameters:
    - rng_key: JAX PRNG key for random number generation.
    - n: Number of samples to generate.
    - concentration_top: Concentration (sharpness) of the top-centered distribution.
    - concentration_side: Concentration (sharpness) of the side-centered distribution.
    - prob_top: Probability weight of the top-centered distribution.
    - prob_side: Probability weight of the side-centered distribution.
    
    Returns:
    - samples: Generated samples from the mixture.
    """
    # Define centers for the two components
    center_top = jnp.array([0.0, 0.0, 0.0, 1.0])  # Top, along the x-axis
    center_side = jnp.array([1.0, 0.0, 0.0, 1.0])  # Side, along the y-axis
    
    # Stack the components into a batch
    components = dist.ProjectedNormal(
        jnp.stack([center_top * concentration_top, center_side * concentration_side], axis=0)
    )

    # Mixture weights
    mixing_probs = jnp.array([prob_top, prob_side])
    
    # Define the mixture distribution
    mixture = dist.MixtureSameFamily(
        mixing_distribution=dist.Categorical(probs=mixing_probs),
        component_distribution=components,
    )
    
    # Sample from the mixture distribution
    samples = mixture.sample(rng_key, sample_shape=(n,))
    
    return samples


# Define the data class for the parameters
@dataclass
class MixtureParameters:
    n: int = 1
    concentration_top: float = 0.0
    concentration_side: float = 0.0
    prob_top: float = 0.5
    prob_side: float = 0.5
    euler_convention: str = 'ZXZ'
    output_path: str = 'nonuniform_pose.star'


# Main function wrapped with Hydra
import hydra
from pathlib import Path

# Dynamically set the config path based on repo directory
repo_root = Path(__file__).parent.parent  # Navigate to repo root (assuming this script is under /src)
config_path = repo_root / "configs"  # Path to the configs folder
output_dir = repo_root / "outputs"  # Path to outputs directory

@hydra.main(config_path=str(config_path), config_name="pose")
def main(cfg: MixtureParameters):
    rng_key = random.PRNGKey(42)
    
    # Convert the parameters to a dictionary and pass to the sampling function
    params = OmegaConf.to_container(cfg, resolve=True)
    samples = sample_mixture_of_projected_normals(rng_key, **params)
    
    # Process the samples into rotation matrices
    rotations = R.from_quat(samples).as_matrix()
    
    # Sample uniform in-plane rotations
    uniform_in_plane_deg = dist.Uniform(low=-180, high=180).sample(rng_key, sample_shape=(cfg.n,))
    rotations_in_plane = R.from_euler('z', uniform_in_plane_deg, degrees=True).as_matrix()
    
    # Combine in-plane and out-of-plane rotations
    euler_angles_deg = R.from_matrix(rotations_in_plane @ rotations).as_euler(cfg.euler_convention, degrees=True)
    
    # Export to starfile
    export_starfile(euler_angles_deg, cfg.output_path)
    
    # Print output
    print("Euler angles in degrees:\n", euler_angles_deg)
    print(f"Star file saved as '{cfg.output_path}'")


if __name__ == "__main__":
    main()
