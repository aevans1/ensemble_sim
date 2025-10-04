This is currently a repository of my own uses cases and wrappers for the [cryoJAX](https://michael-0brien.github.io/cryojax/) library for simulating cryo-EM images.

NOTE: The existing code in `src/ensemble_sim.py` is largely adapted directly from the cryojax tutorial [here](https://github.com/michael-0brien/cryojax/blob/main/docs/examples/dev/simulating-and-reweighting-ensembles.ipynb)

There is a .toml file with dependencies, but some things should be installed from source:
- [jax](https://docs.jax.dev/en/latest/installation.html)
  - note: for gpu, so far its expected this is cuda12, so use `pip install -U "jax[cuda12]"`

You can try getting the other packages, and installing project, via:
```
git clone https://github.com/aevans1/ensemble_sim
cd ensemble_sim
python -m pip install .
```
