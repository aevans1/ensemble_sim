NOTE: The existing code in `src/ensemble_sim.py` is largely adapted directly from the cryojax tutorial [here](https://github.com/mjo22/cryojax/blob/340-write-an-ensemble-reweighting-tutorial/docs/examples/simulating-and-reweighting-ensembles.ipynb)


There is a .toml file with dependencies, but some things should be installed from source:
- `cryojax` should be installed from source, [here](https://github.com/mjo22/cryojax)
- probably other things! 

You can try getting the other packages, and installing project, via:

```
git clone https://github.com/aevans1/ensemble_sim
cd ensemble_sim
python -m pip install .
```

For the source estimation notebook, you will also need the "cryo_reweighting" package, or implement the short functions in there yourself, see [here](https://github.com/Quantitative-Heterogeneity/cryo_reweighting/blob/main/src/cryo_reweighting/optimization.py)
```
git clone https://github.com/Quantitative-Heterogeneity/cryo_reweighting.git
cd cryo_reweighting
python -m pip install .
```

