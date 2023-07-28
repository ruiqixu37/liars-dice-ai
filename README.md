# liars-dice-ai
A liar's dice agent using [Monte Carlo Counterfactual Regret Minimization (MCCFR)](https://proceedings.neurips.cc/paper/2009/file/00411460f7c92d2124a67ea0f4cb5f85-Paper.pdf). The implementation is inspired by [Pluribus](https://www.science.org/doi/10.1126/science.aay2400). 

## Model training
To train the model, run `python mccfr.py`. The model will be saved in `output/`. You can adjust the training time by changing the parameter in `mccfr.py`.