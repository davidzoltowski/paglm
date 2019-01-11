# paGLM : Poisson polynomial approximate GLM

This package implements fast approximate inference for Poisson GLMs using polynomial approximate sufficient statistics. The package handles Poisson GLMs with both exponential and alternative nonlinearities.

Reference: [Scaling the Poisson GLM to massive neural datasets through polynomial approximations.](http://pillowlab.princeton.edu/pubs/abs_Zoltowski_NIPS18.html)

Original reference for PASS-GLM: [PASS-GLM: polynomial approximate sufficient statistics for scalable Bayesian GLM inference.](https://arxiv.org/abs/1709.09216)

# Installation

```
git clone https://github.com/davidzoltowski/paglm.git
cd paglm
pip install -e .
```
