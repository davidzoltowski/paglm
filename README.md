# paGLM : Poisson polynomial approximate GLM

The code is under construction.

This package implements fast approximate inference for Poisson GLMs using polynomial approximate sufficient statistics. The package handles Poisson GLMs with both exponential and alternative nonlinearities.

Reference: 
Zoltowski, David, and Jonathan W. Pillow. [Scaling the Poisson GLM to massive neural datasets through polynomial approximations.](http://pillowlab.princeton.edu/pubs/abs_Zoltowski_NIPS18.html) *Advances in Neural Information Processing Systems.* 2018.

The PASS-GLM approach was proposed in: 
Huggins, Jonathan, Ryan P. Adams, and Tamara Broderick. [PASS-GLM: polynomial approximate sufficient statistics for scalable Bayesian GLM inference.](https://arxiv.org/abs/1709.09216) *Advances in Neural Information Processing Systems.* 2017.

# Installation

```
git clone https://github.com/davidzoltowski/paglm.git
cd paglm
pip install -e .
```
