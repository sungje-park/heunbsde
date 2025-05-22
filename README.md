# Integration Matters for Learning PDEs with Backwards SDEs

Backward stochastic differential equation (BSDE)-based deep learning methods provide an alternative to Physics-Informed Neural Networks (PINNs) for solving high-dimensional partial differential equations (PDEs), offering potential algorithmic advantages in settings such as stochastic optimal control, where the PDEs of interest are tied to an underlying dynamical system. In this paper, we identify the root cause of this performance gap as a discretization bias introduced by the standard Euler-Maruyama (EM) integration scheme applied to one-step self-consistency BSDE losses, which shifts the optimization landscape off target. We find that this bias cannot be satisfactorily addressed through finer step-sizes or multi-step self-consistency losses. To properly handle this issue, we propose a Stratonovich-based BSDE formulation, which we implement with stochastic Heun integration. We show that our proposed approach completely eliminates the bias issues faced by EM integration. Furthermore, our empirical results show that our Heun-based BSDE method consistently outperforms EM-based variants and achieves competitive results with PINNs across multiple high-dimensional benchmarks. Our findings highlight the critical role of integration schemes in BSDE-based PDE solvers, an algorithmic detail that has received little attention thus far in the literature.

## Training
We include example training code for the Hamilton-Jacobi-Bellman (HJB), Black-Scholes-Barenblatt (BSB), and the Fully-Coupled FBSDE adapted from Bender & Zhang (BZ). The training code can be found in /examples and the loss formulations are found in /src/problems.py.

Additionally, the code is designed to used with [Weights & Biases](https://wandb.ai/site/).

The runfile takes arguments to determine the case, loss, and other configuration settings used in the paper. The following provides an example run for the HJB case.
'''
python -m examples.runfile -c hjb -l bsdeheun
'''
For additional configurations, please refer to
'''
python -m examples.runfile -h
'''
or examples/runfile.py.
## Publication
N/A

## Citation
N/A