# Gaussian Process Flavours
Gaussian Process Flavours is a collection of algorithms based on Gaussian Process (GP) Regression implemented in Matlab. Codes are general enough to be used also in different libraries.

## Algorithms
| Algorithm | Code | Description |
|-----------|------|-------------|
| Kernel Eigenvalue Decomposition | KernelDecomposition.m | GP kernel eigenvalue decomposition based on *Mercer's Theorem* |
| Fast Approximate GPIS | FAGP.m | GP Implicit Surface algorithm relying on kernel eigenvalue decomposition to speed up computation |
| Logarithmic GPIS in 2D | LogGPIS2D.m | GP Implicit Surface providing accurate description of the distance from obstacles exploiting gradient information |
| Online Recursive GPIS | recursiveGPIS.m | Online computation of GP suitable for data streams |

The mathematical description can be found in [this pdf file.](./doc/GPflavours.pdf)