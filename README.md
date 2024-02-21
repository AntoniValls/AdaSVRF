# Adaptive stochastic Frank-Wolfe

The Frank-Wolfe method, a prominent first-order optimization algorithm, has gained
traction due to its edge over projected gradient methods, especially in machine learning. However,
its application in non-convex scenarios remains less explored. We delve into two foundational papers:
Reddi et. al.’s Stochastic Frank-Wolfe algorithm (SFW) and Combettes et. al.’s AdaSVRF.
Our computational experiments, using Iris and Obesity datasets, demonstrate AdaSVRF’s superior
performance over SFW, leveraging AdaGrad’s adaptive learning rates and variance reduction. Such
findings highlight AdaSVRF’s potential in optimizing sparse problems with computational efficiency.
