# Adaptive stochastic Frank-Wolfe

The Frank-Wolfe method, a prominent first-order optimization algorithm, has gained
traction due to its edge over projected gradient methods, especially in machine learning. However,
its application in non-convex scenarios remains less explored. We delve into two foundational papers:
Reddi et. al.’s Stochastic Frank-Wolfe algorithm (SFW) and Combettes et. al.’s AdaSVRF.
Our computational experiments, using Iris and Obesity datasets, demonstrate AdaSVRF’s superior
performance over SFW, leveraging AdaGrad’s adaptive learning rates and variance reduction. Such
findings highlight AdaSVRF’s potential in optimizing sparse problems with computational efficiency.

## Performance in the Iris Dataset
![iris_performance](https://github.com/AntoniValls/AdaptiveStochasticFrank-Wolfe/assets/101109878/d0cbb770-0a0c-4d01-acbf-c0ff4182846d)

## Performance in the Obesity Dataset
![obesity_performance](https://github.com/AntoniValls/AdaptiveStochasticFrank-Wolfe/assets/101109878/d6294016-6dd0-405d-b49b-56bfc3dff786)

All work was done by [Shab Pompeiano](https://github.com/Shab98) and me.
