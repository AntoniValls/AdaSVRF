# Adaptive stochastic Frank-Wolfe

The Frank-Wolfe method, a prominent first-order optimization algorithm, has gained
traction due to its edge over projected gradient methods, especially in machine learning. However,
its application in non-convex scenarios remains less explored. We delve into two foundational papers:
Reddi et. al.’s Stochastic Frank-Wolfe algorithm (SFW) and Combettes et. al.’s AdaSVRF.
Our computational experiments, using Iris and Obesity datasets, demonstrate AdaSVRF’s superior
performance over SFW, leveraging AdaGrad’s adaptive learning rates and variance reduction. Such
findings highlight AdaSVRF’s potential in optimizing sparse problems with computational efficiency.

The report with all the references can be found [here](https://github.com/AntoniValls/AdaptiveStochasticFrank-Wolfe/blob/main/Report.pdf). 

All work was done by [Shab Pompeiano](https://github.com/Shab98) and me.

## Performance on the Iris Dataset
![iris_performance](https://github.com/AntoniValls/AdaptiveStochasticFrank-Wolfe/assets/101109878/d0cbb770-0a0c-4d01-acbf-c0ff4182846d)

## Performance on the Obesity Dataset
![obesity_performance](https://github.com/AntoniValls/AdaptiveStochasticFrank-Wolfe/assets/101109878/d6294016-6dd0-405d-b49b-56bfc3dff786)

## Colab link
A deeper explanation about the code can be found at the following colab: https://colab.research.google.com/drive/1nxWqzR-43uCN7jucs2clG4hBHgStOm4u?usp=sharing

