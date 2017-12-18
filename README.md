# Bayesian-Compression-for-Deep-Learning
Remplementation of paper [https://arxiv.org/abs/1705.08665](https://arxiv.org/abs/1705.08665). This repo utilizes code from [Bayesian Optimization](https://github.com/fmfn/BayesianOptimization) and [Bayesian Tutorial](https://github.com/KarenUllrich/Tutorial_BayesianCompressionForDL)

# Results

Network | Dataset | Epochs | Accuracy(before) | Accuracy(after) | Compression Rate | 
--- | --- | --- | --- | --- | ---
3-Layer MLP | MNIST | 50 |  98.33% | 98.33% | 1.3
3-Layer MLP | CIFAR 10 | 50 | 56.26% | 54.47% | 3.5
LeNet | MNIST | 50 | 99.24% | 99.26% | 1.4

# Usage
```bash
python example.py --dataset mnist --nettype mlp --epochs 50
```

# TODO:
1. Clean the train-prune code
2. Modularize Bayesian Layer and Module
3. Fix bug in Convolution