# Active learning for deep learning

Torch implementation of entropy, curiosity-driven (expected model change) and goal-driven (expected error redunction) active learning strategies derived under the Bayesian neural network framework. See our [paper](https://arxiv.org/abs/1711.01732) for details. We benchmarked these active learning strategies under the pool-based active learning setting. All active learning strategies are effective in reducing the number of queries on MNIST digit classification and Visual Question Answering (VQA) tasks. 

### Goal-driven learning

We want to put special emphasis to goal-driven learning on VQA because it is a new result for both theory and application. Goal-driven learning by definition, tries to get labels on training examples that are informative for answering the test questions. It is especially useful for open-ended question answering, where there are always clickbait questions whose answers are inspiring but useless for our target task. Goal-driven learning involves mutual information computation which is extremely computationally expensive and people used to think it's impractical. We found an approximation that is fast and works reasonably well. And we experimentally show that on the VQA dataset.


### MNIST 3-layer MLP (784-1200-1200-10)

- 1000 examples: 95.32 (active) vs 92.91 (random)
- 2000 examples: 97.38 (active) vs 94.40 (random)
- 94.40-crossing: 680 examples (active) vs 1930 examples (random)

### VQA v1.0 LSTM+CNN ([model](https://github.com/GT-Vision-Lab/VQA_LSTM_CNN))

- 50k init + 50k examples: 50.3 (active) vs 49.3 (random)
- 50k init + 90k examples: 52.1 (active) vs 51.2 (random)
- 51.0-crossing: 50k init + [ 68k examples (active) vs 88k examples (random) ]

See our [paper](https://arxiv.org/abs/1711.01732) for curiosity- and goal-driven learning results. 

Each active learning run takes 6~8 days on a Tesla K40.

Although these strategy implementations have been online since April 2016, it was not until June 2017 that curiosity- and goal-driven learning are finally justified in experiments since we well under-estimated the amount of data requried.

Finally as an open question, VQA results show that computers still need to see a lot of examples for their accuracy to budge. As humans we likely haven't gone through as many examples in our life time. How can we achieve human-level learning efficiency? 

### Reference

Xiao Lin, Devi Parikh. "Active Learning for Visual Question Answering: An Empirical Study". *arXiv preprint arXiv:1711.01732*, 2017.
