The goal of this assignment is to build a bigram model from scratch then implement a slightly better model (Bengio et al., 2003).

The bigram model, while very simple, offers a good anchor point to understand the Transformer.

The assignment is based on makemore by Andrej Karpathy.

Assignment setup:
- Install torch: `pip install torch` (see more [instructions](https://pytorch.org/get-started/locally/))

Exercises:
1. Build a simple bigram model for next-character prediction
2. Build the same bigram model using the NLL loss
3. homework (*) Extend the model to trigram
4. Implement a better model: [Bengio et al., 2003](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
5. homework (*): add batch norm to your network

