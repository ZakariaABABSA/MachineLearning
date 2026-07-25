
# Linear Regression: Hyperparameters

> Source: [Google ML Crash Course – Linear Regression: Hyperparameters](https://developers.google.com/machine-learning/crash-course/linear-regression/hyperparameters)

## Hyperparameters vs. Parameters

- **Parameters** = values the model *learns* on its own during training (weight, bias).
- **Hyperparameters** = values *you* set before training even starts, to control how training happens.

Three key hyperparameters: **learning rate**, **batch size**, **epochs**.

## 1. Learning Rate

The learning rate is the "small amount" from gradient descent — the multiplier applied to the slope to decide how big a step to take each iteration.
step_size = gradient * learning_rate
Example: gradient magnitude = 2.5, learning rate = 0.01 → parameter changes by 0.025.

**Three possible outcomes:**

| Learning rate | What happens |
|---|---|
| Too low | Converges, but very slowly (many iterations wasted) |
| Just right | Converges quickly and smoothly |
| Too high | Never converges — loss bounces around or even increases over time |

There's no single "correct" learning rate — it's **problem-dependent**, tied to the specific model and dataset.

## 2. Batch Size

Batch size = how many training examples the model looks at **before updating** weight/bias.

Processing the *entire* dataset before every single update (full-batch) is accurate but too slow/expensive at scale (datasets with 100k+ examples). Two alternatives:

### Stochastic Gradient Descent (SGD)
- Batch size = **1** (one random example per iteration).
- Fast per-step, but very **noisy** — loss can jump around, not decrease smoothly, even far from convergence.

### Mini-batch SGD
- Batch size = **somewhere between 1 and N** (the full dataset size).
- A compromise: averages gradients over a small random batch, then updates once.
- Small batch → behaves more like SGD (more noise). Large batch → behaves more like full-batch (smoother, slower).
- **Bonus insight**: larger batches also help average out the effect of outliers in the data, since one weird example gets diluted among many normal ones.

**Important reframe**: noise isn't automatically bad. Some noise can actually help a model generalize better and avoid getting stuck — this becomes more relevant later with neural networks.

## 3. Epochs

One **epoch** = the model has seen every example in the training set exactly once.

Example: 1,000 examples, mini-batch size of 100 → 10 iterations = 1 epoch.

- Training usually needs **many epochs** (multiple full passes over the data).
- More epochs → generally better model, but longer training time. This is itself a hyperparameter you experiment with.

### How batch size + epochs together determine update count

| Batch type | Update frequency | Example: 1,000 examples, 20 epochs |
|---|---|---|
| Full batch | Once per epoch | 20 updates total |
| SGD (batch=1) | Once per example | 20,000 updates total |
| Mini-batch (batch=100) | Once per batch | 200 updates total |

## Quick Self-Check (from the lesson)

- **"What's the ideal learning rate?"** → It depends — every model/dataset combination has its own ideal value.
- **"Do larger batches handle outliers worse?"** → False. Larger batches *reduce* the negative effect of outliers by averaging more gradients together.
- **"Does doubling the learning rate slow down training?"** → Can be true — too large a rate causes bouncing, which *increases* time to converge, ironically.

## My Takeaways
- [ ] Parameters = learned by the model. Hyperparameters = set by me before training.
- [ ] Learning rate controls step size — too low is slow, too high never converges.
- [ ] Batch size trades off speed vs. noise: SGD (1) = noisy/fast, full-batch = smooth/slow, mini-batch = practical middle ground.
- [ ] Epoch = one full pass through all training data; multiple epochs are normal.
- [ ] These three hyperparameters directly interact — batch size and epoch count together determine how many times the model actually updates its weights.
