# Linear Regression: Gradient Descent

> Source: [Google ML Crash Course – Linear Regression: Gradient Descent](https://developers.google.com/machine-learning/crash-course/linear-regression/gradient-descent)

## What Is Gradient Descent?

Gradient descent is the algorithm that actually **finds** the weight and bias values that minimize loss — instead of guessing them by hand like we did with `bias = 34, weight = -4.6`.

It starts with weight and bias near zero (randomized), then repeats this loop:

1. **Calculate loss** with the current weight and bias.
2. **Determine direction** — which way to nudge weight/bias to reduce loss.
3. **Update** weight and bias by a small step in that direction.
4. **Repeat** until loss stops going down meaningfully.

## The Math (short version)

For each parameter (weight and bias), you take the **derivative** of the loss function — this gives you the *slope* at the current point. The slope tells you which direction increases loss, so you move in the **opposite (negative)** direction.

Update rule:
new_weight = old_weight - (learning_rate * weight_slope)
new_bias   = old_bias   - (learning_rate * bias_slope
Here, `learning_rate` (called "small amount" in the lesson, 0.01 in the example) controls how big each step is.

## Worked Example (same car dataset)

| weight(lbs/1000) | MPG |
|---|---|
| 3.5 | 18 |
| 3.69 | 15 |
| 3.44 | 18 |
| 3.43 | 16 |
| 4.34 | 15 |
| 4.42 | 14 |
| 2.37 | 24 |

Starting at `weight = 0, bias = 0`:
- Initial MSE loss = **303.71**
- Weight slope = **-119.7**, Bias slope = **-34.3**
- With learning rate 0.01:
  - `new_weight = 0 - (0.01 * -119.7) = 1.2`
  - `new_bias = 0 - (0.01 * -34.3) = 0.34`

Repeating this process, loss drops each iteration:

| Iteration | Weight | Bias | MSE Loss |
|---|---|---|---|
| 1 | 0 | 0 | 303.71 |
| 2 | 1.20 | 0.34 | 170.84 |
| 3 | 2.05 | 0.59 | 103.17 |
| 4 | 2.66 | 0.78 | 68.70 |
| 5 | 3.09 | 0.91 | 51.13 |
| 6 | 3.40 | 1.01 | 42.17 |

Notice: this is heading toward my earlier hardcoded values (`weight=-4.6, bias=34`) — gradient descent would eventually get there (and beyond, to the true optimum) if run for enough iterations.

## Convergence

- **Convergence** = the point where more iterations barely reduce loss anymore.
- Past convergence, loss just jitters slightly around its lowest point instead of dropping — that jitter is a sign you're basically done, not a sign of a bug.
- A **loss curve** (loss on y-axis, iteration number on x-axis) is the standard way to check this visually: steep drop early, then flattens out.

## Why Linear Regression Always Converges to the *Best* Answer

The loss function for a linear model is **convex** — shaped like a bowl, not full of bumps/valleys. Practical implication: there's only **one minimum**, so gradient descent can't get "stuck" in a fake local minimum. It will always reach the actual best weight/bias for the data (or something extremely close to it).

Important nuance: the minimum loss is basically never exactly 0. If it were 0, the model would fit every point perfectly — which the course flags as a red flag for **overfitting**, not a good sign (covered later in the course).

## My Takeaways
- [ ] Gradient descent = loss → slope → step → repeat, until loss stops improving.
- [ ] The step size is controlled by the learning rate (a hyperparameter, covered next lesson).
- [ ] Convergence = loss curve flattens out.
- [ ] Linear regression's loss surface is convex → gradient descent is guaranteed to find the global best fit, not just a local one.
- [ ] Loss of exactly 0 is suspicious (overfitting), not ideal.
