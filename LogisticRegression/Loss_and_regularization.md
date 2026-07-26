# Logistic Regression: Loss and Regularization

> Source: [Google ML Crash Course – Loss and Regularization](https://developers.google.com/machine-learning/crash-course/logistic-regression/loss-regularization)

## Same Training Process, Two Key Differences

Logistic regression is trained the same way as linear regression (gradient descent, weights/bias updates), with **two exceptions**:
1. Uses **Log Loss** instead of squared loss (MSE).
2. **Regularization** is critical — not optional like it was for linear regression.

## Why Not Squared Loss?

Squared loss works well when the rate of change is **constant** — e.g. `y' = b + 3x1`: every +1 to `x1` always adds exactly +3 to `y'`.

But the sigmoid curve is **not constant** — it's s-shaped. Near `z = 0`, small changes in `z` cause big swings in output. But near the extremes (`z` very positive or very negative), huge changes in `z` barely move the output at all.

This creates a precision problem:

| z (log-odds) | sigmoid output | digits of precision needed |
|---|---|---|
| 5 | 0.993 | 3 |
| 7 | 0.999 | 3 |
| 8 | 0.9997 | 4 |
| 10 | 0.99998 | 5 |

As predictions push toward 0 or 1, you'd need increasingly more decimal precision just to measure error with squared loss — impractical and memory-heavy.

## Log Loss (the fix)
Log Loss = -(1/N) * Σ [ yi * log(yi') + (1 - yi) * log(1 - yi') ]
Where:
- `N` = number of examples
- `yi` = true label (must be exactly **0 or 1**, since this is a binary outcome)
- `yi'` = model's predicted probability (between 0 and 1)

**Intuition**: instead of measuring raw distance (like squared loss), Log Loss measures the *logarithm* of the error — which naturally expands the scale near 0 and 1, exactly where squared loss falls short.

- If `yi = 1`: only the `yi * log(yi')` term matters. If the model predicted close to 1, loss is tiny. If it predicted close to 0 (very wrong), `log()` of a near-zero number is a huge negative number → loss explodes.
- If `yi = 0`: only the `(1-yi) * log(1-yi')` term matters — same logic, mirrored.

This is why Log Loss punishes **confident wrong predictions** especially harshly.

## Why Regularization Is Critical Here

Without regularization, logistic regression's loss can **never fully reach 0** (sigmoid never outputs exactly 0 or 1) — so during training, gradient descent keeps pushing weights higher and higher trying to get infinitely closer to 0 loss. With many features, this drives weights toward `+∞` or `-∞`, badly overfitting the training data.

Two common fixes:
- **L2 regularization** — penalizes large weights directly (covered in more depth in the Overfitting module).
- **Early stopping** — simply stop training before the model has a chance to overfit.

## My Takeaways
- [ ] Log Loss ≠ squared loss — it's log-based specifically because sigmoid's rate of change isn't constant.
- [ ] Log Loss heavily penalizes confident + wrong predictions (predicting 0.01 when the true label is 1 is very costly).
- [ ] `yi` is strictly 0 or 1 — this is what makes the two-term formula collapse to "only the relevant term counts."
- [ ] Regularization isn't optional in logistic regression — without it, weights can blow up toward infinity trying to chase an unreachable 0 loss.
- [ ] L2 regularization and early stopping are the two standard fixes (full detail comes later in the Overfitting module).
