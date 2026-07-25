# Linear Regression: Loss

> Source: [Google ML Crash Course – Linear Regression: Loss](https://developers.google.com/machine-learning/crash-course/linear-regression/loss)

## What is Loss?

Loss is a number that tells you how wrong a model's predictions are compared to the real (actual) values. The goal of training any model is to **minimize loss** — get it as close to zero as possible.

Think of it visually: for every data point, draw an arrow from that point to the model's line. The length of that arrow is the loss for that point.

## Why We Ignore the Sign

If the model predicts `2` but the real value is `5`, the raw difference is `-3`. But we don't care about direction (over-prediction vs under-prediction) — we only care about **distance**. So every loss function removes the sign, using one of two tricks:

- **Absolute value**: `|actual - predicted|`
- **Squaring**: `(actual - predicted)^2`

## The 5 Main Loss Types

| Name | What it is | Formula |
|---|---|---|
| **L1 loss** | Sum of absolute differences | `Σ \|actual - predicted\|` |
| **MAE** (Mean Absolute Error) | Average of L1 loss over N examples | `(1/N) Σ \|actual - predicted\|` |
| **L2 loss** | Sum of squared differences | `Σ (actual - predicted)^2` |
| **MSE** (Mean Squared Error) | Average of L2 loss over N examples | `(1/N) Σ (actual - predicted)^2` |
| **RMSE** (Root Mean Squared Error) | Square root of MSE | `sqrt((1/N) Σ (actual - predicted)^2)` |

**Key idea:** the only real difference between "L1-family" (MAE) and "L2-family" (MSE) is **squaring**.
- Squaring a big error makes it *much* bigger.
- Squaring a small error (< 1) makes it *even smaller*.

This means MSE reacts strongly to large errors, while MAE treats all errors proportionally.

## Worked Example

Model: `y' = 34 + (-4.6) * x1` (predicting MPG from car weight in 1000s of lbs)

For a 2,370-lb car (x1 = 2.37):

| Value | Calculation | Result |
|---|---|---|
| Prediction | `34 + (-4.6 * 2.37)` | 23.1 |
| Actual value | given | 24 |
| L2 loss | `(24 - 23.1)^2` | **0.81** |

## Choosing Between MAE and MSE

This depends on how you want your model to react to **outliers** (data points far outside the normal range, or points the model predicts very badly).

- **MSE** → pulls the model *toward* outliers (fits them more, at the cost of typical points). Squaring makes big errors dominate the total loss.
- **MAE** → keeps the model *closer to the bulk of normal points*, more robust against outliers.

**Use MSE when:**
- You want to heavily penalize large errors.
- Outliers are meaningful and represent real variance you want the model to capture.
- Note: MSE tends to optimize more smoothly. RMSE is often used afterward to bring error back to the original units.

**Use MAE when:**
- Your dataset has significant outliers you don't want to dominate training.
- You want a loss value that's easy to interpret directly (it's literally "average error" in the label's units).

## Quick Self-Check

Two models fit the same 10 points:
- **Model A**: 6 points exactly on the line, 4 points off by 1 unit each.
  `MSE = (0+1+0+1+0+1+0+1+0+0)/10 = 0.4`
- **Model B**: 8 points exactly on the line, 2 points off by 2 units each.
  `MSE = (0+0+0+4+0+0+0+4+0+0)/10 = 0.8`

**Model B has higher MSE**, even though fewer points are off the line — because squaring punishes the size of the error, not the count of errors.

## My Takeaways
- [ ] Loss = distance between prediction and truth, never signed.
- [ ] MAE = robust, human-interpretable, less sensitive to outliers.
- [ ] MSE = punishes big mistakes hard, pulls model toward outliers.
- [ ] RMSE = MSE but back in original units (easier to interpret than raw MSE).
- [ ] Choice of loss function is a modeling decision tied to how much outliers should matter.
