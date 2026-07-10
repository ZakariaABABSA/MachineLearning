# Linear Regression — Summary

**Source:** [Google ML Crash Course – Linear Regression](https://developers.google.com/machine-learning/crash-course/linear-regression)
**Estimated module length:** 80 minutes

## Overview

Linear regression is a statistical technique used to find the relationship between variables. In an ML context, it finds the relationship between **features** (inputs) and a **label** (the value to predict).

## Example: Predicting fuel efficiency

Given a dataset of car weight (feature) vs. miles per gallon (label), plotting the points shows a downward-sloping trend — heavier cars tend to have lower MPG. We can draw a best-fit line through the data to create a simple model.

## The Linear Regression Equation

**Algebraic form (single variable line):**
y = mx + b
- `y` = miles per gallon (value to predict)
- `m` = slope of the line
- `x` = pounds (input value)
- `b` = y-intercept

**ML form:**
y' = b + w1*x1
- `y'` = the predicted label (output)
- `b` = **bias** — same concept as the y-intercept; a *parameter* learned during training (sometimes written as `w0`)
- `w1` = **weight** of the feature — same concept as the slope; also a *parameter* learned during training
- `x1` = the **feature** (input)

### Worked example
For the car dataset: `bias = 34`, `weight = -4.6`
y' = 34 + (-4.6)(x1)
A 4,000-lb car → predicted fuel efficiency ≈ **15.6 mpg**.

## Models with Multiple Features

Real models usually use more than one feature, each with its own weight:
y' = b + w1x1 + w2x2 + w3x3 + w4x4 + w5*x5
Example — predicting MPG using:
- Engine displacement
- Acceleration
- Number of cylinders
- Horsepower
- Weight

Each of these features can individually show a linear relationship with the label (e.g., bigger engine displacement → lower MPG; slower acceleration time → higher MPG).

## What Gets Updated During Training?

| Component | Updated during training? |
|---|---|
| Bias & weights | ✅ Yes — these are the parameters the model learns |
| Predictions (`y'`) | ❌ No — these are outputs, computed from current parameters |
| Feature values (`x`) | ❌ No — these come from the dataset, fixed inputs |

## Key Terms

- **Bias** — the model's y-intercept-like parameter, learned during training
- **Feature** — an input variable used to make a prediction
- **Label** — the true/target value the model tries to predict
- **Linear regression** — a technique for modeling the relationship between features and a label using a linear equation
- **Parameter** — a value (like bias or weight) the model learns during training
- **Weight** — the coefficient multiplied by a feature's value, learned during training

