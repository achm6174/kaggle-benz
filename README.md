## Summary
This is our solution for Kaggle: Mercedes-Benz Greener Manufacturing.

## Problem
In this competition, Daimler is challenging Kagglers to tackle the curse of dimensionality and reduce the time that cars spend on the test bench. Competitors will work with a dataset representing different permutations of Mercedes-Benz car features to predict the time it takes to pass testing. Winning algorithms will contribute to speedier testing, resulting in lower carbon dioxide emissions without reducing Daimler’s standards.

## Our solution
The summary of the approach is to create first layer of models from linear (underfitting) to nonlinear (overfitting), and then ensemble it in second layer with convex optimization.

As many teams overfit in this challenge, our approach might provide some insights on how to perform a better local evaluation.

## Instruction
#### Download Data
* download `test.csv`, `sample_submission.csv`, `train.csv`, and put into folder `./data`.

#### Run all model
* run `sh run.sh`
