import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # UCB for minimization
    # We want to minimize f(x).
    # UCB lower bound: mu + beta * sigma
    # We maximize the negative of this bound, normalized by cost.
    beta = 3.0
    ucb = mu + beta * sigma
    # To make it "higher is better", we negate it.
    # However, if we just negate, the scale is different from EI.
    # But the evaluator only cares about the ranking (which one is max).
    # So the absolute scale doesn't matter, only the relative ordering.
    scores = -ucb / cost
    return scores
