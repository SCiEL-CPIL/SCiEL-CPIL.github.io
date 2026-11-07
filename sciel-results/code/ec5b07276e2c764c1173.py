import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Lower Confidence Bound (LCB) per unit cost
    # LCB(x) = mu(x) - beta * sigma(x)
    # We want to minimize LCB, so we maximize -LCB.
    # We divide by cost to be cost-aware.
    beta = 2.0
    lcb = mu - beta * sigma
    # Ensure cost is not zero (it's >= 1)
    score = -lcb / cost
    return score
