import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # LCB: mu - beta * sigma
    # We want to maximize -LCB / cost = (beta * sigma - mu) / cost
    beta = 2.0
    score = (beta * sigma - mu) / (cost + 1e-9)
    return score
