import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Cost-Adjusted UCB for minimization
    # UCB_min = mu - beta * sigma
    # We want to minimize UCB_min, so we maximize -UCB_min = beta * sigma - mu
    # Score = (beta * sigma - mu) / cost
    # Adding f_best is a constant shift, doesn't change ranking, but keeps values positive-ish?
    # Actually, mu can be > f_best, making score negative.
    # EI is always >= 0.
    # UCB score can be negative.
    # If all scores are negative, we pick the least negative.
    # This is fine.
    
    beta = 1.0
    score = (beta * sigma - mu) / cost
    return score
