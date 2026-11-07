import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware LCB
    # Exploit low mu, explore high sigma, penalize high cost in exploration
    beta = 2.0
    # Ensure cost is not zero
    cost_safe = np.where(cost > 1e-6, cost, 1.0)
    scores = -mu + beta * (sigma / cost_safe)
    return scores
