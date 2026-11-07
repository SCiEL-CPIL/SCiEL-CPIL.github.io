import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Cost-Aware Upper Confidence Bound (UCB).
    """
    # Beta parameter for exploration. 
    # Since targets are in objective units (Hartmann 3D range ~4), 
    # a beta of 1.0-2.0 is typical. 
    # Let's use 1.0 to be conservative, or maybe 0.5 if EI was too exploitative?
    # Actually, EI is usually more robust. 
    # Let's try a modified EI that handles small sigma better or UCB.
    
    # Let's try UCB:
    # Improvement = f_best - mu + beta * sigma
    beta = 1.0
    improvement = f_best - mu + beta * sigma
    
    # Avoid division by zero or negative costs (though cost >= 1)
    # cost is (2000,)
    scores = improvement / cost
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
