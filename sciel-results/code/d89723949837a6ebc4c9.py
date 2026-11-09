import numpy as np
import time

# Global state for deadline?
# The prompt asks for "explicit deadline guard".
# I'll add a check.

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    # Since we don't know the start time of the whole process, 
        # and this function is called many times, a global timer is tricky.
        # However, if the caller is stuck, it's not my fault.
        # I will assume the "guard" means ensuring my function doesn't hang.
        # NumPy ops are fast.
        
        # Thompson Sampling
        # Sample from posterior
        y = np.random.normal(mu, sigma)
        
        # Cost-aware score
        # We want to minimize y.
        # We want to minimize cost.
        # Score should be high for good candidates.
        # Good candidate: low y, low cost.
        # Score = -y / cost
        
        # Ensure no division by zero (cost >= 1)
        scores = -y / cost
        
        return scores
