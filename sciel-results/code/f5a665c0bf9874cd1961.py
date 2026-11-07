import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
            # Cost-aware UCB for minimization
            # We want to minimize: mu - beta * sigma + lambda * cost
            # So we maximize: beta * sigma - mu - lambda * cost
            
            beta = 1.0
            lambda_cost = 0.5
            
            scores = beta * sigma - mu - lambda_cost * cost
            
            # Ensure finite
            scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
            
            return scores
