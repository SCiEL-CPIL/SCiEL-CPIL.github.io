import numpy as np

def acquisition(mu, sigma, f_best, cost):
        mu = np.asarray(mu, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        cost = np.asarray(cost, dtype=float)
        
        # Safety for cost
        cost_safe = np.maximum(cost, 1e-9)
        
        # Approximation of expected minimum
        # E[min(f, f_best)] ≈ f_best - 0.5 * max(0, f_best - mu - 0.5*sigma)
        # Reduction in regret = f_best - E[min] = 0.5 * max(0, f_best - mu - 0.5*sigma)
        
        # Calculate the term
        # We want to maximize this reduction per unit cost
        term = f_best - mu - 0.5 * sigma
        reduction = 0.5 * np.maximum(0.0, term)
        
        scores = reduction / cost_safe
        
        # Handle NaNs/Infs
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=0.0)
        
        return scores
