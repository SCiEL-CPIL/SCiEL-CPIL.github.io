import numpy as np
import time

_start_time = time.time()
_TIME_LIMIT = 300.0
_MARGIN = 5.0

def acquisition(mu, sigma, f_best, cost):
        # Deadline guard
        if time.time() - _start_time > _TIME_LIMIT - _MARGIN:
            # Fallback: Pure exploration / random-ish
            # Return high scores for high sigma
            sigma_safe = np.maximum(sigma, 1e-8)
            cost_safe = np.maximum(cost, 1e-8)
            return sigma_safe / cost_safe

        sigma_safe = np.maximum(sigma, 1e-8)
        cost_safe = np.maximum(cost, 1e-8)
        
        # Cost-Aware LCB (UCB for minimization)
        # We want to minimize mu - beta*sigma
        # So we maximize -(mu - beta*sigma) / cost = (beta*sigma - mu) / cost
        beta = 2.0
        
        scores = (beta * sigma_safe - mu) / cost_safe
        
        # Ensure finite
        scores = np.where(np.isfinite(scores), scores, 0.0)
        
        return scores
