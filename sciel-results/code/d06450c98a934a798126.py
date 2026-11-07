import numpy as np
import time

_start_time = time.time()
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best, cost):
    # Time guard
    current_time = time.time()
    if current_time - _start_time > _DEADLINE:
        return np.random.rand(len(mu)) * 1e-6 + 1e-9

    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    cost = np.asarray(cost, dtype=np.float64).ravel()
    
    cost_safe = np.maximum(cost, 1e-9)
    
    # Adaptive exploration parameter
    # We want high exploration early (when total cost is low) and low exploration late.
    # Total budget is 30.
    # Let's estimate total cost. The 'cost' array passed in likely represents the cost 
    # of evaluating each candidate? No, the prompt says "cost have shape (2000,)".
    # "Evaluating x costs 1 + mean(x scaled to [0, 1])."
    # "Runs start with 5 uniform random points (cost counted)."
    # "While total cost is below 30..."
    # The input 'cost' is likely the cost associated with each candidate point x_i.
    # So we can estimate the total cost spent so far? 
    # Actually, we don't know how many points have been evaluated.
    # However, we can use the magnitude of sigma or the number of points?
    # A simpler heuristic: 
    # If we assume the 'cost' array is just the cost of the candidates, 
    # we don't have direct access to the number of iterations or total spent cost.
    # BUT, the GP's sigma is typically high in unexplored regions.
    # A common strategy is to use a fixed kappa that works well, or one that decays.
    # Without iteration count, we can't easily decay based on time.
    # However, we can look at the distribution of sigma.
    # If most sigma is high, we are early. If low, we are late.
    # Let's try a moderate fixed kappa first, but weighted by cost.
    # The current program uses max(EI/cost, UCB/cost).
    # My strategy: Pure LCB / cost.
    # LCB = mu - kappa * sigma. We want to maximize -LCB = -mu + kappa * sigma.
    
    # Let's refine kappa.
    # For Griewank, which is multimodal but with a global minimum at 0 (shifted),
    # aggressive exploration is good.
    # Let's try kappa = 2.0.
    
    kappa = 2.0
    
    # Calculate score
    # We want to maximize the potential improvement while accounting for cost.
    # Score = (-mu + kappa * sigma) / cost
    scores = (-mu + kappa * sigma) / cost_safe
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
