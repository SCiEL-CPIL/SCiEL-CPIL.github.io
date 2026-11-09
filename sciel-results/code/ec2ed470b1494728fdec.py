import numpy as np
import time
from scipy.optimize import linprog

def heuristics(prizes, weights):
    """
    Computes heuristic values for items in a multiple-constraint knapsack problem.
    Combines LP dual shadow prices with a sharpened prize-to-weight ratio.
    """
    start_time = time.time()
    
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    n_items = weights.shape[1]
    n_constraints = weights.shape[0]
    capacity = 25.0
    
    # Deadline guard: if we're running too long, return a simple heuristic
    if time.time() - start_time > 300:
        total_prizes = prizes.sum(axis=0)
        total_weights = weights.sum(axis=0)
        ratio = total_prizes / (total_weights + 1e-9)
        scores = ratio ** 2.0
        scores = np.where(np.isfinite(scores) & (scores > 0), scores, 1e-9)
        return scores
    
    # Solve the dual of the continuous relaxation to get shadow prices
    # Dual: Min sum(lambda_j * C_j) s.t. sum(lambda_j * w_ji) >= p_i, lambda_j >= 0
    c = np.full(n_constraints, capacity)
    A_ub = -weights.T 
    b_ub = -prizes
    bounds = [(0, None)] * n_constraints
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            lambda_vals = res.x
        else:
            lambda_vals = np.ones(n_constraints)
    except Exception:
        lambda_vals = np.ones(n_constraints)
        
    # Calculate the "cost" of each item based on shadow prices
    costs = weights.T @ lambda_vals
    
    # Prevent division by zero
    costs = np.maximum(costs, 1e-9)
    
    # Base heuristic: Profit / Cost from LP dual
    eta_base = prizes / costs
    eta_base = np.maximum(eta_base, 1e-9)
    
    # Also compute the simple prize-to-weight ratio (summed across constraints)
    total_prizes = prizes.sum(axis=0)
    total_weights = weights.sum(axis=0)
    ratio = total_prizes / (total_weights + 1e-9)
    
    # Combine: use the LP-based heuristic but sharpen it
    # The LP dual gives per-constraint shadow prices, which is more nuanced
    # but the simple ratio captures the overall value density
    
    # Normalize both to have similar scale, then combine
    eta_base_norm = eta_base / (eta_base.max() + 1e-9)
    ratio_norm = ratio / (ratio.max() + 1e-9)
    
    # Blend: give more weight to the LP-based heuristic as it accounts for constraint tightness
    eta_combined = 0.6 * eta_base_norm + 0.4 * ratio_norm
    
    # Sharpen with power to create stronger differentiation
    # Power > 1 makes high-attractiveness items stand out more
    scores = eta_combined ** 2.0
    
    # Ensure all scores are positive (task says non-positive become 1e-9)
    scores = np.where(scores > 0, scores, 1e-9)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 1e-9)
    
    # Final deadline check
    if time.time() - start_time > 310:
        # Fallback: return the simple sharpened ratio
        fallback = ratio ** 2.0
        fallback = np.where(np.isfinite(fallback) & (fallback > 0), fallback, 1e-9)
        return fallback
        
    return scores
