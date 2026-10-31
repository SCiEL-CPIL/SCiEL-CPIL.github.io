import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Calculate remaining space if item is placed in each bin
    r = bins - item
    
    # Initialize scores
    # We use a "dead space" heuristic based on Weibull distribution assumptions.
    # Score is negative expected waste.
    # Waste(r) = r * P(X > r). For Weibull(k=2, lambda=30), P(X>r) = exp(-(r/30)^2).
    # We add a small linear penalty to prefer filling bins (r=0) over empty bins (r=C) in case of ties.
    
    # Scale parameter for Weibull, estimated based on capacity 100 and typical item sizes
    lambda_w = 30.0
    k_w = 2.0
    epsilon = 0.1
    
    # Calculate base score
    # Avoid division by zero or negative base in exp if r < 0, but we mask those anyway.
    # For r >= 0:
    scores = np.zeros_like(r)
    
    # Mask for feasible bins
    feasible = r >= 0
    
    # Calculate for feasible bins
    r_feasible = r[feasible]
    
    # Weibull survival function S(r) = exp(-(r/lambda)^k)
    s = np.exp(-np.power(r_feasible / lambda_w, k_w))
    
    # Score = - (r * S(r) + epsilon * r)
    scores[feasible] = -(r_feasible * s + epsilon * r_feasible)
    
    # Infeasible bins get very low score
    scores[~feasible] = -1e9
    
    return scores
