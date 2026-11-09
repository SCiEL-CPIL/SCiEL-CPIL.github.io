import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    # If we have used too much time, fallback to simple metric
    if time.time() - _start_time > 290:
        z = (f_best - mu) / (sigma + 1e-9)
        EI = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        EI = np.maximum(EI, 0)
        return EI / (cost + 1e-9)

    # 1. Calculate Expected Improvement
    z = (f_best - mu) / (sigma + 1e-9)
    EI = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    EI = np.maximum(EI, 0.0)

    # 2. Identify Pareto-optimal candidates
    # We want to maximize EI and minimize cost.
    # Sort by cost ascending.
    sorted_idx = np.argsort(cost)
    
    pareto_indices = []
    max_ei_seen = -np.inf
    
    for i in sorted_idx:
        if EI[i] > max_ei_seen:
            pareto_indices.append(i)
            max_ei_seen = EI[i]
            
    # 3. Calculate Hypervolume Contribution
    scores = np.zeros(len(mu))
    
    if not pareto_indices:
        return scores
        
    # Reference points
    # We are maximizing EI and minimizing Cost.
    # Hypervolume is calculated in the (EI, -Cost) space or similar.
    # Let's use the area in (EI, Cost) space bounded by the front and the reference point (0, MaxCost).
    # Reference: EI_ref = 0, Cost_ref = max(cost)
    
    C_ref = np.max(cost)
    E_ref = 0.0
    
    k = len(pareto_indices)
    
    # pareto_indices are already sorted by cost ascending (due to sorted_idx iteration)
    # This means EI is descending (non-increasing) along the front.
    
    for j in range(k):
        idx = pareto_indices[j]
        
        # Width in EI dimension
        if j < k - 1:
            next_ei = EI[pareto_indices[j+1]]
            width = EI[idx] - next_ei
        else:
            width = EI[idx] - E_ref
            
        # Height in Cost dimension (distance from reference cost)
        # Since we minimize cost, the "height" of the contribution is how much cheaper it is than the reference.
        height = C_ref - cost[idx]
        
        # Contribution
        scores[idx] = width * height
        
    return scores
