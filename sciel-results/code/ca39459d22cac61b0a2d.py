import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    start_time = time.time()
    deadline = start_time + 320.0
    
    D = np.asarray(distance_matrix, dtype=np.float64)
    n = D.shape[0]
    demands = np.asarray(demands, dtype=np.float64)
    
    # Ensure D is safe for inversion
    D_safe = np.where(D > 0, D, 1.0)
    
    # Strategy: Demand-aware heuristic
    # eta[i,j] = (1 / d(i,j)^2) * (1 / (1 + demand[j]))
    # This biases ants towards:
    # 1. Close neighbors (inverse square distance)
    # 2. Low demand nodes? Wait, the inspiration says "penalty for high-demand nodes"
    #    but the goal is to "pack heavier customers early to minimize trips".
    #    If we penalize high demand, ants avoid them.
    #    If we want to pack them, we should REWARD them?
    #    Let's re-read the inspiration: "demand-aware, multi-criteria matrix that combines 
    #    inverse distance with a penalty for high-demand nodes (e.g., eta[i,j] = d(i,j)^-2 * (1 / (1 + demand[j]))), 
    #    directly biasing the pheromone update toward routes that pack heavier customers early"
    #    
    #    Actually, if eta is attractiveness, a HIGHER eta is better.
    #    If we use 1/(1+demand), high demand gets LOWER attractiveness.
    #    This would make ants AVOID high demand customers.
    #    Why would that help?
    #    Maybe it's to prevent getting stuck with a high demand customer that doesn't fit?
    #    Or maybe the inspiration text was slightly misphrased and meant "reward"?
    #    Let's look at the measured score: 1.351. This is BETTER than 1.4524.
    #    So the formula `d^-2 * 1/(1+demand)` worked well.
    #    Wait, if I avoid high demand, I might visit many small customers, filling the truck 
    #    without taking a single large one, which might be inefficient if large ones are far.
    #    However, the prompt says "biasing ... toward routes that pack heavier customers early".
    #    This implies the heuristic should INCREASE attractiveness for high demand?
    #    If the formula in the inspiration was `1/(1+demand)`, that DECREASES attractiveness for high demand.
    #    Contradiction in the inspiration description vs formula?
    #    Let's trust the formula that got 1.351.
    #    
    #    Let's try a hybrid: 
    #    High gamma distance focus + demand factor.
    #    
    #    Let's stick to the inspiration's formula which measured 1.351.
    #    eta[i,j] = D[i,j]^-2 * (1 / (1 + demands[j]))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        dist_factor = D_safe ** -2
        demand_factor = 1.0 / (1.0 + demands)
        
        # Broadcast demand factor to matrix shape (n, n)
        # demand_factor is shape (n,), we need (n, n) where column j is scaled by demand[j]
        eta = dist_factor * demand_factor[np.newaxis, :]
        
    # Clean up non-finite or non-positive values
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    return eta
