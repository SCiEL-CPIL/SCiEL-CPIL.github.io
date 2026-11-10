"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _greedy_cost(distance_matrix, demands, capacity, heuristic):
    """
    Fast greedy evaluation for initial screening.
    """
    n = len(demands)
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    current = 0
    remaining_cap = capacity
    total_dist = 0.0
    
    for _ in range(n - 1):
        scores = heuristic[current].copy()
        feasible_mask = (~visited) & (demands <= remaining_cap)
        scores[~feasible_mask] = -np.inf
        scores[0] = -np.inf
        
        if not np.any(np.isfinite(scores)):
            total_dist += distance_matrix[current, 0]
            current = 0
            remaining_cap = capacity
            continue
        
        best_j = np.argmax(scores)
        total_dist += distance_matrix[current, best_j]
        remaining_cap -= demands[best_j]
        visited[best_j] = True
        current = best_j
    
    total_dist += distance_matrix[current, 0]
    return total_dist

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Returns an edge-attractiveness matrix.
    Strategy: 
    1. Define a parametric family combining distance, demand, and Clarke-Wright savings.
    2. Use Nelder-Mead optimization to find the best parameters.
    3. Return the optimized heuristic matrix.
    """
    n = len(demands)
    d_safe = distance_matrix + 1e-9
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin

    # Precompute Clarke-Wright savings
    # savings[i, j] = dist[0, i] + dist[0, j] - dist[i, j]
    savings = distance_matrix[0, :][:, None] + distance_matrix[0, :][None, :] - distance_matrix
    # Normalize savings to be positive and bounded
    savings = np.maximum(savings, 0.0)
    max_savings = np.max(savings)
    if max_savings > 0:
        savings = savings / max_savings
    
    # Demand efficiency: 1 / (1 + demand)
    demand_eff = 1.0 / (1.0 + demands)
    
    # Depot pressure: nodes with higher demand are "heavier"
    # We want to visit them early when capacity is full.
    # A simple proxy is 1/demand, but let's use a smoother function.
    depot_pressure = 1.0 / (1.0 + demands)

    def make_heuristic(params):
        alpha, beta, gamma, delta = params
        # Clamp parameters to avoid numerical issues
        alpha = np.clip(alpha, 0.1, 5.0)
        beta = np.clip(beta, 0.0, 5.0)
        gamma = np.clip(gamma, 0.0, 5.0)
        delta = np.clip(delta, 0.0, 5.0)
        
        # Base distance component
        base = 1.0 / (d_safe ** alpha)
        
        # Demand penalty component
        demand_term = 1.0 / (1.0 + beta * demands[None, :])
        
        # Savings boost component
        savings_term = 1.0 + gamma * savings
        
        # Depot pressure component (applied to destination node)
        pressure_term = 1.0 + delta * depot_pressure[None, :]
        
        h = base * demand_term * savings_term * pressure_term
        h = np.maximum(h, 1e-9)
        np.fill_diagonal(h, 1e-9)
        return h

    def objective(params):
        h = make_heuristic(params)
        return _greedy_cost(distance_matrix, demands, capacity, h)

    # Initial guess
    x0 = np.array([1.5, 0.5, 1.0, 0.5])
    
    # Nelder-Mead optimization
    try:
        from scipy.optimize import minimize
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4})
        best_params = res.x
    except Exception:
        best_params = x0
    
    # Check time
    if time.time() > deadline - 10.0:
        best_params = x0
        
    # Final heuristic
    best_heuristic = make_heuristic(best_params)
    
    # Final Safety Checks
    if not np.all(np.isfinite(best_heuristic)):
        best_heuristic = 1.0 / d_safe
    if np.any(best_heuristic <= 0):
        best_heuristic = np.clip(best_heuristic, 1e-9, None)
    if best_heuristic.shape != distance_matrix.shape:
        best_heuristic = 1.0 / d_safe
        
    return best_heuristic
# EVOLVE-BLOCK-END
