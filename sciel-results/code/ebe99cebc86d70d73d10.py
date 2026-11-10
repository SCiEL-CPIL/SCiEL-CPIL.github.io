"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _greedy_cost(params, distance_matrix, demands, capacity):
    """
    Constructs a route using a greedy strategy guided by the heuristic defined by params.
    Returns the total distance of the constructed route.
    """
    alpha, beta, gamma = params
    
    # Clamp parameters to reasonable ranges to avoid numerical issues
    alpha = np.clip(alpha, 0.1, 5.0)
    beta = np.clip(beta, 0.0, 5.0)
    gamma = np.clip(gamma, 0.1, 5.0)
    
    n = len(demands)
    d_safe = distance_matrix + 1e-9
    
    # Calculate heuristic matrix
    # H[i,j] = 1 / (d[i,j]^alpha * (1 + beta * (demand[j]/capacity)^gamma))
    demand_ratio = (demands / capacity) ** gamma
    # Broadcast demand penalty to columns
    penalty = 1.0 + beta * demand_ratio[None, :]
    dist_term = d_safe ** alpha
    h = 1.0 / (dist_term * penalty)
    
    # Ensure positive
    h = np.maximum(h, 1e-9)
    
    # Greedy construction
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    current = 0
    remaining_cap = capacity
    total_dist = 0.0
    
    for _ in range(n - 1):
        # Get scores for unvisited, feasible nodes
        scores = h[current].copy()
        feasible_mask = (~visited) & (demands <= remaining_cap)
        scores[~feasible_mask] = -np.inf
        scores[0] = -np.inf
        
        if not np.any(np.isfinite(scores)):
            # Return to depot
            total_dist += distance_matrix[current, 0]
            current = 0
            remaining_cap = capacity
            # Check if all visited
            if np.all(visited):
                break
            continue
        
        next_node = np.argmax(scores)
        total_dist += distance_matrix[current, next_node]
        remaining_cap -= demands[next_node]
        visited[next_node] = True
        current = next_node
        
    total_dist += distance_matrix[current, 0]
    return total_dist

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Returns an edge-attractiveness matrix.
    Strategy:
    1. Define a parametric heuristic family with 3 parameters: alpha (dist exponent), beta (demand penalty weight), gamma (demand penalty exponent).
    2. Use Nelder-Mead optimization to find parameters that minimize the cost of a greedy route constructed using that heuristic.
    3. Return the heuristic matrix for the best parameters found.
    """
    n = len(demands)
    d_safe = distance_matrix + 1e-9
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    # Initial guess: standard 1/d heuristic corresponds to alpha=1, beta=0, gamma=1
    x0 = np.array([1.0, 0.0, 1.0])
    
    # Bounds for Nelder-Mead are not directly supported in the same way as L-BFGS-B,
    # but we can handle it by clipping inside the objective function.
    
    best_cost = np.inf
    best_params = x0
    
    try:
        # Run optimization
        # maxiter and maxfev are set high, but we will break manually if time runs out
        # We use a wrapper to check time
        result = minimize(
            _greedy_cost, 
            x0, 
            args=(distance_matrix, demands, capacity),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'maxfev': 1000, 'xatol': 1e-4, 'fatol': 1e-6}
        )
        
        if result.success or result.fun < best_cost:
            best_cost = result.fun
            best_params = result.x
            
    except Exception as e:
        # Fallback to initial guess if optimization fails
        pass
        
    # Check time
    if time.time() > deadline:
        # Return heuristic based on best_params so far
        pass

    # Final safety check: if optimization didn't run or failed, use x0
    # We can also try a few random perturbations if time permits
    # But Nelder-Mead should be sufficient.
    
    # Construct the final heuristic
    alpha, beta, gamma = best_params
    alpha = np.clip(alpha, 0.1, 5.0)
    beta = np.clip(beta, 0.0, 5.0)
    gamma = np.clip(gamma, 0.1, 5.0)
    
    demand_ratio = (demands / capacity) ** gamma
    penalty = 1.0 + beta * demand_ratio[None, :]
    dist_term = d_safe ** alpha
    h = 1.0 / (dist_term * penalty)
    
    h = np.maximum(h, 1e-9)
    np.fill_diagonal(h, 1e-9)
    
    # Validate
    if not np.all(np.isfinite(h)):
        return 1.0 / d_safe
    if h.shape != distance_matrix.shape:
        return 1.0 / d_safe
        
    return h
# EVOLVE-BLOCK-END
