<<<<<<< SEARCH
import numpy as np

def heuristics(prizes, weights):
    eps = 1e-9
    ratios = prizes / (weights + eps)  # shape (5, 100)
    # Geometric mean across constraints
    log_ratios = np.log(ratios)
    geo_mean = np.exp(np.mean(log_ratios, axis=0))  # shape (100,)
    return geo_mean
=======
import numpy as np
import time

def heuristics(prizes, weights):
    eps = 1e-9
    start_time = time.time()
    deadline = start_time + 250  # safety margin for 330s wall-clock limit
    
    # Shape: (5, 100)
    n_constraints, n_items = prizes.shape
    capacity = 25.0
    
    # Lagrangian relaxation approach
    # Dual variables (lambda) for each constraint, initialized to 0
    lam = np.zeros(n_constraints)
    
    # Gradient descent on the Lagrangian dual
    # The Lagrangian dual function: g(lam) = max_{x in {0,1}^n, weights*x <= cap} (prizes - lam*weights) * x
    # For MCKP, the inner problem is separable: pick item i if reduced cost > 0
    # Reduced cost for item i: sum_j prizes[j,i] - sum_j lam[j] * weights[j,i]
    
    # We want to find lam such that the total weight of selected items approaches capacity
    # Use subgradient method: lam_new = max(0, lam + alpha * (selected_weight - capacity))
    
    # Initialize with a reasonable step size
    alpha = 1.0
    best_heuristic = None
    
    for iteration in range(1000):
        if time.time() > deadline:
            break
        
        # Compute reduced costs: shape (5, 100)
        reduced_costs = prizes - np.outer(lam, weights)  # lam[j] * weights[j,i]
        
        # For each item, total reduced cost = sum of reduced costs across constraints
        total_reduced_cost = np.sum(reduced_costs, axis=0)  # shape (100,)
        
        # Select items with positive reduced cost (greedy for Lagrangian inner problem)
        selected = total_reduced_cost > 0
        
        # Check if selected items fit within capacity
        # If they don't fit, we need to adjust
        selected_weights = weights[:, selected]  # shape (5, k)
        if selected_weights.shape[1] > 0:
            total_selected_weight = np.sum(selected_weights, axis=1)  # shape (5,)
        else:
            total_selected_weight = np.zeros(n_constraints)
        
        # Subgradient: total_selected_weight - capacity
        subgrad = total_selected_weight - capacity
        
        # Update lambda
        lam = np.maximum(0, lam + alpha * subgrad)
        
        # Decay step size
        alpha *= 0.98
        
        # Compute heuristic: items with higher reduced cost are more attractive
        # Use the total reduced cost as the heuristic score
        heuristic = total_reduced_cost.copy()
        
        # Ensure positive values (replace non-positive with small epsilon)
        heuristic[heuristic <= 0] = eps
        
        if best_heuristic is None:
            best_heuristic = heuristic
        else:
            # Keep the best heuristic based on some criterion
            # For now, just keep updating
            best_heuristic = heuristic
    
    # Final heuristic: use the last computed one
    if best_heuristic is not None:
        return best_heuristic
    else:
        # Fallback: geometric mean
        ratios = prizes / (weights + eps)
        log_ratios = np.log(ratios)
        geo_mean = np.exp(np.mean(log_ratios, axis=0))
        return geo_mean
>>>>>>> REPLACE
BET: {mechanism: "Lagrangian dual prices estimate constraint tightness, scoring items by marginal value net of resource cost, which better guides ants than static ratios", q: 0.65, delta: [0.01, 0.09]}
