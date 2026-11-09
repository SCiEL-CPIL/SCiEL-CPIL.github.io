import numpy as np
import time

def heuristics(prizes, weights):
    """
    Calculate heuristic attractiveness scores for items in a Multiple-Constraint Knapsack Problem.
    Uses Lagrangian relaxation to estimate dual prices and score items by reduced cost.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin

    # 1. Robust Shape Handling
    p = np.asarray(prizes, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)

    # Ensure 2D shape (n_constraints, n_items)
    if p.ndim == 1:
        # If 1D, assume single effective constraint or aggregate.
        # We cannot do 5-constraint Lagrangian. Fallback to simple ratio.
        # Reshape to (1, N) to use vectorized ops if possible, but logic differs.
        # Let's just handle 1D separately as a fallback.
        n_items = p.shape[0]
        total_prizes = p
        # Assume weights is also 1D
        if w.ndim == 1:
             raw_scores = total_prizes / (w + 1e-9)
        else:
             raw_scores = np.ones(n_items)
        raw_scores[~np.isfinite(raw_scores)] = 1e-9
        raw_scores[raw_scores <= 0] = 1e-9
        return raw_scores

    if p.ndim != 2:
        raise ValueError("Prizes must be 1D or 2D")

    # Check orientation: Expected (5, 100). If (100, 5), transpose.
    if p.shape[0] != 5 and p.shape[1] == 5:
        p = p.T
        w = w.T

    n_constraints, n_items = p.shape
    capacity = 25.0
    capacities = np.full(n_constraints, capacity)

    # 2. Lagrangian Relaxation
    # Initialize dual prices lambda
    # Heuristic initialization: average prize/weight ratio per constraint?
    # Or just 1.0
    lambdas = np.ones(n_constraints)
    
    # Pre-compute sums for efficiency?
    # L_i = sum_k p_ik - sum_k lambda_k w_ik
    # This is a matrix multiplication: p.sum(axis=0) - w.T @ lambdas
    
    # Subgradient Method to minimize dual function g(lambda)
    # g(lambda) = sum_k lambda_k C_k + sum_i max(0, L_i)
    # grad g(lambda) = C - sum_{i: L_i > 0} w_i
    
    # Step size schedule
    # Start with a reasonable step, decay over time.
    # Max iterations for this sub-problem: keep it small, e.g., 100-200.
    max_lagrangian_iters = 200
    
    # To avoid taking too long, we check time every few iterations.
    
    for it in range(max_lagrangian_iters):
        if time.time() > deadline:
            break
            
        # Compute L_i for all items
        # L_i = total_prize_i - dot(w_i, lambdas)
        total_prizes = p.sum(axis=0)
        # w is (K, N). w.T is (N, K). lambdas is (K,).
        # w.T @ lambdas gives (N,)
        weighted_weights = w.T @ lambdas
        L = total_prizes - weighted_weights
        
        # Identify items with positive Lagrangian value (selected in relaxed problem)
        mask = L > 0
        
        # Compute subgradient: C - sum of weights of selected items
        # sum of weights of selected items: w * mask (broadcasting)
        # w is (K, N), mask is (N,). w * mask gives (K, N). Sum over N.
        selected_weights_sum = np.sum(w * mask, axis=1)
        
        subgrad = capacities - selected_weights_sum
        
        # Step size
        # Use a simple decaying step size
        # alpha = 1.0 / (it + 1) ? Or normalized by norm of subgrad?
        # Let's use a fixed small step initially, then decay.
        # Or Barzilai-Borwein style? Keep it simple.
        # alpha = 0.1 / (1 + it * 0.01)
        
        # Normalize subgradient to avoid huge steps if weights are small
        norm_subgrad = np.linalg.norm(subgrad)
        if norm_subgrad < 1e-6:
            break # Converged or stuck
            
        # Step size
        alpha = 1.0 / (1.0 + 0.1 * it)
        
        # Update lambdas
        # lambdas = lambdas - alpha * subgrad
        # But we need lambdas >= 0
        new_lambdas = lambdas - alpha * subgrad
        lambdas = np.maximum(0, new_lambdas)
        
        # Optional: Check for convergence
        # if np.all(new_lambdas == lambdas): break

    # Final Scores
    # Score_i = max(0, L_i)
    # We use the final lambdas
    total_prizes = p.sum(axis=0)
    weighted_weights = w.T @ lambdas
    L = total_prizes - weighted_weights
    raw_scores = np.maximum(0, L)
    
    # Ensure positive
    raw_scores[raw_scores <= 0] = 1e-9
    raw_scores[~np.isfinite(raw_scores)] = 1e-9
    
    return raw_scores
