import numpy as np
import time

def heuristics(prizes, weights):
    # 1. Input processing
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Ensure shapes are (5, 100)
    if prizes.ndim == 1:
        # Fallback if 1D, though contract says (5, 100)
        # If 1D, assume prizes are the total prize and weights are (5, 100)
        # But contract says prizes is (5, 100). 
        # Let's handle the case where prizes might be passed as (100,) by mistake?
        # No, "prizes and weights, of shape (5, 100)".
        # If prizes is 1D, it's invalid per contract, but let's try to be robust.
        # If prizes is 1D, we can't sum axis 0. We assume it's the total prize.
        P = prizes
        w = weights
    else:
        P = prizes.sum(axis=0) # Total prize per item (100,)
        w = weights             # (5, 100)
        
    n = P.shape[0]
    m = w.shape[0]
    capacity = 25.0
    
    # 2. Time Guard
    t_start = time.time()
    deadline = t_start + 280.0  # 20s safety margin
    
    def check_time():
        return time.time() > deadline
        
    # 3. Lagrangian Relaxation / Dual Variable Estimation
    # We want to estimate lambda (dual prices) such that the solution to the
    # relaxed problem (max sum(P_i - lambda^T w_i) subject to nothing)
    # approximates the true knapsack solution.
    # We use subgradient descent on the dual function.
    # Dual function D(lambda) = max_{x in {0,1}^n} sum (P_i - lambda^T w_i) x_i
    # Since x_i are independent in the relaxation:
    # x_i = 1 if P_i - lambda^T w_i > 0 else 0
    # The subgradient of D(lambda) is (sum x_i w_i - capacity).
    # We want sum x_i w_i ~= capacity.
    
    # Initialize lambda
    # A good initial guess is uniform or based on average density.
    # Let's start with small positive values.
    lambda_vec = np.ones(m) * 0.1
    
    # Subgradient descent parameters
    # Step size alpha_k = c / k or c / sqrt(k)
    # Let's run a fixed number of iterations or until time limit.
    
    # Pre-compute for speed
    # We need to compute x for a given lambda.
    # reduced_cost = P - w.T @ lambda
    # x = reduced_cost > 0
    
    # To make the heuristic more robust, we might want to smooth the selection.
    # But for dual estimation, hard selection is standard.
    
    # Let's run 1000 iterations of subgradient descent.
    # Each iteration is O(n*m) = 500 ops. 1000 * 500 = 500k ops. Very fast.
    
    # We can vectorize the subgradient update if we do it in batches?
    # No, lambda changes sequentially.
    
    # Optimization:
    # Instead of simple subgradient, we can use a more aggressive step size or
    # multiple restarts.
    
    # Let's try a few different step size schedules and pick the lambda that
    # yields the best "proxy" score.
    # Proxy score: The total prize of the items selected by the relaxed solution,
    # penalized by the constraint violation?
    # Actually, the best indicator is how close the resource usage is to capacity
    # AND the high total prize.
    
    # However, the goal of the heuristic is to guide ACO.
    # ACO uses h_i.
    # If h_i = P_i - lambda^T w_i, then ACO prefers items with high reduced cost.
    # This is exactly what we want.
    
    # Let's refine lambda.
    
    best_lambda = lambda_vec.copy()
    best_proxy_score = -np.inf
    
    # Try different step sizes
    step_sizes = [0.1, 0.5, 1.0, 2.0]
    
    for step in step_sizes:
        if check_time():
            break
            
        lam = np.ones(m) * 0.1
        # Decay step size
        # alpha_k = step / (k + 1)
        
        for k in range(2000):
            if check_time():
                break
                
            # Compute reduced costs
            rc = P - w.T @ lam
            
            # Determine selection
            x = (rc > 0).astype(float)
            
            # Compute resource usage
            usage = w.T @ x  # (m,)
            
            # Subgradient
            sg = usage - capacity
            
            # Update lambda
            lam = lam + step / (k + 1.0) * sg
            lam = np.maximum(lam, 0.0) # Dual variables must be non-negative
            
            # Evaluate proxy score
            # We want a lambda that results in usage close to capacity and high prize.
            # Proxy: Total Prize - penalty for violation
            total_prize = np.dot(P, x)
            # Penalty for exceeding capacity
            violation = np.maximum(usage - capacity, 0).sum()
            proxy = total_prize - 10.0 * violation
            
            if proxy > best_proxy_score:
                best_proxy_score = proxy
                best_lambda = lam.copy()
                
    # 4. Generate Heuristic from Best Lambda
    # h_i = P_i - best_lambda^T w_i
    # We must ensure h_i > 0 for the ACO probability calculation (or it becomes 1e-9).
    # The problem statement says: "Heuristic entries that are not positive become 1e-9".
    # If many items have negative reduced cost, they will be ignored.
    # This is good! It focuses the ACO on items that are "profitable" given the resource prices.
    
    h = P - w.T @ best_lambda
    
    # 5. Validation and Sanitization
    # Ensure finite and handle non-positives
    # The contract says non-positive becomes 1e-9. We can do this ourselves to be safe,
    # or let the ACO do it. Let's do it to ensure the output is clean.
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    
    # Normalize? 
    # The ACO uses pheromone * heuristic.
    # If h is very large or small, it might affect the dynamics.
    # But the relative ordering is what matters most for the initial phase.
    # Let's keep the scale as is, or normalize to [0, 1] or similar?
    # The seed program didn't strictly normalize, just clipped.
    # Let's clip to avoid extreme values.
    h = np.clip(h, 1e-9, 1e6)
    
    return h
