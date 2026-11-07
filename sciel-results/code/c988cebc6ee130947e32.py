import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes a heuristic matrix for ACO.
    Strategy: Direct optimization of the heuristic matrix entries using Nelder-Mead.
    We treat the heuristic as a rank-1 perturbation of the inverse distance matrix:
    eta_ij = (1/dist_ij) * (1 + w_i * v_j + w_j * v_i)
    This allows the heuristic to learn customer-specific "attractiveness" or "repulsiveness"
    in a low-dimensional parameter space (2*n parameters) rather than a full n^2 matrix.
    We use the exact ACO simulation as the fitness function.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin

    n = len(demands)
    demands = np.array(demands, dtype=float)
    dist = distance_matrix.copy()
    
    # Ensure diagonal is large so it's never chosen
    np.fill_diagonal(dist, 1e9)
    
    # Base heuristic: inverse distance
    base_eta = 1.0 / dist
    base_eta[~np.isfinite(base_eta)] = 1e-9
    base_eta[base_eta < 1e-9] = 1e-9

    # Parameters to optimize: two vectors of length n
    # eta_ij = base_eta_ij * (1 + w_i * v_j + w_j * v_i)
    # We initialize with zeros (which gives eta = base_eta)
    
    def simulate_aco(eta_matrix, n_ants=20, n_iters=30, seed=42):
        """
        Fast simulation of the exact ACO process described in the task.
        Returns the best route cost found.
        """
        rng = np.random.default_rng(seed)
        tau = np.ones((n, n))
        best_route_cost = float('inf')
        
        for it in range(n_iters):
            iteration_best_cost = float('inf')
            
            for ant in range(n_ants):
                current = 0
                load = 0.0
                visited = np.zeros(n, dtype=bool)
                visited[0] = True
                total_dist = 0.0
                
                while not np.all(visited):
                    # Find feasible customers
                    # Vectorized check for feasible
                    feasible_mask = (~visited) & (load + demands <= capacity)
                    feasible_mask[0] = False # Depot is visited
                    feasible_indices = np.where(feasible_mask)[0]
                    
                    if len(feasible_indices) == 0:
                        total_dist += distance_matrix[current, 0]
                        current = 0
                        load = 0.0
                        continue
                    
                    # Calculate probabilities
                    # Note: tau is (n,n), eta is (n,n)
                    # We only need the row 'current' for feasible indices
                    probs = tau[current, feasible_indices] * eta_matrix[current, feasible_indices]
                    sum_probs = np.sum(probs)
                    
                    if sum_probs == 0 or not np.isfinite(sum_probs):
                        idx = rng.integers(0, len(feasible_indices))
                    else:
                        probs = probs / sum_probs
                        # Handle NaN/Inf in probs
                        if not np.all(np.isfinite(probs)):
                            probs = np.where(np.isfinite(probs), probs, 0)
                            sum_p = np.sum(probs)
                            if sum_p == 0:
                                idx = rng.integers(0, len(feasible_indices))
                            else:
                                probs = probs / sum_p
                                idx = rng.choice(len(feasible_indices), p=probs)
                        else:
                            idx = rng.choice(len(feasible_indices), p=probs)
                    
                    next_node = feasible_indices[idx]
                    total_dist += distance_matrix[current, next_node]
                    load += demands[next_node]
                    visited[next_node] = True
                    current = next_node
                
                if current != 0:
                    total_dist += distance_matrix[current, 0]
                
                if total_dist < iteration_best_cost:
                    iteration_best_cost = total_dist
                if total_dist < best_route_cost:
                    best_route_cost = total_dist
            
            tau = 0.9 * tau + 1.0 / iteration_best_cost
            
            # Check time
            if time.time() > deadline - 5.0:
                return best_route_cost

        return best_route_cost

    def evaluate_params(params):
        """
        Takes a flat array of parameters [w_0..w_n-1, v_0..v_n-1]
        Returns the cost of the ACO simulation with the resulting heuristic.
        """
        w = params[:n]
        v = params[n:]
        
        # Construct perturbation factor
        # factor_ij = 1 + w_i * v_j + w_j * v_i
        # We can compute this efficiently
        # term1 = w_i * v_j -> outer product w, v
        # term2 = w_j * v_i -> outer product v, w (which is term1.T)
        # So factor = 1 + outer(w, v) + outer(v, w)
        
        factor = 1.0 + np.outer(w, v) + np.outer(v, w)
        
        # Ensure factor is positive to keep eta positive
        # If factor becomes negative or zero, eta becomes non-positive, which is clamped to 1e-9.
        # To avoid dead zones, we can clamp factor to a small positive value, e.g., 0.1
        # Or simply let the clamp in the final eta handle it.
        # However, if factor is negative, eta is negative -> 1e-9. This is a valid state.
        
        eta = base_eta * factor
        
        # Clamp
        eta[~np.isfinite(eta)] = 1e-9
        eta[eta < 1e-9] = 1e-9
        
        # Symmetry check: factor_ij = 1 + w_i v_j + w_j v_i. factor_ji = 1 + w_j v_i + w_i v_j. Same.
        # So eta is symmetric.
        
        return simulate_aco(eta)

    # Initialize parameters
    # Start with zeros
    x0 = np.zeros(2 * n)
    
    # Initial best
    best_cost = evaluate_params(x0)
    best_params = x0.copy()
    
    # Nelder-Mead optimization
    # We need to be careful with the number of iterations and the simplex size.
    # Standard scipy.optimize.minimize with method='Nelder-Mead'
    # But we can implement a simple version or use scipy if available.
    # The prompt says SciPy is available.
    
    from scipy.optimize import minimize
    
    # We want to minimize the cost.
    # However, simulate_aco has randomness (seeded, but still).
    # The function is deterministic for a fixed seed.
    
    # Set bounds? w and v can be any real number.
    # If they get too large, factor can be huge, making eta huge.
    # If they get too small (negative large), factor can be negative, making eta 1e-9.
    # This is fine.
    
    # We will run a few iterations of Nelder-Mead.
    # To save time, we can limit the number of function evaluations.
    # Each function evaluation is a full ACO simulation (30 iters * 20 ants).
    # This is relatively fast (vectorized).
    
    # Let's estimate time per eval.
    # 50 nodes, 20 ants, 30 iters.
    # Inner loop: 50 steps.
    # Vectorized operations on arrays of size ~50.
    # Should be < 1ms per ant? Maybe 10-20ms per eval.
    # 280s / 20ms = 14000 evals. Plenty.
    
    # Let's try to run Nelder-Mead for a fixed number of iterations or until time limit.
    
    # We need a callback to stop if time is up.
    stopped = [False]
    
    def callback(xk):
        if time.time() > deadline - 10.0:
            stopped[0] = True
            return True
        return False

    # Initial simplex
    # Default Nelder-Mead creates a simplex based on initial step size.
    # We can specify initial_simplex if we want.
    # Let's let scipy handle it but set maxiter.
    
    try:
        res = minimize(
            evaluate_params, 
            x0, 
            method='Nelder-Mead', 
            options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True},
            callback=callback
        )
        
        # Check if result is better
        if res.fun < best_cost:
            best_cost = res.fun
            best_params = res.x
            
    except Exception as e:
        # Fallback to x0 if optimization fails
        pass

    # Final construction
    w = best_params[:n]
    v = best_params[n:]
    
    factor = 1.0 + np.outer(w, v) + np.outer(v, w)
    result = base_eta * factor
    
    # Final safety clamps
    result[~np.isfinite(result)] = 1e-9
    result[result < 1e-9] = 1e-9
    
    # Ensure shape
    if result.shape != distance_matrix.shape:
        result = np.resize(result, distance_matrix.shape)
        
    return result
