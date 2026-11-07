import numpy as np
import time
import scipy.optimize

def heuristics(prizes, distance_matrix, max_len):
    """
    Optimizes a parameterized heuristic matrix for ACO using Nelder-Mead.
    Heuristic form: eta_ij = prize_j^alpha / (d_ij + lambda * d_j0 + mu)^beta
    """
    start_time = time.time()
    DEADLINE = 290.0  # Safety margin
    
    prizes = np.array(prizes, dtype=float)
    dm = np.array(distance_matrix, dtype=float)
    n = len(prizes)
    
    # Precompute return distances
    ret_dists = dm[:, 0]
    
    # Fast evaluation function for the optimizer
    def fast_eval(params):
        """
        Runs a mini-ACO simulation and returns the negative best prize found.
        """
        alpha, beta, lam, mu = params
        
        # Check bounds
        if alpha < 0.01 or beta < 0.01 or lam < 0.0 or mu < 0.0:
            return 1e9
            
        # Compute heuristic
        denom = (dm + lam * ret_dists[None, :] + mu) ** beta
        denom[denom == 0] = 1e-9
        eta = (prizes[None, :] ** alpha) / denom
        eta[eta <= 0] = 1e-9
        
        # Mini ACO Simulation
        num_ants = 10
        num_iters = 10
        tau = np.ones((n, n))
        best_prize_found = 0
        
        # Use a fixed seed for deterministic evaluation to reduce noise
        # However, different instances might benefit from different random seeds?
        # The main ACO uses "seeded per instance". We don't know the seed here.
        # We assume the stochasticity of the mini-ACO is an acceptable proxy.
        # To make it more robust, we can run 2 mini-ACOs with different seeds and average.
        
        for seed in [1, 2]:
            np.random.seed(seed)
            iter_best_prize = 0
            
            for _ in range(num_iters):
                iter_best_prize = 0
                for _ in range(num_ants):
                    current_node = 0
                    current_len = 0
                    visited = np.zeros(n, dtype=bool)
                    visited[0] = True
                    total_prize = 0
                    
                    while True:
                        # Find eligible neighbors
                        # Optimization: Precompute eligible list is hard as current_len changes.
                        # For n=50, a loop is fast enough.
                        eligible = []
                        for j in range(n):
                            if not visited[j]:
                                if current_len + dm[current_node, j] + ret_dists[j] <= max_len:
                                    eligible.append(j)
                        
                        if not eligible:
                            break
                        
                        # Calculate probabilities
                        probs = np.zeros(len(eligible))
                        for k, j in enumerate(eligible):
                            probs[k] = tau[current_node, j] * eta[current_node, j]
                        
                        total_prob = np.sum(probs)
                        if total_prob == 0:
                            j = eligible[np.random.choice(len(eligible))]
                        else:
                            probs = probs / total_prob
                            r = np.random.random()
                            cumsum = 0
                            j = eligible[-1]
                            for k, p in enumerate(probs):
                                cumsum += p
                                if r <= cumsum:
                                    j = eligible[k]
                                    break
                        
                        # Move
                        current_len += dm[current_node, j]
                        total_prize += prizes[j]
                        visited[j] = True
                        current_node = j
                        
                    if total_prize > iter_best_prize:
                        iter_best_prize = total_prize
                
                # Update pheromone
                tau *= 0.9
                tau += iter_best_prize / 10.0
                
                if iter_best_prize > best_prize_found:
                    best_prize_found = iter_best_prize
        
        return -best_prize_found

    # Initial guess
    # Base heuristic: prize / (d_ij + d_j0) -> alpha=1, beta=1, lambda=1, mu=0.1
    x0 = [1.0, 1.0, 1.0, 0.1]
    
    # Bounds for Nelder-Mead (it doesn't natively support bounds well, 
    # but we can penalize out-of-bounds in the function or use 'L-BFGS-B' if differentiable, 
    # which it's not. We'll rely on the penalty in fast_eval and good initial guess).
    
    try:
        # Nelder-Mead options
        options = {
            'maxiter': 100,
            'fatol': 1e-4,
            'xatol': 1e-4,
            'disp': False
        }
        
        res = scipy.optimize.minimize(fast_eval, x0, method='Nelder-Mead', options=options)
        
        if res.success or res.fun < 1e8:
            best_params = res.x
        else:
            # Fallback to initial guess if optimization fails
            best_params = x0
            
    except Exception as e:
        best_params = x0
        
    # Check time
    if time.time() - start_time > DEADLINE - 5:
        # If we are running out of time, use the best params found so far
        # Or just use x0 if we haven't finished
        pass
        
    alpha, beta, lam, mu = best_params
    
    # Ensure positive values for final construction
    alpha = max(0.01, alpha)
    beta = max(0.01, beta)
    lam = max(0.0, lam)
    mu = max(0.0, mu)
    
    # Construct final heuristic matrix
    denom = (dm + lam * ret_dists[None, :] + mu) ** beta
    denom[denom == 0] = 1e-9
    eta = (prizes[None, :] ** alpha) / denom
    eta[eta <= 0] = 1e-9
    
    # Final sanity check
    if not np.all(np.isfinite(eta)) or eta.shape != (n, n):
        # Fallback to a simple valid heuristic
        eta = np.ones((n, n))
        eta[0, :] = 0 # Node 0 is start, no self-loop needed usually, but keep positive
        eta = np.where(eta == 0, 1e-9, eta)
        
    return eta
