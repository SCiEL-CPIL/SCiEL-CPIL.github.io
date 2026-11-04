import numpy as np
import scipy.optimize
import time

N = 100

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    def project_to_feasible(h):
        """
        Project h onto the feasible set: 0 <= h_i <= 1, sum(h) = N/2.
        """
        h = np.clip(h, 0.0, 1.0)
        for _ in range(100):
            current_sum = np.sum(h)
            diff = N / 2.0 - current_sum
            if abs(diff) < 1e-12:
                break
            h += diff / N
            h = np.clip(h, 0.0, 1.0)
        
        # Final exact correction using water-filling approach
        residual = N / 2.0 - np.sum(h)
        if abs(residual) > 1e-14:
            # Try to distribute residual among interior points
            for _ in range(50):
                residual = N / 2.0 - np.sum(h)
                if abs(residual) < 1e-14:
                    break
                mask = (h > 1e-10) & (h < 1.0 - 1e-10)
                n_mask = np.sum(mask)
                if n_mask == 0:
                    # All at bounds, need to adjust one element
                    if residual > 0:
                        # Need to increase, find element with smallest value
                        idx = np.argmin(h)
                        h[idx] += residual
                    else:
                        # Need to decrease, find element with largest value
                        idx = np.argmax(h)
                        h[idx] += residual
                else:
                    h[mask] += residual / n_mask
                h = np.clip(h, 0.0, 1.0)
        
        # Ensure exact sum
        residual = N / 2.0 - np.sum(h)
        if abs(residual) > 1e-15:
            # Add residual to the element closest to 0.5
            idx = np.argmin(np.abs(h - 0.5))
            h[idx] += residual
            h = np.clip(h, 0.0, 1.0)
        return h

    def get_correlations(h):
        """
        Compute linear correlations of h and 1-h for all non-zero shifts.
        """
        g = 1.0 - h
        c = np.correlate(h, g, mode='full')
        # c has length 2N-1. Index N-1 is shift 0.
        # We want shifts k = -(N-1) ... -1 and 1 ... N-1
        c_nonzero = np.concatenate([c[:N-1], c[N:]])
        return c_nonzero

    def score_h(h):
        """Calculate the true max overlap score."""
        c_all = get_correlations(h)
        return np.max(c_all)

    def lse_objective(h, beta):
        """
        LogSumExp of correlations for smooth approximation of max.
        """
        c_all = get_correlations(h)
        max_c = np.max(c_all)
        if max_c > 500:
            max_c = 500.0
        lse = max_c + np.log(np.sum(np.exp(beta * (c_all - max_c)))) / beta
        return lse

    def objective_de(h):
        """
        Objective for Differential Evolution.
        """
        h_proj = project_to_feasible(h)
        return score_h(h_proj)

    # Initial best guess: Uniform
    best_h = np.full(N, 0.5)
    best_score = score_h(best_h)
    
    # Run multiple DE runs with different seeds
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    seeds_to_try = [42, 123, 456, 789, 1000]
    
    for seed in seeds_to_try:
        if time.time() - start_time > time_limit - 30:
            break
            
        try:
            de_result = scipy.optimize.differential_evolution(
                objective_de, 
                bounds, 
                maxiter=30, 
                popsize=12,
                tol=1e-5,
                seed=seed,
                polish=False
            )
            h_de = de_result.x
            h_de = project_to_feasible(h_de)
            score_de = score_h(h_de)
            
            if score_de < best_score:
                best_score = score_de
                best_h = h_de.copy()
                
        except Exception:
            pass
    
    # Polish the best result using L-BFGS-B with LSE objective
    if time.time() - start_time < time_limit - 10:
        # Use adaptive beta: start high, decrease if stuck
        for beta in [100.0, 50.0, 20.0]:
            if time.time() - start_time > time_limit - 5:
                break
                
            h0 = best_h.copy()
            
            def obj_polish(h):
                h_proj = project_to_feasible(h)
                return lse_objective(h_proj, beta)

            try:
                res = scipy.optimize.minimize(
                    obj_polish, 
                    h0, 
                    method='L-BFGS-B', 
                    bounds=bounds, 
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
                )
                h_polished = res.x
                h_polished = project_to_feasible(h_polished)
                score_polished = score_h(h_polished)
                
                if score_polished < best_score:
                    best_score = score_polished
                    best_h = h_polished.copy()
                    
            except Exception:
                pass
    
    # Final local search: try small perturbations
    if time.time() - start_time < time_limit - 2:
        rng = np.random.RandomState(42)
        for _ in range(200):
            if time.time() - start_time > time_limit - 1:
                break
            h_perturbed = best_h + rng.normal(0, 0.01, N)
            h_perturbed = project_to_feasible(h_perturbed)
            score_perturbed = score_h(h_perturbed)
            if score_perturbed < best_score:
                best_score = score_perturbed
                best_h = h_perturbed.copy()
    
    return best_h
