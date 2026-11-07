"""Combined optimizer: Sobol global scan + CMA-ES local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

class Optimizer:
    """
    Optimizer for BBOB f7 (Step Ellipsoidal).
    
    Strategy:
    1. Global Sobol scan to locate the basin/step.
    2. CMA-ES local refinement from the best point.
    3. Final local random perturbation to handle step boundaries.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 280.0  # Safety margin before 330s kill
        d = self.dim
        budget = self.budget
        
        calls_made = 0
        best_x = np.zeros(d)
        best_f = np.inf
        
        def time_ok():
            return time.time() - start_time < timeout

        def safe_func(x):
            nonlocal calls_made
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            return fx

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Global Sobol Scan
        # Use a fraction of budget for global exploration
        # Sobol sequences provide better low-discrepancy properties than Halton
        scan_budget = int(budget * 0.3)  # 30% for scanning
        if scan_budget < 50:
            scan_budget = 50
        if scan_budget > budget - 50:
            scan_budget = budget - 50
            
        # Generate Sobol points
        # scipy.stats.qmc.Sobol is available in recent versions
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d, scramble=True)
            # Next power of 2 >= scan_budget for better quality, but capped
            # Actually, just use scan_budget points
            sobol_pts = sampler.random(scan_budget)
            # Map from [0,1)^d to [-5,5]^d
            sobol_pts = sobol_pts * 10.0 - 5.0
        except ImportError:
            # Fallback to Halton if Sobol not available
            def halton_sequence(n, dim):
                points = np.zeros((n, dim))
                primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
                for i in range(1, n + 1):
                    for d_idx in range(dim):
                        prime = primes[d_idx]
                        f = 1.0
                        r = 0.0
                        i_val = i
                        while i_val > 0:
                            f /= prime
                            r += f * (i_val % prime)
                            i_val //= prime
                        points[i - 1, d_idx] = r
                return points
            sobol_pts = halton_sequence(scan_budget, d)
            sobol_pts = sobol_pts * 10.0 - 5.0

        # Evaluate Sobol points
        for pt in sobol_pts:
            if not time_ok() or calls_made >= budget:
                break
            f_val = safe_func(pt)
            update_best(pt, f_val)

        # Phase 2: CMA-ES Refinement
        # Use remaining budget for local search
        if time_ok() and calls_made < budget and best_x is not None:
            # CMA-ES Parameters
            lam = 10  # Population size
            mu = lam // 2
            
            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / np.sum(w)
            mueff = 1.0 / np.sum(w**2)
            
            # Learning rates
            cc = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
            cs = (mueff + 2) / (d + mueff + 5)
            c1 = 2 / ((d + 1.3)**2 + mueff)
            cmu = (2 * (mueff - 2 + 1/mueff)) / ((d + 2)**2 + mueff)
            
            # Initial sigma
            sigma = 1.0
            x_mean = best_x.copy()
            
            # Evolution paths
            p_sigma = np.zeros(d)
            p_c = np.zeros(d)
            
            # Covariance matrix
            C = np.eye(d)
            # Cholesky decomposition for sampling
            try:
                C_sqrt = np.linalg.cholesky(C)
            except:
                C_sqrt = np.eye(d)
                
            # Max iterations for CMA-ES
            max_iter = (budget - calls_made) // lam
            if max_iter < 10:
                max_iter = 10
            if max_iter > 200:
                max_iter = 200

            for it in range(max_iter):
                if not time_ok() or calls_made + lam > budget:
                    break
                    
                # Sample population
                pop = np.zeros((lam, d))
                for i in range(lam):
                    z = np.random.standard_normal(d)
                    pop[i] = x_mean + sigma * (C_sqrt @ z)
                
                # Evaluate population
                fvals = np.zeros(lam)
                for i in range(lam):
                    if not time_ok() or calls_made >= budget:
                        fvals[i] = np.inf
                        continue
                    fvals[i] = safe_func(pop[i])
                    update_best(pop[i], fvals[i])
                
                # Sort by fitness
                indices = np.argsort(fvals)
                pop_sorted = pop[indices]
                fvals_sorted = fvals[indices]
                
                # Update mean
                w_sum = np.sum(w)
                x_mean_old = x_mean.copy()
                x_mean = np.sum(w[:, None] * pop_sorted[:mu], axis=0)
                
                # Update step size control
                y_mean = (x_mean - x_mean_old) / sigma
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * y_mean
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs)**(2 * (it + 1))) <
                           (1 + 2 / (d + 1)) * np.sqrt((d + 1 / d) - 1))
                if h_sigma:
                    p_c = (1 - cs) * p_c + np.sqrt(cs * (2 - cs) * mueff) * y_mean
                else:
                    p_c = (1 - cs) * p_c
                    
                sigma *= np.exp(cs / d_sigma * (np.linalg.norm(p_c) / np.sqrt(1 - (1 - cs)**(it + 1)) - 1))
                
                # Update covariance matrix
                v = (x_mean - x_mean_old) / sigma
                C = (1 + c1 * cmu * np.outer(v, v) - c1) * C + cmu * np.sum(w[:, None, None] * np.outer(pop_sorted[:mu] - x_mean_old, pop_sorted[:mu] - x_mean_old) / sigma**2, axis=0)
                
                # Symmetrize
                C = (C + C.T) / 2
                
                # Update Cholesky
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except:
                    # Regularize if not positive definite
                    C += 1e-6 * np.eye(d)
                    C = (C + C.T) / 2
                    try:
                        C_sqrt = np.linalg.cholesky(C)
                    except:
                        break

        # Phase 3: Local Random Perturbation
        # Step functions often have sharp boundaries. CMA-ES might stop just outside the optimal step.
        if time_ok() and calls_made < budget and best_x is not None:
            rng = np.random.default_rng(42)
            # Define a few radii to search
            radii = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
            
            for r in radii:
                if not time_ok() or calls_made >= budget:
                    break
                
                # Generate random points in a sphere of radius r around current_best_x
                n_local = min(20, budget - calls_made)
                
                for _ in range(n_local):
                    if not time_ok() or calls_made >= budget:
                        break
                    
                    # Random direction
                    dir_vec = rng.standard_normal(d)
                    dir_vec /= np.linalg.norm(dir_vec)
                    
                    # Random distance in [0, r]
                    dist = rng.uniform(0, r)
                    
                    x_candidate = best_x + dist * dir_vec
                    x_candidate = np.clip(x_candidate, -5.0, 5.0)
                    
                    fx = safe_func(x_candidate)
                    
                    if fx < best_f:
                        best_f = fx
                        best_x = x_candidate.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
