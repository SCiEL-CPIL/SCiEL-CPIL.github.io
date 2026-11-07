import numpy as np
import time

class Optimizer:
    """
    Optimizer for BBOB f13 (Sharp Ridge) using a hybrid of 
    Adaptive Population Search (CMA-ES-like) and Nelder-Mead polishing.
    
    The Sharp Ridge function is ill-conditioned. Standard local methods like NM 
    often struggle to find the ridge direction. CMA-ES adapts the covariance 
    matrix, allowing the search to elongate along the ridge and shrink 
    perpendicular to it, which is ideal for this problem.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Safety margin: Evaluator kills at 330s. We target finishing well before.
        deadline = time.time() + 290.0 
        
        best_x = None
        best_f = float('inf')
        calls = 0
        rng = np.random.default_rng(42) # Fixed seed for reproducibility

        def safe_call(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return None
            if time.time() > deadline:
                return None
            calls += 1
            try:
                # Clip to box to ensure validity, though func does it too
                x_clipped = np.clip(x, -5.0, 5.0)
                f_val = func(x_clipped)
                if f_val < best_f:
                    best_f = f_val
                    best_x = x_clipped.copy()
                return f_val
            except Exception:
                return None

        # Phase 1: Global Random Search
        # Use 50 samples to better locate the basin/ridge in 5D
        n_random = 50
        for _ in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, size=dim)
            f_val = safe_call(x)
            if f_val is None:
                break

        if best_x is None:
            # Fallback if all random calls failed
            x0 = np.zeros(dim)
            f0 = safe_call(x0)
            if f0 is None:
                return best_x, best_f
            best_x = x0
            best_f = f0

        # Phase 2: CMA-ES-like Adaptive Search
        # Initialize parameters
        # Start with the best point found so far
        mean = best_x.copy()
        
        # Initial sigma: large enough to explore, small enough to converge
        # The ridge is sharp, so we want to be careful. 
        # 1.0 is a reasonable start for [-5,5] box.
        sigma = 1.0 
        
        # Covariance matrix: Identity initially
        C = np.eye(dim)
        
        # Population size
        lambda_pop = 10
        mu = lambda_pop // 2 # Number of parents
        
        # CMA-ES weights
        # Standard weights based on rank
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        # Variance reduction factor
        c_sigma = (mu + 2) / (dim + mu + 5)
        c_c = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
        
        # Step sizes for mean and covariance
        # These are simplified updates for robustness in low budget
        # Full CMA-ES is complex; we use a simplified adaptive scheme.
        
        # Keep track of history for adaptation
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Main Loop
        # We will run until budget is exhausted or time runs out
        # Each iteration consumes lambda_pop calls.
        
        # To ensure we finish, we check budget before generating population
        while calls + lambda_pop <= budget and time.time() < deadline:
            
            # Generate population
            # Cholesky decomposition for sampling from covariance
            # Handle numerical issues
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset to identity
                C = np.eye(dim)
                L = np.eye(dim)
            
            # Sample
            z = rng.standard_normal((lambda_pop, dim))
            pop = mean + sigma * (z @ L.T)
            
            # Clip population to box
            pop = np.clip(pop, -5.0, 5.0)
            
            # Evaluate population
            f_vals = np.array([safe_call(p) for p in pop])
            
            # Handle None values (should not happen if budget check is correct, but safe)
            # If any None, replace with inf and continue
            f_vals = np.where(f_vals is None, np.inf, f_vals)
            # Note: safe_call returns None, but we need to handle it in array
            # Actually, safe_call returns float or None. 
            # Let's re-evaluate if any were None
            # The list comprehension above might contain None.
            # Let's do it more carefully.
            
            # Re-implementing evaluation to handle None properly in array
            f_vals_list = []
            for p in pop:
                if calls >= budget or time.time() > deadline:
                    break
                val = safe_call(p)
                if val is None:
                    val = np.inf
                f_vals_list.append(val)
            
            if len(f_vals_list) < lambda_pop:
                break
                
            f_vals = np.array(f_vals_list)
            
            # Sort population by fitness
            idx = np.argsort(f_vals)
            pop_sorted = pop[idx]
            f_sorted = f_vals[idx]
            
            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = pop_sorted[0].copy()
            
            # Update mean (Weighted average of top mu)
            parents = pop_sorted[:mu]
            w = weights
            mean_old = mean.copy()
            mean = np.sum(w[:, np.newaxis] * parents, axis=0)
            
            # Update sigma and covariance
            # This is a simplified version. Full CMA-ES uses evolution paths.
            # We will use a simpler adaptive scheme:
            # If we made progress, keep sigma. If not, reduce it.
            # Also, update C to align with the spread of parents.
            
            # Calculate spread of parents
            if mu > 1:
                # Estimate covariance from parents
                # Center parents
                parents_centered = parents - mean
                # Weighted covariance
                C_new = np.zeros((dim, dim))
                for i in range(mu):
                    C_new += w[i] * np.outer(parents_centered[i], parents_centered[i])
                
                # Mix with previous C for stability
                # c1 is adaptation rate for covariance
                c1 = 2 / ((dim + 1.3)**2 + mu)
                C = (1 - c1) * C + c1 * (C_new + (1 - c_sigma) * np.outer(p_sigma, p_sigma))
                
                # Ensure positive definite
                # Add small regularization if needed
                C += 1e-9 * np.eye(dim)
                
                # Update sigma based on ratio of new sigma to old sigma
                # This is a rough heuristic. 
                # A better way: sigma = sigma * exp(c_sigma * (||p_sigma|| / sqrt(2*mu) - 1))
                # But we don't track p_sigma perfectly in this simplified version.
                # Let's use a simple rule: if best improved, sigma stays or grows slightly?
                # No, usually sigma shrinks as we converge.
                
                # Let's stick to a simpler adaptive strategy:
                # Reduce sigma if the spread of the population is small relative to sigma
                spread = np.max(pop_sorted, axis=0) - np.min(pop_sorted, axis=0)
                avg_spread = np.mean(spread)
                if avg_spread < 0.1 * sigma:
                    sigma *= 0.8
                elif avg_spread > 2.0 * sigma:
                    sigma *= 1.1
            else:
                # If only 1 parent, we can't estimate covariance well
                # Just shrink sigma slightly to explore locally
                sigma *= 0.9

            # If we are stuck (no improvement in best_f for many iterations), 
            # we might need to restart or increase sigma.
            # For now, the adaptive sigma reduction should handle convergence.
            # If sigma becomes too small, we might get stuck.
            if sigma < 1e-6:
                # Restart with a larger sigma around the current best
                sigma = 0.5
                C = np.eye(dim)
                mean = best_x.copy()

        # Phase 3: Final Polish with Nelder-Mead
        # Use remaining budget for fine-tuning the best point
        if calls < budget and time.time() < deadline and best_x is not None:
            # Initialize NM simplex
            x_start = best_x.copy()
            f_start = best_f
            
            # Step size for NM
            # Based on current sigma or a small fraction of the box
            nm_step = max(sigma * 0.5, 0.1)
            
            simplex = [x_start]
            f_values = [f_start]
            
            for i in range(dim):
                if calls >= budget or time.time() > deadline:
                    break
                xi = x_start.copy()
                xi[i] += nm_step
                fi = safe_call(xi)
                if fi is None:
                    break
                simplex.append(xi)
                f_values.append(fi)
            
            if len(simplex) == dim + 1:
                simplex = np.array(simplex)
                f_values = np.array(f_values)
                
                alpha = 1.0
                gamma = 2.0
                rho = 0.5
                sigma_nm = 0.5
                
                # NM Loop
                while calls < budget and time.time() < deadline:
                    idx = np.argsort(f_values)
                    simplex = simplex[idx]
                    f_values = f_values[idx]
                    
                    # Check convergence
                    spread_f = np.max(f_values) - np.min(f_values)
                    spread_x = np.max(simplex, axis=0) - np.min(simplex, axis=0)
                    
                    if np.max(spread_x) < 1e-7 or spread_f < 1e-12:
                        break
                    
                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    # Reflection
                    xr = centroid + alpha * (centroid - simplex[-1])
                    xr = np.clip(xr, -5.0, 5.0)
                    fr = safe_call(xr)
                    if fr is None:
                        break
                    
                    if fr < f_values[0]:
                        # Expansion
                        xe = centroid + gamma * (centroid - simplex[-1])
                        xe = np.clip(xe, -5.0, 5.0)
                        fe = safe_call(xe)
                        if fe is None:
                            break
                        if fe < fr:
                            simplex[-1] = xe
                            f_values[-1] = fe
                        else:
                            simplex[-1] = xr
                            f_values[-1] = fr
                    elif fr < f_values[-2]:
                        # Reflection accepted
                        simplex[-1] = xr
                        f_values[-1] = fr
                    else:
                        # Contraction
                        if fr < f_values[-1]:
                            # Outside contraction
                            xc = centroid + rho * (xr - centroid)
                        else:
                            # Inside contraction
                            xc = centroid - rho * (centroid - simplex[-1])
                        xc = np.clip(xc, -5.0, 5.0)
                        fc = safe_call(xc)
                        if fc is None:
                            break
                        if fc <= min(fr, f_values[-1]):
                            simplex[-1] = xc
                            f_values[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                if calls >= budget or time.time() > deadline:
                                    break
                                simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                                fi = safe_call(simplex[i])
                                if fi is None:
                                    break
                                f_values[i] = fi
                            if fi is None:
                                break

        return best_x, best_f
