"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Sobol Initialization + IDW Surrogate-Guided Adaptive Refinement for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # State
        best_x = np.zeros(dim)
        best_f = np.inf
        call_count = 0
        rng = np.random.default_rng(42)
        
        # Data storage for Surrogate
        X_data = []
        y_data = []
        
        def do_call(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget or time.time() > deadline:
                return None
            x = np.clip(x, -5, 5)
            try:
                f = float(func(x))
            except Exception:
                f = np.inf
            call_count += 1
            X_data.append(x)
            y_data.append(f)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Sobol Space-Filling Exploration
        # We want to cover the domain [-5, 5]^5 systematically.
        # 3^5 = 243 points is a good balance. 
        # Sobol sequence gives better coverage than random or grid for low dimensions.
        # We will use a quasi-random sequence. Since scipy is available, we can use it, 
        # but to be safe with imports, we can implement a simple Halton or use numpy's random 
        # with stratification. 
        # Actually, let's use a simple stratified random sampling (Latin Hypercube) 
        # which is easy to implement and effective for 5D.
        
        n_phase1 = 100  # 100 points for initial map
        
        # Latin Hypercube Sampling in [-5, 5]
        # For each dimension, divide into n_phase1 strata, pick one point in each.
        # This ensures 1D uniformity.
        points = np.zeros((n_phase1, dim))
        for d in range(dim):
            # Create strata: [0, 1/n, 1/n, 2/n, ..., 1]
            strata = np.arange(n_phase1) / n_phase1
            # Shuffle to decorrelate dimensions
            perm = rng.permutation(n_phase1)
            # Sample uniformly within each stratum
            # Sample in [0, 1)
            samples = (strata[perm] + rng.random(n_phase1)) / n_phase1
            # Map to [-5, 5]
            points[:, d] = -5 + 10 * samples
        
        # Evaluate Phase 1
        for p in points:
            f = do_call(p)
            if f is None:
                break

        if call_count >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: Surrogate-Guided Refinement
        # We have ~100 points. We will iteratively propose new points using IDW.
        # IDW Prediction: f(x) = sum(w_i * f_i) / sum(w_i), where w_i = 1 / (dist_i^2 + eps)
        # We want to find the minimum of the IDW surface.
        
        # To find the minimum of IDW, we can:
        # 1. Sample a large set of candidate points (e.g., 1000 random points in [-5,5]).
        # 2. Evaluate IDW for all of them.
        # 3. Pick the top K (e.g., 10) candidates with lowest predicted value.
        # 4. Evaluate the true function at these K points.
        # 5. Update X_data, y_data.
        # 6. Repeat until budget is low.
        
        # Optimization: As we get closer to the optimum, the IDW surface becomes smoother 
        # and less useful for exploration. We should switch to local search (Nelder-Mead) 
        # once we have a good local basin.
        
        # Strategy:
        # - Use IDW for first ~200 calls (total).
        # - Then switch to Nelder-Mead on the best point found so far.
        
        n_refinement = 100 # 100 more points via IDW
        # Total used so far: 100. Remaining: 400.
        # Let's use IDW for 150 more points (total 250), then NM for the rest.
        
        idw_eps = 1e-4
        n_candidates = 500 # Number of candidates to sample for IDW min
        
        for step in range(n_refinement):
            if call_count >= budget or time.time() > deadline:
                break
            
            # Build IDW model from current data
            X = np.array(X_data)
            y = np.array(y_data)
            
            # Sample candidate points
            # To focus refinement, we can sample candidates in a box around the current best,
            # or globally. Given multimodality, global is safer early, local later.
            # Let's mix: 50% global, 50% local around best.
            
            cand_global = rng.uniform(-5, 5, size=(n_candidates//2, dim))
            # Local candidates: best_x + noise
            # Scale noise by current "spread" or just fixed 1.0
            cand_local = best_x + rng.normal(0, 1.0, size=(n_candidates//2, dim))
            cand_local = np.clip(cand_local, -5, 5)
            
            candidates = np.vstack([cand_global, cand_local])
            
            # Compute IDW values for candidates
            # dists shape: (n_cand, n_data)
            dists = np.linalg.norm(candidates[:, np.newaxis, :] - X[np.newaxis, :, :], axis=2)
            weights = 1.0 / (dists**2 + idw_eps)
            
            # Predicted values: (n_cand)
            pred_vals = np.sum(weights * y[np.newaxis, :], axis=1) / np.sum(weights, axis=1)
            
            # Find the best candidate (lowest predicted value)
            # To avoid evaluating the exact same point again, we pick the best one
            # that is not already in X_data (or close to it).
            
            # Sort candidates by predicted value
            sorted_idx = np.argsort(pred_vals)
            
            next_x = None
            for idx in sorted_idx[:5]: # Check top 5
                x_cand = candidates[idx]
                # Check if we've already evaluated this point (within tolerance)
                # Simple check: distance to all existing points
                d_to_existing = np.linalg.norm(X - x_cand, axis=1)
                if np.min(d_to_existing) > 0.01:
                    next_x = x_cand
                    break
            
            if next_x is None:
                # If all top candidates are duplicates, just pick the best one
                next_x = candidates[sorted_idx[0]]
            
            f = do_call(next_x)
            if f is None:
                break

        if call_count >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 3: Nelder-Mead Local Optimization
        # Now that we have a good map and a likely global basin, run NM from best_x.
        
        # Construct initial simplex
        # If we have many points, we could pick a diverse set, but standard NM uses best + basis.
        # Let's use a slightly larger initial step to ensure we are in the basin.
        initial_step = 0.5
        
        simplex = [best_x.copy()]
        for i in range(dim):
            v = best_x.copy()
            v[i] += initial_step
            simplex.append(v)
            
        # Evaluate simplex vertices (skip if already evaluated? 
        # do_call doesn't cache, so we might waste calls on duplicates. 
        # Given NM is fast, this is acceptable if we ensure simplex points are new.
        # However, best_x is already evaluated. 
        # Let's just evaluate the new vertices.
        
        # We need to handle the fact that best_x is already in X_data.
        # NM loop manages its own population.
        
        pop = []
        # Add best
        pop.append((best_f, best_x))
        
        for i in range(dim):
            v = best_x.copy()
            v[i] += initial_step
            f_v = do_call(v)
            if f_v is None:
                return best_x, best_f
            pop.append((f_v, v))
            
        # Sort pop
        pop.sort(key=lambda item: item[0])
        
        # Nelder-Mead Loop
        alpha_n = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        max_nm_iter = 100
        
        for _ in range(max_nm_iter):
            if call_count + dim + 2 > budget or time.time() > deadline:
                break
                
            pop.sort(key=lambda item: item[0])
            
            f_best = pop[0][0]
            f_worst = pop[-1][0]
            
            if abs(f_worst - f_best) < 1e-6:
                break
            
            x_worst = pop[-1][1]
            f_worst_val = pop[-1][0]
            
            x_centroid = np.mean([p[1] for p in pop[:-1]], axis=0)
            
            # Reflection
            x_ref = x_centroid + alpha_n * (x_centroid - x_worst)
            f_ref = do_call(x_ref)
            if f_ref is None:
                break
            
            if f_ref < pop[0][0]:
                # Expansion
                x_exp = x_centroid + gamma * (x_centroid - x_worst)
                f_exp = do_call(x_exp)
                if f_exp is None:
                    break
                
                if f_exp < f_ref:
                    pop[-1] = (f_exp, x_exp)
                else:
                    pop[-1] = (f_ref, x_ref)
            elif f_ref < f_worst_val:
                pop[-1] = (f_ref, x_ref)
            else:
                # Contraction
                x_con = x_centroid + rho * (x_worst - x_centroid)
                f_con = do_call(x_con)
                if f_con is None:
                    break
                
                if f_con < f_worst_val:
                    pop[-1] = (f_con, x_con)
                else:
                    # Shrink
                    x_best = pop[0][1]
                    for i in range(1, len(pop)):
                        x_shrink = x_best + sigma * (pop[i][1] - x_best)
                        f_shrink = do_call(x_shrink)
                        if f_shrink is None:
                            return best_x, best_f
                        pop[i] = (f_shrink, x_shrink)
                    
                    pop.sort(key=lambda item: item[0])

        return best_x, best_f

# EVOLVE-BLOCK-END
