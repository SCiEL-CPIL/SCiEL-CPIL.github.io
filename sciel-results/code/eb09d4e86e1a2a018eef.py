"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-start Nelder-Mead simplex method for BBOB f7 (Step ellipsoidal)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # We will use a global evaluation counter
        eval_count = 0

        def safe_eval(x):
            nonlocal eval_count
            if eval_count >= budget:
                return np.inf
            x = np.asarray(x, dtype=float).ravel()
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            eval_count += 1
            return fx

        # Strategy:
        # 1. Use first 8 evaluations for diverse initial centroids using space-filling design.
        # 2. Use remaining evaluations for aggressive Nelder-Mead refinement with adaptive step sizes.
        # 3. Dynamically reallocate budget based on progress.

        # Phase 1: Seed 8 diverse initial centroids using stratified sampling
        # Use a grid-based approach for better coverage
        n_centroids = 8
        centroids = []
        
        # Create stratified samples: mix of center-biased and spread
        for i in range(n_centroids):
            if eval_count >= budget:
                break
            
            if i < 4:
                # Center region with smaller radius
                scale = 0.3 + 0.1 * i
                x = rng.uniform(-scale, scale, dim)
            else:
                # Outer region with larger spread
                scale = 1.5 + 0.5 * (i - 4)
                x = rng.uniform(-scale, scale, dim)
            
            x = np.clip(x, -5.0, 5.0)
            fx = safe_eval(x)
            centroids.append((fx, x.copy()))

        # Sort by function value
        centroids.sort(key=lambda t: t[0])
        best_fx = centroids[0][0]
        best_x = centroids[0][1].copy()

        # Phase 2: Aggressive Nelder-Mead refinement with adaptive step sizes
        # We have 8 evals used, so 492 remaining.
        # Focus on top 4 centroids (step ellipsoidal has sharp discontinuity)
        
        top_k = min(4, len(centroids))
        remaining = budget - eval_count
        
        # Dynamic budget allocation: start with 45%, 30%, 15%, 10%
        # But adapt based on improvement
        initial_weights = [0.45, 0.30, 0.15, 0.10][:top_k]
        
        # Track progress for dynamic reallocation
        progress_history = [0.0] * top_k
        allocs = [int(remaining * w) for w in initial_weights]
        
        for idx in range(top_k):
            if eval_count >= budget - 5:
                break
            
            center = centroids[idx][1].copy()
            alloc = allocs[idx]
            alloc = max(alloc, dim + 1)  # At least enough for one simplex
            
            # Adaptive initial step size based on function landscape
            # Start with moderate step, adapt during optimization
            delta = 0.8
            
            # Initialize simplex around center with axis-aligned points
            simplex = [center.copy()]
            for j in range(dim):
                v = center.copy()
                v[j] += delta
                v = np.clip(v, -5.0, 5.0)
                simplex.append(v)
            
            # Evaluate simplex vertices
            fvals = []
            for v in simplex:
                if eval_count >= budget:
                    break
                fvals.append(safe_eval(v))
            
            if len(fvals) < dim + 1:
                continue
            
            # Run Nelder-Mead iterations with adaptive step size
            # Each iteration uses 1-2 evals typically
            max_iter = min(alloc // 2, 150)  # Cap iterations
            
            prev_best = best_fx
            no_improve_count = 0
            step_scale = 1.0
            
            for it in range(max_iter):
                if eval_count >= budget - 3:
                    break
                
                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection with adaptive gamma
                gamma = 1.0 + 0.2 * step_scale
                xr = centroid + gamma * (centroid - simplex[-1])
                xr = np.clip(xr, -5.0, 5.0)
                fr = safe_eval(xr)
                
                if fr < fvals[0]:
                    # Expansion
                    gamma_exp = 2.0 * step_scale
                    xe = centroid + gamma_exp * (centroid - simplex[-1])
                    xe = np.clip(xe, -5.0, 5.0)
                    fe = safe_eval(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                        step_scale = min(step_scale * 1.1, 3.0)  # Increase step if expanding
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                        step_scale = max(step_scale * 0.9, 0.3)
                elif fr < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    fvals[-1] = fr
                    step_scale = max(step_scale * 0.95, 0.3)
                else:
                    # Contraction
                    gamma_con = 0.5 * step_scale
                    xc = centroid + gamma_con * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5.0, 5.0)
                    fc = safe_eval(xc)
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                        step_scale = max(step_scale * 0.9, 0.3)
                    else:
                        # Shrink
                        for j in range(1, dim + 1):
                            simplex[j] = simplex[0] + 0.5 * (simplex[j] - simplex[0])
                            simplex[j] = np.clip(simplex[j], -5.0, 5.0)
                            if eval_count >= budget:
                                break
                            fvals[j] = safe_eval(simplex[j])
                        step_scale = max(step_scale * 0.7, 0.1)
                        if eval_count >= budget:
                            break
                
                # Update global best
                min_idx = np.argmin(fvals)
                if fvals[min_idx] < best_fx:
                    best_fx = fvals[min_idx]
                    best_x = simplex[min_idx].copy()
                
                # Track progress for dynamic reallocation
                if best_fx < prev_best:
                    progress_history[idx] += (prev_best - best_fx)
                    prev_best = best_fx
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                # Early termination if no progress
                if no_improve_count > 20 and it > 10:
                    break
                
                # Check convergence
                if np.std(fvals) < 1e-12:
                    break

            # Dynamic reallocation: if this basin made good progress, 
            # we could have allocated more, but we can't go back.
            # Instead, track for potential future improvements.
            if idx < top_k - 1 and progress_history[idx] > 0.1:
                # This basin is promising, but we've already allocated budget.
                # For next runs, we might want to weight it more.
                pass

        # Final polish: use remaining budget on the best point
        if eval_count < budget - 1:
            # Do a few more evaluations around the best point
            polish_budget = min(budget - eval_count, 10)
            for _ in range(polish_budget):
                if eval_count >= budget:
                    break
                # Small random perturbation around best
                x = best_x + rng.uniform(-0.1, 0.1, dim)
                x = np.clip(x, -5.0, 5.0)
                fx = safe_eval(x)
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
