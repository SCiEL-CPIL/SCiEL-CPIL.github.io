"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Adaptive Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 300s limit, aim for 280s
        time_limit = 280 
        dim = self.dim
        budget = self.budget
        bounds_low = -5.0
        bounds_high = 5.0

        best_x = None
        best_f = np.inf
        calls_used = 0

        def safe_call(x):
            nonlocal calls_used
            if calls_used >= budget or time.time() - start_time > time_limit:
                return np.inf
            x_clipped = np.clip(np.asarray(x, dtype=np.float64), bounds_low, bounds_high)
            calls_used += 1
            try:
                val = func(x_clipped)
                return float(val)
            except Exception:
                return np.inf

        # --- Phase 1: Initial Exploration ---
        # We use a mix of structured points and random points to quickly identify basins.
        # AOCC is heavily weighted towards early progress, so we need to find a *good*
        # point quickly, not just *any* point.
        
        # Start with the origin and axis-aligned points, then random
        initial_points = []
        
        # 1. Origin
        initial_points.append(np.zeros(dim))
        
        # 2. Axis-aligned +/-
        for i in range(dim):
            for sign in [-1, 1]:
                p = np.zeros(dim)
                p[i] = sign * 3.0
                initial_points.append(p)
        
        # 3. Random points in [-4, 4] to cover the space
        # We will evaluate as many as we have budget for, up to 30
        max_initial = min(30, budget - 50) # Leave 50 for local search
        for _ in range(max_initial):
            p = np.random.uniform(bounds_low + 1.0, bounds_high - 1.0, dim)
            initial_points.append(p)

        # Shuffle to ensure diversity in early calls? 
        # Actually, structured first is often better for AOCC as we might hit the basin center early.
        # But if we hit a bad basin, we need randoms. 
        # Let's interleave: Center, then mix randoms and structured.
        
        # Let's just evaluate them in a semi-random order to avoid bias, 
        # but keep the first few very specific.
        # Actually, standard NM is sensitive to start point. 
        # Let's evaluate the first 10 carefully, then start NM on the best.
        
        # Sort initial points by norm? No, just evaluate them.
        # To maximize early AOCC, we should evaluate the most promising points first.
        # "Most promising" is hard to know. Let's stick to a diverse set.
        
        # Evaluate initial points
        evaluated = []
        for p in initial_points:
            if calls_used >= budget or time.time() - start_time > time_limit:
                break
            f = safe_call(p)
            if f < best_f:
                best_f = f
                best_x = p.copy()
            evaluated.append((f, p.copy()))

        if not evaluated:
            return best_x, best_f

        # Sort by fitness
        evaluated.sort(key=lambda x: x[0])

        # --- Phase 2: Multi-start Nelder-Mead ---
        # We will run multiple local optimizations.
        # Each local optimization starts from a good initial point.
        # If one stagnates, we stop it and start another.
        
        # Parameters for Nelder-Mead
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink
        eps = 1e-8
        
        # We select the top K initial points to start NM from
        K = min(5, len(evaluated))
        
        # We have a limited budget. We should allocate the remaining budget to these local searches.
        # A good strategy: Use 50% of remaining budget for the best start, 
        # 20% for second best, etc.
        # Or simply run NM until convergence or a small budget limit, then next.
        
        # Let's define a max_iter for each local search to prevent getting stuck in a bad basin.
        # Total remaining budget:
        rem_budget = budget - calls_used
        
        # If we have very little budget, just return best
        if rem_budget < 10:
            return best_x, best_f

        # We will try to run local searches from the top K points.
        # But we must be careful with time.
        
        for k in range(K):
            if calls_used >= budget or time.time() - start_time > time_limit:
                break
            
            f_start, x_start = evaluated[k]
            
            # If this start is already worse than the global best by a lot, skip?
            # Actually, if f_start > best_f + 1.0, it might be a different basin.
            # But if it's the same basin, it might still lead to a better solution.
            # Let's just proceed.
            
            # Initialize Simplex
            simplex = [x_start.copy()]
            for i in range(dim):
                dx = x_start.copy()
                # Initial step size: adaptive to the scale of the problem
                # If we are near 0, step is 0.5. If far, maybe larger.
                step = 0.5
                dx[i] += step
                simplex.append(dx)
            
            f_simplex = [safe_call(s) for s in simplex]
            
            # Update global best with simplex points
            for i in range(dim + 1):
                if f_simplex[i] < best_f:
                    best_f = f_simplex[i]
                    best_x = simplex[i].copy()

            # Local Nelder-Mead Loop
            # Max iterations for this local search: 
            # Let's say we allow 50 iterations per local search.
            # 5 searches * 50 iters * 6 calls/iter = 1500 calls. Too many.
            # We have ~470 calls left.
            # So we should limit iterations per search to roughly (rem_budget / K) / 6.
            # But we want to finish early if converged.
            
            max_iter_local = 50
            no_improve_count = 0
            prev_best_local = best_f
            
            for it in range(max_iter_local):
                if calls_used >= budget or time.time() - start_time > time_limit:
                    break
                
                # Sort simplex by fitness
                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = [f_simplex[i] for i in order]
                
                # Check convergence: if spread is small, break
                if f_simplex[-1] - f_simplex[0] < 1e-6:
                    break
                
                # Check stagnation
                if f_simplex[0] >= prev_best_local - 1e-10:
                    no_improve_count += 1
                else:
                    no_improve_count = 0
                    prev_best_local = f_simplex[0]
                
                if no_improve_count > 10:
                    break # Stagnated, move to next start point

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                f_worst = f_simplex[-1]

                # 1. Reflection
                xr = centroid + alpha * (centroid - worst)
                fr = safe_call(xr)
                
                if fr < f_simplex[0]:
                    # 2. Expansion
                    xe = centroid + gamma * (centroid - worst)
                    fe = safe_call(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        f_simplex[-1] = fe
                    else:
                        simplex[-1] = xr
                        f_simplex[-1] = fr
                elif fr < f_simplex[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    f_simplex[-1] = fr
                else:
                    # 3. Contraction
                    xc = centroid + rho * (worst - centroid)
                    fc = safe_call(xc)
                    if fc < f_worst:
                        simplex[-1] = xc
                        f_simplex[-1] = fc
                    else:
                        # 4. Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_simplex[i] = safe_call(simplex[i])
                
                # Update global best
                for i in range(dim + 1):
                    if f_simplex[i] < best_f:
                        best_f = f_simplex[i]
                        best_x = simplex[i].copy()

            # After local search, if we have budget left, and we found a better best_x,
            # we might want to restart NM from the new best_x to refine further.
            # But we already did that in the loop.
            # If we broke due to stagnation, we proceed to the next start point k+1.
            
            # Optimization: If the local search significantly improved the global best,
            # and we still have budget, it might be worth doing one more refinement
            # from the *new* global best, even if we've used up our "K" slots.
            # But to keep it simple and fast, we'll rely on the K slots.

        return best_x, best_f
# EVOLVE-BLOCK-END
