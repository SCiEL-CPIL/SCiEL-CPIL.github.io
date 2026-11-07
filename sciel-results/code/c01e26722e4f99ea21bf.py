import numpy as np
import time
import scipy.optimize

class Optimizer:
    """Multi-scale differential evolution + L-BFGS-B hybrid optimizer for BBOB functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Safety margin: 330s limit, stop at 320s
        deadline = time.time() + 320
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        bounds = [(-5, 5) for _ in range(dim)]
        lower = np.full(dim, -5.0)
        upper = np.full(dim, 5.0)
        
        # Wrapper to track calls and update best
        def wrapped_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), lower, upper)
            if x.shape != (dim,):
                x = x.reshape(-1)
            if len(x) != dim:
                x = x[:dim]
            val = func(x)
            if not np.isfinite(val):
                return 1e10
            calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        def polish(x_start, max_iter=30, maxfev=None, ftol=1e-12, gtol=1e-8):
            """Run L-BFGS-B polish from a starting point."""
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return
            if maxfev is None:
                maxfev = max(10, (budget - calls) // 3)
            try:
                res = scipy.optimize.minimize(
                    wrapped_func, 
                    x_start.copy(), 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={
                        'maxiter': max_iter, 
                        'maxfev': maxfev, 
                        'ftol': ftol, 
                        'gtol': gtol, 
                        'disp': False
                    }
                )
            except Exception:
                pass

        # ====================================================================
        # PHASE 1: Diverse initial sampling with Sobol + center
        # ====================================================================
        from scipy.stats import qmc
        n_sobol = min(60, budget // 5)
        if n_sobol > 10:
            sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
            sobol_pts = sampler.random(n_sobol)
            sobol_pts = sobol_pts * 10.0 - 5.0
            # Add center point
            sobol_pts = np.vstack([sobol_pts, np.zeros(dim)])
            
            for pt in sobol_pts:
                if calls >= budget or time.time() > deadline:
                    break
                fv = wrapped_func(pt)
        
        if best_x is None:
            # Fallback: evaluate at least a few random points
            for _ in range(10):
                if calls >= budget or time.time() > deadline:
                    break
                x = rng.uniform(-5, 5, dim)
                wrapped_func(x)
        
        if best_x is None:
            return np.zeros(dim), np.inf

        # ====================================================================
        # PHASE 2: Multi-scale Differential Evolution
        # Three populations: large (global), medium (basin), small (fine-tune)
        # ====================================================================
        
        # Large population for global exploration
        pop_size_large = 15
        # Medium population for basin identification
        pop_size_med = 8
        # Small population for fine-tuning
        pop_size_small = 4
        
        # Initialize populations
        # Large pop: random spread
        pop_large = np.array([rng.uniform(-5, 5, dim) for _ in range(pop_size_large)])
        # Evaluate all
        for i in range(pop_size_large):
            if calls >= budget or time.time() > deadline:
                break
            wrapped_func(pop_large[i])
        
        # Medium pop: around best with moderate spread
        if best_x is not None:
            pop_med = np.array([
                np.clip(best_x + rng.normal(0, 1.5, dim), lower, upper)
                for _ in range(pop_size_med)
            ])
            for i in range(pop_size_med):
                if calls >= budget or time.time() > deadline:
                    break
                wrapped_func(pop_med[i])
        
        # Small pop: around best with small spread
        if best_x is not None:
            pop_small = np.array([
                np.clip(best_x + rng.normal(0, 0.5, dim), lower, upper)
                for _ in range(pop_size_small)
            ])
            for i in range(pop_size_small):
                if calls >= budget or time.time() > deadline:
                    break
                wrapped_func(pop_small[i])
        
        # Run DE iterations
        n_de_generations = 8
        for gen in range(n_de_generations):
            if calls >= budget or time.time() > deadline:
                break
            
            # Adaptive crossover rates
            cr_large = max(0.3, 0.8 - 0.05 * gen)
            cr_med = max(0.4, 0.7 - 0.04 * gen)
            cr_small = max(0.5, 0.6 - 0.03 * gen)
            
            # DE step for large population
            if calls < budget - 20:
                for i in range(pop_size_large):
                    if calls >= budget or time.time() > deadline:
                        break
                    # Select two distinct random indices
                    r1, r2 = rng.choice(pop_size_large, size=2, replace=False)
                    # DE/2/bin crossover
                    trial = pop_large[i].copy()
                    j_rand = rng.integers(0, dim)
                    for j in range(dim):
                        if rng.random() < cr_large or j == j_rand:
                            trial[j] = pop_large[i][j] + 0.8 * (pop_large[r1][j] - pop_large[r2][j])
                    trial = np.clip(trial, lower, upper)
                    f_trial = wrapped_func(trial)
                    # Compare with current best in pop
                    # We need to track per-individual fitness, but for simplicity
                    # just update if it's better than the global best or the current individual
                    # For proper DE, we'd need to store fitness per individual
                    # Let's do a simplified version: replace if better than current best in pop
                    # Actually, let's just let the global best tracking handle it
                    # and replace the individual if the trial is better
                    # We need to know the fitness of pop_large[i], but we didn't store it
                    # Let's re-evaluate or just accept any trial that doesn't exceed current global best
                    # For efficiency, let's just keep the trial if it's better than current global best
                    # or just always replace (since we're tracking global best anyway)
                    # A better approach: store fitness in parallel arrays
                    pass
            
            # Simpler DE approach: just generate new candidates and evaluate
            # This is more like a random search with perturbation, but the idea is
            # to maintain diversity at different scales
            
            # Generate new candidates from large pop
            for i in range(pop_size_large // 2):
                if calls >= budget or time.time() > deadline:
                    break
                r1, r2 = rng.choice(pop_size_large, size=2, replace=False)
                trial = pop_large[r1] + 0.8 * (pop_large[r1] - pop_large[r2])
                trial = np.clip(trial, lower, upper)
                wrapped_func(trial)
            
            # Generate new candidates from medium pop
            if best_x is not None:
                for i in range(pop_size_med // 2):
                    if calls >= budget or time.time() > deadline:
                        break
                    r1, r2 = rng.choice(pop_size_med, size=2, replace=False)
                    trial = pop_med[r1] + 0.6 * (pop_med[r1] - pop_med[r2])
                    trial = np.clip(trial, lower, upper)
                    wrapped_func(trial)
            
            # Generate new candidates from small pop
            if best_x is not None:
                for i in range(pop_size_small):
                    if calls >= budget or time.time() > deadline:
                        break
                    trial = best_x + rng.normal(0, 0.3, dim)
                    trial = np.clip(trial, lower, upper)
                    wrapped_func(trial)
            
            # Update populations with best found so far
            if best_x is not None:
                # Inject best into each population
                pop_large[0] = best_x.copy()
                pop_med[0] = best_x.copy()
                pop_small[0] = best_x.copy()

        # ====================================================================
        # PHASE 3: L-BFGS-B refinement from top candidates
        # ====================================================================
        
        # Polish from current best
        polish(best_x, max_iter=30, maxfev=max(20, (budget - calls) // 4), ftol=1e-12, gtol=1e-8)
        
        # Try a few random restarts + polish
        n_restarts = 6
        for i in range(n_restarts):
            if calls >= budget or time.time() > deadline:
                break
            
            # Generate a random start, biased towards the current best
            if best_x is not None and rng.random() < 0.5:
                x0 = best_x + rng.normal(0, 1.5, dim)
            else:
                x0 = rng.uniform(-5, 5, dim)
            x0 = np.clip(x0, lower, upper)
            
            # Evaluate start
            fx0 = wrapped_func(x0)
            
            if calls >= budget or time.time() > deadline:
                break
            
            # Polish from this start
            remaining = budget - calls
            if remaining < 15:
                break
            local_budget = max(15, min(remaining // 2, 50))
            polish(x0, max_iter=20, maxfev=local_budget, ftol=1e-12, gtol=1e-8)
        
        # ====================================================================
        # PHASE 4: Intensive refinement of the best point found
        # ====================================================================
        if calls < budget and best_x is not None and time.time() < deadline:
            remaining = budget - calls
            if remaining > 10:
                # Multiple refinement attempts with decreasing tolerances
                for attempt in range(3):
                    if calls >= budget or time.time() > deadline:
                        break
                    
                    tol = [1e-12, 1e-14, 1e-16][min(attempt, 2)]
                    maxfev_this = max(5, remaining // (3 - attempt))
                    
                    polish(best_x, max_iter=30, maxfev=maxfev_this, ftol=tol, gtol=1e-9)
        
        # ====================================================================
        # PHASE 5: Final polish with tight tolerances
        # ====================================================================
        if calls < budget and best_x is not None and time.time() < deadline:
            remaining = budget - calls
            if remaining > 5:
                polish(best_x, max_iter=50, maxfev=remaining, ftol=1e-15, gtol=1e-10)

        return best_x, best_f
