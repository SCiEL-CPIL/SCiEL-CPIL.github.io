"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid global sampling + Nelder-Mead local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 330s hard limit

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def update_best(x, fx):
            nonlocal best_x, best_f, calls_made
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 1: Deterministic global sampling (Latin Hypercube + center)
        # Use ~150 calls for global exploration
        n_global = min(150, budget // 3)
        lxs = rng.uniform(-5, 5, size=(n_global, dim))
        
        # Add center and some structured points
        extra_points = [
            np.zeros(dim),
            np.full(dim, 0.5),
            np.full(dim, -0.5),
            np.full(dim, 1.0),
            np.full(dim, -1.0),
        ]
        
        all_points = list(lxs) + extra_points
        
        for x in all_points:
            if calls_made >= budget or time.time() > deadline:
                break
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            update_best(x, fx)

        # Phase 2: Local refinement using Nelder-Mead on the best points
        # Take the top 5 points from global phase
        # We need to re-evaluate or store them. Since we didn't store all,
        # let's just use the best found so far and a few random nearby points.
        # Better: Store the top K from the global phase.
        # Let's redo global phase to store results.
        
        # Reset for cleaner implementation
        best_x = None
        best_f = np.inf
        calls_made = 0
        results = []

        n_global = min(100, budget // 4)
        lxs = rng.uniform(-5, 5, size=(n_global, dim))
        extra_points = [
            np.zeros(dim),
            np.full(dim, 0.5),
            np.full(dim, -0.5),
        ]
        
        all_points = list(lxs) + extra_points
        
        for x in all_points:
            if calls_made >= budget or time.time() > deadline:
                break
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            results.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Sort results and pick top 3 for local refinement
        results.sort(key=lambda t: t[0])
        top_k = results[:3]

        # Phase 3: Nelder-Mead on top points
        # Use scipy's Nelder-Mead if available, else implement simple version
        try:
            from scipy.optimize import minimize
        except ImportError:
            minimize = None

        # We have budget - calls_made left
        remaining_budget = budget - calls_made
        
        # Allocate budget for local refinement
        # Each NM call uses 1 func eval. 
        # Set maxiter to use up remaining budget across the 3 starts.
        # But we must be careful not to exceed total budget.
        
        for i, (f_init, x_init) in enumerate(top_k):
            if calls_made >= budget or time.time() > deadline:
                break
            
            # Simple Nelder-Mead implementation to avoid scipy dependency issues or overhead
            # Or use scipy if it's fast enough. Let's implement a simple one to be safe and fast.
            
            # Simple Nelder-Mead
            x = x_init.copy()
            f = f_init
            
            # Initialize simplex
            simplex = [x.copy()]
            for j in range(dim):
                x_new = x.copy()
                step = 0.1 if x_new[j] == 0 else 0.1 * abs(x_new[j])
                x_new[j] += step
                x_new = np.clip(x_new, -5, 5)
                f_new = func(x_new)
                calls_made += 1
                if f_new < f:
                    f = f_new
                    x = x_new.copy()
                simplex.append(x_new.copy())
            
            # We have dim+1 points in simplex
            # Sort simplex
            simplex.sort(key=lambda pt: func(pt) if False else self._eval_cache(func, pt, calls_made))
            # This caching is complex. Let's just run a simple hill climb with adaptive step size.
            # It's often robust enough for these low-dim problems.
            
            # Adaptive Hill Climb
            sigma = 0.5
            x_cur = x_init.copy()
            f_cur = f_init
            
            # We have remaining_budget - (calls_made - calls_at_start_of_this_loop) left
            # Let's just run until budget or time is up, or convergence.
            
            # Actually, let's use a proper Nelder-Mead from scratch to ensure quality.
            # But given the constraint, a robust (1+1)-ES with adaptive sigma is often better.
            
            # Let's do a focused (1+1)-ES with aggressive exploration then exploitation.
            # This is essentially what the parent did, but let's make it more structured.
            
            # For now, let's just do a simple local search with decreasing step size.
            # This is a fallback. If scipy is available, use it.
            
            if minimize is not None:
                # Use scipy minimize
                # We need a wrapper that counts calls
                # But scipy minimize might make too many calls.
                # Let's limit it.
                
                # We'll just run a few iterations of NM manually.
                pass
            
            # Manual simple local search
            # Start from x_init
            x_loc = x_init.copy()
            f_loc = f_init
            
            # Adaptive step size
            step = 1.0
            for _ in range(remaining_budget // 3):  # rough allocation
                if calls_made >= budget or time.time() > deadline:
                    break
                
                # Try a few random directions
                found_improvement = False
                for _ in range(5):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    direction = rng.normal(size=dim)
                    direction /= (np.linalg.norm(direction) + 1e-8)
                    x_new = np.clip(x_loc + step * direction, -5, 5)
                    f_new = func(x_new)
                    calls_made += 1
                    
                    if f_new < f_loc:
                        x_loc = x_new
                        f_loc = f_new
                        step *= 1.2  # increase step if improving
                        found_improvement = True
                        if f_new < best_f:
                            best_f = f_new
                            best_x = x_new.copy()
                        break
                    else:
                        step *= 0.8  # decrease step if not improving
                
                if not found_improvement:
                    step *= 0.5
                    if step < 1e-4:
                        break

        return best_x, best_f

    def _eval_cache(self, func, x, calls_made):
        # This is a placeholder, not used in the main logic above
        pass
# EVOLVE-BLOCK-END
