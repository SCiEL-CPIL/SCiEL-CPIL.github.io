"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start local search with structured global sampling for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        deadline = time.time() + 285.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        # Phase 1: Coarse global sampling to identify promising basins
        # Use a structured grid + Halton-like sequence for coverage
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        call_count = 0

        def check_deadline():
            return time.time() > deadline

        def record(x, f):
            nonlocal best_x, best_f, call_count
            if f < best_f:
                best_x, best_f = x.copy(), f

        # Coarse grid: 3^5 = 243 points
        grid_vals = np.linspace(-4, 4, 3)
        grid_points = list(np.meshgrid(*([grid_vals] * dim), indexing='ij'))
        grid_points = np.array([p.flatten() for p in zip(*grid_points)])
        
        for p in grid_points:
            if check_deadline() or call_count >= budget - 100:
                break
            f = func(p)
            call_count += 1
            record(p, f)

        # Add random points for better coverage
        n_random = min(200, budget - 100 - call_count)
        for _ in range(n_random):
            if check_deadline() or call_count >= budget - 100:
                break
            p = rng.uniform(-4.5, 4.5, dim)
            f = func(p)
            call_count += 1
            record(p, f)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            call_count += 1

        # Phase 2: Local refinement around the best point using adaptive step sizes
        # Use a simple Nelder-Mead-like approach with shrinking steps
        x = best_x.copy()
        fx = best_f
        step = 1.0
        remaining = budget - call_count
        iterations = 0

        while iterations < remaining and not check_deadline():
            # Try perturbations in each dimension
            improved = False
            for d in range(dim):
                if check_deadline() or iterations >= remaining:
                    break
                
                # Positive step
                x_trial = x.copy()
                x_trial[d] = np.clip(x_trial[d] + step, -5, 5)
                f_trial = func(x_trial)
                call_count += 1
                iterations += 1
                
                if f_trial < fx:
                    x, fx = x_trial, f_trial
                    improved = True
                else:
                    # Negative step
                    x_trial = x.copy()
                    x_trial[d] = np.clip(x_trial[d] - step, -5, 5)
                    f_trial = func(x_trial)
                    call_count += 1
                    iterations += 1
                    
                    if f_trial < fx:
                        x, fx = x_trial, f_trial
                        improved = True
            
            if improved:
                step *= 0.9  # Slightly reduce step on success
            else:
                step *= 0.5  # Reduce step more on failure
                if step < 1e-6:
                    break

        # Phase 3: Final polish with very small steps if budget remains
        if call_count < budget and not check_deadline():
            x = best_x if best_x is not None else x
            fx = best_f if best_x is not None else fx
            for _ in range(min(50, budget - call_count)):
                if check_deadline():
                    break
                for d in range(dim):
                    if check_deadline() or call_count >= budget:
                        break
                    for sign in [1, -1]:
                        if call_count >= budget:
                            break
                        x_trial = x.copy()
                        x_trial[d] = np.clip(x_trial[d] + sign * 0.01, -5, 5)
                        f_trial = func(x_trial)
                        call_count += 1
                        if f_trial < fx:
                            x, fx = x_trial, f_trial

            if fx < best_f:
                best_x, best_f = x, fx

        return best_x, best_f
# EVOLVE-BLOCK-END
