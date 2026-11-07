"""Combined optimizer: Simulated-annealing basin hopping + Nelder-Mead polish for Rastrigin f3."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """SA basin-hopping over integer lattice + Nelder-Mead polish for Rastrigin f3."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 330s kill
        rng = np.random.default_rng(42)

        calls = [0]

        def call_func(x):
            calls[0] += 1
            return func(x)

        def time_ok():
            return time.time() < deadline and calls[0] < budget

        # ---------- Phase 1: Deterministic Lattice Descent ----------
        # Rastrigin with shift s: f(m) ~ ||m - s||^2 for integer m.
        # We find the integer m* closest to s by minimizing f(m) on the lattice.
        # This is a convex optimization problem on the lattice.
        
        x_int = np.zeros(dim, dtype=int)
        f_current = call_func(x_int.astype(float))
        f_best = f_current
        x_best = x_int.copy()

        # Coordinate descent on integer lattice
        # Max iterations to prevent infinite loops (though convexity guarantees termination)
        max_lattice_iters = 50 
        for _ in range(max_lattice_iters):
            if not time_ok():
                break
            improved = False
            for ax in range(dim):
                # Try -1
                x_prop = x_int.copy()
                x_prop[ax] -= 1
                if x_prop[ax] >= -4: # Stay in bounds
                    f_prop = call_func(x_prop.astype(float))
                    if f_prop < f_current:
                        x_int = x_prop
                        f_current = f_prop
                        improved = True
                        if f_prop < f_best:
                            f_best = f_prop
                            x_best = x_int.copy()
                
                # Try +1
                x_prop = x_int.copy()
                x_prop[ax] += 1
                if x_prop[ax] <= 4: # Stay in bounds
                    f_prop = call_func(x_prop.astype(float))
                    if f_prop < f_current:
                        x_int = x_prop
                        f_current = f_prop
                        improved = True
                        if f_prop < f_best:
                            f_best = f_prop
                            x_best = x_int.copy()
            
            if not improved:
                break

        # ---------- Phase 2: Nelder-Mead polish from best integer basin ----------
        # Use scipy's Nelder-Mead for fast convergence to basin minimum
        from scipy.optimize import minimize

        x0 = x_best.astype(float)

        # Wrap func for scipy (it passes 1-D arrays)
        def wrapper(x):
            x = np.clip(x, -5, 5)
            return call_func(x)

        # Run Nelder-Mead with tight tolerance
        try:
            res = minimize(wrapper, x0, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12,
                                    'adaptive': True})
            if res.fun < f_best:
                f_best = res.fun
                x_best = np.clip(res.x, -5, 5).copy()
        except Exception:
            pass

        # ---------- Phase 3: Verification and Polish ----------
        # The lattice descent should have found the best basin.
        # We can spend remaining budget on high-precision polish.
        # No need for random basin exploration if lattice descent worked.
        # Just ensure we have enough budget for a thorough Nelder-Mead run.

        # ---------- Phase 4: Final (1+1)-ES fine-tuning with remaining budget ----------
        if time_ok():
            sigma = 0.05
            for _ in range(budget - calls[0]):
                if not time_ok():
                    break
                step = sigma * rng.standard_normal(dim)
                y = np.clip(x_best + step, -5, 5)
                fy = call_func(y)
                if fy < f_best:
                    f_best = fy
                    x_best = y.copy()
                    sigma *= 1.03
                else:
                    sigma *= 0.95
                if sigma < 0.0005:
                    break

        return x_best, f_best
# EVOLVE-BLOCK-END
