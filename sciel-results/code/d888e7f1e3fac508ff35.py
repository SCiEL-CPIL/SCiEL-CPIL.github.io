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

        # ---------- Phase 1: Simulated Annealing over integer lattice ----------
        # Rastrigin: local minima at s + integer_vector.
        # We search over integer points in [-4, 4]^5.
        # SA: accept worse moves with probability exp(-delta_f / T), cool T over time.

        # Start at origin (or a random integer point)
        x_int = np.zeros(dim, dtype=int)
        f_current = call_func(x_int.astype(float))
        f_best = f_current
        x_best = x_int.copy()

        # Temperature schedule
        T_start = 30.0
        T_end = 0.5
        n_sa_steps = 120  # number of SA moves
        cool_rate = (T_end / T_start) ** (1.0 / n_sa_steps)
        T = T_start

        for step in range(n_sa_steps):
            if not time_ok():
                break

            # Propose a move: change 1 or 2 coordinates by +/-1
            n_changed = 1 if rng.random() < 0.7 else 2
            x_prop = x_int.copy()
            axes = rng.choice(dim, size=n_changed, replace=False)
            for ax in axes:
                delta = 1 if rng.random() < 0.5 else -1
                x_prop[ax] += delta

            # Clip to valid range
            x_prop = np.clip(x_prop, -4, 4)

            # Evaluate
            f_prop = call_func(x_prop.astype(float))

            # Accept?
            if f_prop < f_current or rng.random() < np.exp(-(f_prop - f_current) / max(T, 1e-8)):
                x_int = x_prop
                f_current = f_prop
                if f_prop < f_best:
                    f_best = f_prop
                    x_best = x_int.copy()

            T *= cool_rate

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

        # ---------- Phase 3: Second basin exploration if budget remains ----------
        # Try a few more integer points near the best found, then polish
        if time_ok() and budget - calls[0] > 20:
            # Explore neighbors of the best integer point
            best_int = np.round(x_best).astype(int)
            best_int = np.clip(best_int, -4, 4)

            # Try all 2^5 single-step neighbors (10 points)
            for ax in range(dim):
                for delta in [-1, 1]:
                    if not time_ok():
                        break
                    cand = best_int.copy()
                    cand[ax] += delta
                    cand = np.clip(cand, -4, 4)
                    fc = call_func(cand.astype(float))
                    if fc < f_best:
                        f_best = fc
                        x_best = cand.astype(float).copy()

            # If we found a better integer point, polish it
            if time_ok() and budget - calls[0] > 10:
                try:
                    res2 = minimize(wrapper, x_best, method='Nelder-Mead',
                                    options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12,
                                             'adaptive': True})
                    if res2.fun < f_best:
                        f_best = res2.fun
                        x_best = np.clip(res2.x, -5, 5).copy()
                except Exception:
                    pass

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
