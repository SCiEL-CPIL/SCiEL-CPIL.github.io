"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES based optimizer for ill-conditioned rotated functions."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 50s safety margin before 330s kill

        # Initial point at center of box
        x0 = np.zeros(self.dim)

        # Wrap func to count evaluations and enforce time limit
        eval_count = [0]
        max_evals = self.budget

        def wrapper(x):
            if eval_count[0] >= max_evals:
                return 1e10
            x = np.clip(x, -5, 5)
            val = func(x)
            eval_count[0] += 1
            return val

        # Use CMA-ES via scipy
        # popsize for dim=5 is ~9, so ~55 generations = ~495 evals
        # maxiter controls generations
        popsize = 4 + int(3 * np.log(self.dim))
        max_iter = (max_evals - 1) // popsize  # leave 1 eval margin
        max_iter = max(max_iter, 10)

        try:
            result = minimize(
                wrapper,
                x0,
                method='CMA',
                bounds=[(-5, 5)] * self.dim,
                options={
                    'maxiter': max_iter,
                    'popsize': popsize,
                    'seed': 42,
                    'verbose': 0,
                }
            )
            best_x = np.clip(result.x, -5, 5)
            best_f = wrapper(best_x)
            eval_count[0] += 1

        except Exception:
            # Fallback: simple random search + local hill climb
            rng = np.random.default_rng(42)
            best_x = x0.copy()
            best_f = func(best_x)
            eval_count[0] = 1

            sigma = 2.0
            while eval_count[0] < max_evals:
                if time.time() > deadline:
                    break
                y = np.clip(best_x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                eval_count[0] += 1
                if fy < best_f:
                    best_x, best_f = y, fy
                    sigma *= 1.3
                else:
                    sigma *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
