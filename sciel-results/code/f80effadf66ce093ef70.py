# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Grid-based decomposition with local Nelder-Mead simplex per cell."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        all_X = []
        all_F = []

        def eval_point(x):
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=float)
            all_X.append(x.copy())
            all_F.append(f.copy())
            return f

        eval_count = [0]

        def count_eval(x):
            if eval_count[0] >= budget:
                raise StopIteration
            eval_count[0] += 1
            return eval_point(x)

        # Grid: 2x2x2x2 = 16 cells
        grid_size = 2
        n_cells = grid_size ** dim  # 16

        cell_bounds = []
        cell_centers = []
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    for l in range(grid_size):
                        lo_c = lo + span * np.array([i, j, k, l]) / grid_size
                        hi_c = lo + span * np.array([i + 1, j + 1, k + 1, l + 1]) / grid_size
                        cell_bounds.append((lo_c, hi_c))
                        cell_centers.append((lo_c + hi_c) / 2.0)

        # 16 weight vectors uniformly distributed
        weights = np.linspace(0, 1, n_cells)

        # Phase 1: For each cell, run Nelder-Mead with the weight corresponding to its index
        for idx in range(n_cells):
            if eval_count[0] >= budget:
                break

            w = weights[idx]
            lo_c, hi_c = cell_bounds[idx]
            x0 = cell_centers[idx]

            def scalar_obj(x, w=w):
                f = count_eval(x)
                return w * f[0] + (1 - w) * f[1]

            try:
                minimize(
                    scalar_obj,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 30,
                        'maxfev': 12,
                        'xatol': 1e-5,
                        'fatol': 1e-7,
                        'adaptive': True
                    }
                )
            except (StopIteration, Exception):
                pass

        # Phase 2: Use remaining budget for additional scalarizations with intermediate weights
        remaining = budget - eval_count[0]
        if remaining > 0:
            # Try a few more weight values in the middle range
            extra_weights = [0.25, 0.5, 0.75]
            for w in extra_weights:
                if eval_count[0] >= budget:
                    break

                def scalar_obj(x, w=w):
                    f = count_eval(x)
                    return w * f[0] + (1 - w) * f[1]

                # Start from the center of the whole space
                x0 = (lo + hi) / 2.0
                try:
                    minimize(
                        scalar_obj,
                        x0,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 20,
                            'maxfev': min(8, budget - eval_count[0]),
                            'xatol': 1e-5,
                            'fatol': 1e-7,
                            'adaptive': True
                        }
                    )
                except (StopIteration, Exception):
                    pass

        # Phase 3: Use any remaining budget for random sampling to fill gaps
        remaining = budget - eval_count[0]
        for _ in range(remaining):
            if eval_count[0] >= budget:
                break
            x0 = lo + span * np.random.rand(dim)
            try:
                count_eval(x0)
            except (StopIteration, Exception):
                pass

        return all_X
# EVOLVE-BLOCK-END
