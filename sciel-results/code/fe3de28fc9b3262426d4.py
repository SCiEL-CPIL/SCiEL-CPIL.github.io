"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with coarse grid for BBOB f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin before 300s limit

        # Count evaluations to stay within budget
        self._eval_count = 0

        def wrapped_func(x):
            if self._eval_count >= self.budget:
                return float('inf')
            self._eval_count += 1
            x = np.clip(x, -5.0, 5.0)
            return float(func(x))

        # Coarse grid to find basins for the 21-peak function
        # Use a 3x3x3x3x3 grid = 243 points, leaving ~257 for L-BFGS-B
        grid_points = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        for m in range(3):
                            pt = np.array([
                                -5 + 10 * i / 2,
                                -5 + 10 * j / 2,
                                -5 + 10 * k / 2,
                                -5 + 10 * l / 2,
                                -5 + 10 * m / 2
                            ])
                            grid_points.append(pt)

        # Shuffle grid points for diversity
        rng = np.random.default_rng(42)
        rng.shuffle(grid_points)

        best_x = None
        best_f = float('inf')

        # Phase 1: Evaluate grid points to find promising basins
        for pt in grid_points:
            if self._eval_count >= self.budget - 200:  # leave budget for local optimization
                break
            if time.time() > deadline:
                break
            f_val = wrapped_func(pt)
            if f_val < best_f:
                best_f = f_val
                best_x = pt.copy()

        # Phase 2: Run L-BFGS-B from multiple starting points
        # Use the best grid point plus several others
        start_points = [best_x] if best_x is not None else [np.zeros(self.dim)]
        
        # Add some additional starting points from the grid (top performers)
        grid_evals = []
        for pt in grid_points[:10]:
            if self._eval_count >= self.budget - 200:
                break
            f_val = wrapped_func(pt)
            grid_evals.append((f_val, pt))
        
        grid_evals.sort(key=lambda x: x[0])
        for _, pt in grid_evals[:5]:
            start_points.append(pt)

        # Add a few random points
        for _ in range(3):
            if self._eval_count >= self.budget - 200:
                break
            rp = rng.uniform(-4, 4, self.dim)
            start_points.append(rp)

        # Remove duplicates (approximately)
        unique_starts = []
        for sp in start_points:
            is_dup = False
            for us in unique_starts:
                if np.allclose(sp, us, atol=0.1):
                    is_dup = True
                    break
            if not is_dup:
                unique_starts.append(sp)

        # Run L-BFGS-B from each starting point
        for sp in unique_starts:
            if self._eval_count >= self.budget - 5:
                break
            if time.time() > deadline:
                break
            
            try:
                result = minimize(
                    wrapped_func,
                    sp,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * self.dim,
                    options={'maxiter': 50, 'maxls': 20, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if result.success or result.fun < best_f:
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
