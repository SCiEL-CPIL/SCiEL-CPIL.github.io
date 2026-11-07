"""Hybrid LHS + BFGS optimizer for BBOB f19 (Griewank-Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Deterministic multi-start LHS + accelerated BFGS local search for BBOB f19."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        n = self.dim
        max_calls = self.budget

        best_x = np.zeros(n)
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= max_calls:
                return None
            try:
                x = np.asarray(x, dtype=np.float64)
                x = np.clip(x, -5.0, 5.0)
                f = float(func(x))
                calls_made += 1
                if np.isfinite(f) and f < best_f:
                    best_f = f
                    best_x = x.copy()
                return f
            except Exception:
                return None

        # Phase 1: Stratified Latin Hypercube sampling for global coverage
        # Use a stratified LHS grid with multiple points per cell
        n_lhs = 20  # 20 points per dimension for 5D = 20 points total (one per cell)
        # Actually for 5D, we want a reasonable number of starting points
        # Let's use 15 starting points from LHS
        n_starts = 15
        
        # Generate LHS points
        rng = np.random.default_rng(42)
        lhs_points = []
        for _ in range(n_starts):
            # Each dimension gets a random stratified point
            pt = np.zeros(n)
            for d in range(n):
                # Stratify: divide [-5, 5] into n_starts cells, pick random point in one cell
                cell = rng.integers(0, n_starts)
                lower = -5.0 + cell * (10.0 / n_starts)
                upper = -5.0 + (cell + 1) * (10.0 / n_starts)
                pt[d] = rng.uniform(lower, upper)
            lhs_points.append(pt)
        
        # Ensure we also include some deterministic points for coverage
        # Add center and corners
        lhs_points.append(np.zeros(n))
        # Add a few more spread points
        for _ in range(5):
            pt = rng.uniform(-4, 4, n)
            lhs_points.append(pt)
        
        # Evaluate all LHS points
        for pt in lhs_points:
            if calls_made >= max_calls or time.time() > deadline:
                break
            f = safe_call(pt)
            if f is None:
                continue
        
        # Phase 2: BFGS local refinement from each basin
        # Sort starting points by their function values and refine the best ones
        # We'll do BFGS from the best LHS points found
        # Collect evaluated points and their values
        evaluated_points = []
        for pt in lhs_points:
            if calls_made >= max_calls or time.time() > deadline:
                break
            # We already evaluated these, but let's track them
            # Actually, let's just do BFGS from the current best and a few others
        
        # BFGS from the best point found so far
        if calls_made < max_calls and time.time() < deadline:
            result = minimize(
                lambda x: func(np.clip(x, -5, 5)),
                best_x.copy(),
                method='BFGS',
                options={'maxiter': 50, 'maxfun': 30}
            )
            if result.success and np.isfinite(result.fun):
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5).copy()
        
        # BFGS from other promising points
        # Try BFGS from a few other starting points
        other_starts = []
        # Add some perturbed versions of the best point
        for _ in range(3):
            if calls_made >= max_calls or time.time() > deadline:
                break
            perturbed = best_x + rng.normal(0, 1.0, n)
            perturbed = np.clip(perturbed, -5, 5)
            f = safe_call(perturbed)
            if f is not None and np.isfinite(f):
                other_starts.append(perturbed)
        
        for start_pt in other_starts:
            if calls_made >= max_calls or time.time() > deadline:
                break
            result = minimize(
                lambda x: func(np.clip(x, -5, 5)),
                start_pt.copy(),
                method='BFGS',
                options={'maxiter': 30, 'maxfun': 20}
            )
            if result.success and np.isfinite(result.fun):
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5).copy()
        
        # Phase 3: If we still have budget, do a second round of LHS + BFGS
        if calls_made < max_calls * 0.7 and time.time() < deadline - 30:
            # Generate a second set of LHS points, more concentrated
            for _ in range(10):
                if calls_made >= max_calls or time.time() > deadline:
                    break
                # Concentrate around the best region
                center = best_x
                spread = 2.0
                pt = center + rng.normal(0, spread, n)
                pt = np.clip(pt, -5, 5)
                f = safe_call(pt)
                if f is not None and np.isfinite(f):
                    # Quick BFGS from this point if it's promising
                    if f < best_f + 1.0 and calls_made < max_calls and time.time() < deadline:
                        result = minimize(
                            lambda x: func(np.clip(x, -5, 5)),
                            pt.copy(),
                            method='BFGS',
                            options={'maxiter': 20, 'maxfun': 15}
                        )
                        if result.success and np.isfinite(result.fun):
                            if result.fun < best_f:
                                best_f = result.fun
                                best_x = np.clip(result.x, -5, 5).copy()
        
        # Phase 4: Final polish with Nelder-Mead if budget allows
        if calls_made < max_calls and time.time() < deadline - 10:
            result = minimize(
                lambda x: func(np.clip(x, -5, 5)),
                best_x.copy(),
                method='Nelder-Mead',
                options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if np.isfinite(result.fun) and result.fun < best_f:
                best_f = result.fun
                best_x = np.clip(result.x, -5, 5).copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
