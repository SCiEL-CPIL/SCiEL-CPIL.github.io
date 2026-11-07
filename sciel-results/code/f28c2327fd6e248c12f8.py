# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B optimizer for BBOB f7 (Step ellipsoidal) on [-5,5]^d."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0
        self._best_x = None
        self._best_f = np.inf

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        # Reset state
        self._call_count = 0
        self._best_x = None
        self._best_f = np.inf

        dim = self.dim
        budget = self.budget

        def count_eval(x):
            self._call_count += 1
            val = func(x)
            if val < self._best_f:
                self._best_f = val
                self._best_x = np.array(x, dtype=float).copy()
            return val

        def wrapped_func(x):
            return count_eval(x)

        # Generate diverse starting points
        # Strategy: mix of uniform, structured grid, and boundary points
        starts = []

        # 1. Center point
        starts.append(np.zeros(dim))

        # 2. Uniform random points
        n_uniform = 5
        for _ in range(n_uniform):
            starts.append(rng.uniform(-4, 4, dim))

        # 3. Structured points: corners and midpoints of the box
        # Try points that emphasize different coordinates
        for i in range(dim):
            # Point with large value in coordinate i
            pt = rng.uniform(-4, 4, dim)
            pt[i] = rng.choice([-3, 3])
            starts.append(pt)

        # 4. Small perturbations around origin (in case minimum is near center)
        for scale in [0.5, 1.0, 2.0]:
            starts.append(rng.standard_normal(dim) * scale)

        # 5. Additional diverse points
        for _ in range(3):
            starts.append(rng.uniform(-5, 5, dim))

        # Deduplicate and limit
        unique_starts = []
        seen = set()
        for s in starts:
            key = tuple(np.round(s, 3))
            if key not in seen:
                seen.add(key)
                unique_starts.append(np.clip(s, -5, 5))

        # Sort by heuristic: prefer points closer to center (f7 minimum often near shifted center)
        # But keep diversity, so just use as-is

        # Allocate budget: give more evaluations to early starts, less to later
        # Use a decay schedule
        n_starts = len(unique_starts)
        evals_per_start = []
        remaining = budget
        for i in range(n_starts):
            # Exponential decay: first start gets most, later ones get less
            frac = max(0.05, 0.5 ** (i / 3.0))
            alloc = int(remaining * frac / (1 - 0.5 ** ((n_starts - i - 1) / 3.0)) + 0.5)
            alloc = max(10, min(alloc, remaining - 10 * (n_starts - i - 1)))
            evals_per_start.append(alloc)
            remaining -= alloc

        # Ensure we don't exceed budget
        total_alloc = sum(evals_per_start)
        if total_alloc > budget:
            # Scale down
            factor = budget / total_alloc
            evals_per_start = [max(5, int(e * factor)) for e in evals_per_start]

        # Run L-BFGS-B from each start
        best_x = np.zeros(dim)
        best_f = np.inf
        best_idx = 0

        for idx, start_pt in enumerate(unique_starts):
            if self._call_count >= budget or time.time() >= deadline:
                break

            # Skip if this start is too close to an already-evaluated good point
            if best_f < np.inf:
                if np.linalg.norm(start_pt - best_x) < 0.1:
                    continue

            maxiter = evals_per_start[idx]
            # L-BFGS-B counts each function call
            # Use a reasonable maxiter

            try:
                result = minimize(
                    wrapped_func,
                    start_pt,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={
                        'maxiter': maxiter,
                        'maxfun': maxiter,
                        'ftol': 1e-12,
                        'gtol': 1e-8,
                        'disp': False,
                    }
                )

                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
                    best_idx = idx

            except Exception:
                pass

            # Update global best tracking
            if self._best_f < best_f:
                best_f = self._best_f
                best_x = self._best_x.copy()

            # Early termination: if we've found a very good solution,
            # spend remaining budget on local refinement
            if best_f < 1.0 and self._call_count < budget - 50:
                # Do a few more local refinements
                break

        # Local refinement around the best solution found
        if self._call_count < budget - 10 and time.time() < deadline:
            x_local = best_x.copy()
            f_local = best_f

            # Multi-scale coordinate descent
            step_sizes = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]

            for step in step_sizes:
                if self._call_count >= budget - 5 or time.time() >= deadline:
                    break
                improved = True
                while improved and self._call_count < budget - 5 and time.time() < deadline:
                    improved = False
                    for d in range(dim):
                        if self._call_count >= budget - 5 or time.time() >= deadline:
                            break
                        # Try positive step
                        x_try = x_local.copy()
                        x_try[d] = min(5.0, x_try[d] + step)
                        f_try = count_eval(x_try)
                        if f_try < f_local - 1e-12:
                            x_local = x_try
                            f_local = f_try
                            improved = True
                        # Try negative step
                        if self._call_count >= budget - 5 or time.time() >= deadline:
                            break
                        x_try = x_local.copy()
                        x_try[d] = max(-5.0, x_try[d] - step)
                        f_try = count_eval(x_try)
                        if f_try < f_local - 1e-12:
                            x_local = x_try
                            f_local = f_try
                            improved = True

            # Update best if improved
            if f_local < best_f:
                best_f = f_local
                best_x = x_local.copy()

        # Final update
        if self._best_f < best_f:
            best_f = self._best_f
            best_x = self._best_x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
