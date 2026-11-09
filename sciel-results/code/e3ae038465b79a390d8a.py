import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.max_time = 290.0  # Safety margin from 330s limit

    def __call__(self, func):
        self.start_time = time.time()
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf

        def wrapper(x):
            if self.calls >= self.budget:
                return 1e100
            if time.time() - self.start_time > self.max_time:
                return 1e100
            val = func(x)
            self.calls += 1
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            return val

        # Phase 1: Global search with Differential Evolution
        # DE maintains a population and adapts - good for shifted/rotated landscape
        # Limit maxiter to control function call count
        # With popsize=10, each generation uses ~10 evaluations
        # maxiter=20 -> ~200 evaluations for DE
        de_budget_calls = 150
        max_iter_de = 15  # ~15 * 10 = 150 calls roughly

        try:
            de_result = differential_evolution(
                wrapper,
                bounds=[(-5, 5)] * self.dim,
                maxiter=max_iter_de,
                popsize=10,
                tol=1e-6,
                atol=1e-8,
                seed=42,
                init='sobol',
                polish=False,  # We'll do our own polish
                updating='deferred'
            )
        except Exception:
            de_result = None

        # Collect good starting points for local polish
        start_points = []

        # Add DE result if available
        if de_result is not None and hasattr(de_result, 'x') and de_result.x is not None:
            start_points.append(de_result.x.copy())

        # Add current best
        if self.best_x is not None:
            start_points.append(self.best_x.copy())

        # Add center and a few deterministic points as fallback
        start_points.append(np.zeros(self.dim))
        rng = np.random.default_rng(123)
        for _ in range(3):
            start_points.append(rng.uniform(-4, 4, self.dim))

        # Deduplicate and limit start points
        unique_starts = []
        for p in start_points:
            p = np.clip(p, -5, 5)
            dup = False
            for q in unique_starts:
                if np.linalg.norm(p - q) < 1e-6:
                    dup = True
                    break
            if not dup:
                unique_starts.append(p)

        # Phase 2: Local polish with L-BFGS-B from multiple starts
        # Sort by current best value at each start point (evaluate if needed)
        start_vals = []
        for p in unique_starts:
            if self.calls >= self.budget:
                break
            if time.time() - self.start_time > self.max_time:
                break
            f = wrapper(p)
            start_vals.append((f, p))

        start_vals.sort(key=lambda x: x[0])

        # Polish top starts with L-BFGS-B
        for f_init, x_init in start_vals[:5]:
            if self.calls >= self.budget:
                break
            if time.time() - self.start_time > self.max_time:
                break

            try:
                res = minimize(
                    wrapper,
                    x_init,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 30, 'ftol': 1e-14, 'gtol': 1e-8}
                )
                if res.success:
                    # Check if we should continue or break
                    if self.best_f < 1e-6:
                        break
            except Exception:
                pass

        # Phase 3: Final high-precision polish if budget remains
        if self.calls < self.budget and self.best_x is not None:
            if time.time() - self.start_time < self.max_time:
                try:
                    res = minimize(
                        wrapper,
                        self.best_x,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * self.dim,
                        options={'maxiter': 20, 'ftol': 1e-16, 'gtol': 1e-10}
                    )
                except Exception:
                    pass

        return self.best_x, self.best_f
