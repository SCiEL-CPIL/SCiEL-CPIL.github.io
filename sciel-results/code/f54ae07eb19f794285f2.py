"""Multi-start Powell's method with dense stratified LHS for Lunacek bi-Rastrigin f24."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Powell with dense stratified Latin Hypercube initialization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._calls = 0
        self._deadline = None

    def __call__(self, func):
        self._calls = 0
        self._deadline = time.time() + 280.0  # safety margin under 300s

        dim = self.dim
        budget = self.budget
        calls = [0]

        def f(x):
            if calls[0] >= budget:
                return 1e12
            if time.time() > self._deadline:
                return 1e12
            calls[0] += 1
            val = func(np.asarray(x, dtype=float))
            return float(val)

        best_x = None
        best_f = np.inf

        rng = np.random.default_rng(42)

        def cma_es(x0, sigma0, maxfev, deadline):
            """Run CMA-ES from x0 with initial step size sigma0."""
            from cma import CMAEvolutionStrategy
            opts = {
                'bounds': [(-5, 5)] * dim,
                'maxfev': maxfev,
                'seed': 42,
                'verbose': -9,
                'tolopt': 1e-8,
                'tolfun': 1e-10,
            }
            es = CMAEvolutionStrategy(x0, sigma0, opts)
            while not es.stop() and calls[0] < budget and time.time() < deadline:
                xs = es.ask()
                fs = [f(x) for x in xs]
                es.tell(xs, fs)
            return es.result

        # Phase 1: Initial diverse evaluation to find promising regions
        # Use stratified sampling for broad coverage
        n_init = 20
        init_points = []
        for i in range(n_init):
            point = np.zeros(dim)
            for d in range(dim):
                frac = (i + rng.random()) / n_init
                point[d] = -5.0 + frac * 10.0
            init_points.append(point)
        
        # Add some strategic points
        strategic = [
            np.zeros(dim),
            np.full(dim, 2.0),
            np.full(dim, -2.0),
            np.full(dim, 4.0),
            np.full(dim, -4.0),
        ]
        init_points.extend(strategic)

        for x0 in init_points:
            if calls[0] >= budget or time.time() > self._deadline:
                break
            fx0 = f(x0)
            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()

        # Phase 2: CMA-ES with adaptive restarts
        # Start with moderate sigma, restart if stagnation detected
        remaining = budget - calls[0]
        if best_x is not None and remaining > 30 and time.time() < self._deadline:
            # First CMA-ES run from best point
            sigma0 = 1.5
            maxfev1 = min(150, remaining)
            try:
                res = cma_es(best_x, sigma0, maxfev1, self._deadline)
                if res.fval < best_f:
                    best_f = res.fval
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 3: Additional CMA-ES restarts from diverse points if budget allows
        remaining = budget - calls[0]
        if remaining > 40 and time.time() < self._deadline:
            # Try from a few other promising initial points
            other_starts = [
                np.full(dim, 2.5),
                np.full(dim, -2.5),
                np.full(dim, 0.5),
                np.full(dim, -0.5),
            ]
            for x0 in other_starts:
                if calls[0] >= budget or time.time() > self._deadline:
                    break
                small_budget = min(40, budget - calls[0])
                if small_budget < 10:
                    break
                try:
                    res = cma_es(x0, 1.0, small_budget, self._deadline)
                    if res.fval < best_f:
                        best_f = res.fval
                        best_x = res.x.copy()
                except Exception:
                    pass

        # Phase 4: L-BFGS-B polish
        remaining = budget - calls[0]
        if remaining > 10 and time.time() < self._deadline and best_x is not None:
            try:
                res = minimize(
                    f,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 30, 'maxfun': min(30, remaining), 'ftol': 1e-12}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 5: Final local refinement with Nelder-Mead if budget remains
        remaining = budget - calls[0]
        if remaining > 15 and time.time() < self._deadline and best_x is not None:
            try:
                res = minimize(
                    f,
                    best_x,
                    method='Nelder-Mead',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 50, 'maxfev': min(30, remaining), 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
