import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.func = None
        self.best_x = None
        self.best_f = float('inf')
        self.start_time = None
        self.deadline = None

    def _evaluate(self, x):
        if self.calls >= self.budget or time.time() > self.deadline:
            return 1e10
        self.calls += 1
        val = self.func(x)
        if val < self.best_f:
            self.best_f = val
            self.best_x = np.array(x, dtype=float)
        return val

    def __call__(self, func):
        self.func = func
        self.calls = 0
        self.best_f = float('inf')
        self.best_x = None
        self.start_time = time.time()
        self.deadline = self.start_time + 290.0
        
        d = self.dim
        rng = np.random.default_rng(42)

        def wrapper_func(x):
            if self.calls >= self.budget or time.time() > self.deadline - 5:
                return 1e10
            return self._evaluate(x)

        # Phase 1: Initial random sampling to find good starting points
        # Bent Cigar has a narrow valley, but random sampling helps ensure we are in the basin.
        n_init = 10
        init_points = []
        for _ in range(n_init):
            if self.calls >= self.budget - 50:
                break
            x = rng.uniform(-5, 5, size=d)
            f = wrapper_func(x)
            if f < 1e9:
                init_points.append((f, x))

        if not init_points:
            x0 = np.zeros(d)
            f0 = wrapper_func(x0)
            init_points = [(f0, x0)]

        # Sort by function value
        init_points.sort(key=lambda item: item[0])
        best_start = init_points[0][1]

        # Phase 2: L-BFGS-B from best initial point
        # L-BFGS-B is efficient for convex-like problems but can struggle with extreme ill-conditioning.
        # We use a moderate maxiter to avoid wasting budget.
        try:
            res = minimize(
                wrapper_func,
                best_start,
                method='L-BFGS-B',
                bounds=[(-5, 5)] * d,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
        except Exception:
            pass

        # Phase 3: Powell's Method
        # Powell's method is a derivative-free optimization algorithm that uses a set of directions
        # to search for the minimum. It adapts these directions to align with the narrow valley
        # of the Bent Cigar function, making it much more effective than Nelder-Mead in
        # ill-conditioned regimes.
        # We run it from the current best point found by L-BFGS-B.
        
        if self.calls < self.budget - 10 and time.time() < self.deadline - 10:
            try:
                res_powell = minimize(
                    wrapper_func,
                    self.best_x if self.best_x is not None else best_start,
                    method='Powell',
                    bounds=[(-5, 5)] * d,
                    options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
            except Exception:
                pass

        # Phase 4: Final L-BFGS-B Polish
        # Powell's method might not converge to machine precision.
        # L-BFGS-B can take over for a final polish if budget remains.
        if self.calls < self.budget - 5 and time.time() < self.deadline - 5:
            try:
                res_final = minimize(
                    wrapper_func,
                    self.best_x if self.best_x is not None else best_start,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * d,
                    options={'maxiter': 50, 'ftol': 1e-15, 'gtol': 1e-10}
                )
            except Exception:
                pass

        if self.best_x is None:
            self.best_x = np.zeros(d)
            self.best_f = 0.0

        return self.best_x, self.best_f
