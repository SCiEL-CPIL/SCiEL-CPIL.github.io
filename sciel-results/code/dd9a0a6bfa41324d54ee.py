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
        # DE uses a population-based approach that works well for multimodal problems
        # Budget allocation: use ~60% of budget for DE, rest for local refinement
        de_budget = int(self.budget * 0.6)
        
        try:
            res_de = differential_evolution(
                wrapper,
                bounds=[(-5, 5)] * self.dim,
                maxiter=de_budget // 50,  # Approximate iterations based on population size
                popsize=15,  # Population size multiplier
                tol=1e-10,
                seed=42,
                polish=False,  # We'll do our own polishing
                init='sobol'  # Better initial distribution
            )
        except Exception:
            pass

        # Phase 2: Local refinement with L-BFGS-B from best point
        if self.best_x is not None and self.calls < self.budget:
            try:
                res_lbfgs = minimize(
                    wrapper,
                    self.best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-8}
                )
            except Exception:
                pass

        # Phase 3: Additional local refinement with Nelder-Mead for precision
        if self.best_x is not None and self.calls < self.budget:
            try:
                res_nm = minimize(
                    wrapper,
                    self.best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
            except Exception:
                pass

        return self.best_x, self.best_f
