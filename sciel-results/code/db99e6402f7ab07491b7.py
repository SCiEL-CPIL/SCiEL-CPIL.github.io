import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = None

    def __call__(self, func):
        self.deadline = time.time() + 290
        self.calls = 0
        self.best_x = np.zeros(self.dim)
        self.best_f = np.inf

        def wrapped_func(x):
            if time.time() > self.deadline:
                raise TimeoutError
            x = np.clip(x, -5, 5)
            self.calls += 1
            f = func(x)
            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()
            return f

        n = self.dim
        rng = np.random.default_rng(42)

        # Generate diverse starting points
        # Mix of structured and random starts
        x0_list = []
        # Origin
        x0_list.append(np.zeros(n))
        # Small perturbations around origin
        for _ in range(2):
            x0_list.append(rng.uniform(-1, 1, n))
        # Medium radius
        for _ in range(2):
            x0_list.append(rng.uniform(-3, 3, n))
        # Larger radius
        for _ in range(2):
            x0_list.append(rng.uniform(-4.5, 4.5, n))
        # Axis-aligned points
        for i in range(min(n, 3)):
            pt = np.zeros(n)
            pt[i] = rng.choice([-4, 4])
            x0_list.append(pt)

        # Adaptive budget allocation: give more budget to promising starts
        # Start with equal allocation, then adjust
        n_starts = len(x0_list)
        base_budget = self.budget // n_starts
        # Reserve some budget for final polish
        polish_budget = min(50, self.budget // 10)
        main_budget = self.budget - polish_budget
        per_start_budget = max(10, main_budget // n_starts)

        # Sort starts by heuristic: prefer those closer to origin first
        # (Griewank-Rosenbrock often has minimum near center)
        x0_list.sort(key=lambda x: np.linalg.norm(x))

        for i, x0 in enumerate(x0_list):
            if self.calls >= self.budget - polish_budget:
                break
            if time.time() > self.deadline - 10:
                break

            remaining = self.budget - self.calls - polish_budget
            if remaining <= 0:
                break

            # Adaptive: give less budget to later starts if we're making progress
            # and more to early starts
            alloc = min(per_start_budget, remaining)
            # Roughly 8-10 function calls per Nelder-Mead iteration for n=5
            maxiter = max(1, alloc // 9)

            try:
                res = minimize(wrapped_func, x0, method='Nelder-Mead',
                               options={
                                   'maxiter': maxiter,
                                   'xatol': 1e-6,
                                   'fatol': 1e-10,
                                   'initial_simplex': None  # Use default
                               })
            except TimeoutError:
                break
            except Exception:
                continue

            # Update best
            if self.best_f > res.fun:
                self.best_f = res.fun
                self.best_x = res.x.copy()

        # Final polish: run a tighter Nelder-Mead from the best point
        if self.calls < self.budget and time.time() < self.deadline - 5:
            remaining = self.budget - self.calls
            if remaining > 10:
                try:
                    maxiter = max(1, remaining // 9)
                    res = minimize(wrapped_func, self.best_x.copy(), method='Nelder-Mead',
                                   options={
                                       'maxiter': maxiter,
                                       'xatol': 1e-8,
                                       'fatol': 1e-12
                                   })
                    if self.best_f > res.fun:
                        self.best_f = res.fun
                        self.best_x = res.x.copy()
                except Exception:
                    pass

        return self.best_x, self.best_f
