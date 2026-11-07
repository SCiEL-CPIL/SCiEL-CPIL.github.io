import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = None

    def __call__(self, func):
        self.deadline = time.time() + 285  # 15s margin before 300s kill
        self.calls = 0
        self.best_x = np.zeros(self.dim)
        self.best_f = np.inf

        def wrapped_func(x):
            if self.calls >= self.budget:
                return np.inf
            if time.time() > self.deadline:
                return np.inf
            x = np.clip(x, -5, 5)
            self.calls += 1
            f = float(func(x))
            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()
            return f

        # Strategy: Multi-start CMA-ES-like with adaptive step size
        # For dim=5, CMA-ES is effective. We'll use a simple population-based approach.

        dim = self.dim
        budget = self.budget

        # Phase 1: Exploration with multiple random starts and local refinement
        # We'll use a combination of global exploration and local exploitation

        # Generate diverse starting points
        np.random.seed(42)  # Fixed seed for reproducibility
        starts = []
        # Origin
        starts.append(np.zeros(dim))
        # Random points in [-4, 4]
        for _ in range(5):
            starts.append(np.random.uniform(-4, 4, dim))
        # Points near boundaries
        for _ in range(2):
            starts.append(np.random.uniform(-5, -4, dim))
            starts.append(np.random.uniform(4, 5, dim))

        # For each start, do a local search using coordinate descent + Nelder-Mead
        # Budget allocation: split remaining budget among starts

        num_starts = min(len(starts), 8)
        calls_per_start = max(10, (budget - 10) // num_starts)

        for i in range(num_starts):
            if self.calls >= budget - 5:
                break
            if time.time() > self.deadline - 5:
                break

            x0 = starts[i].copy()
            f0 = wrapped_func(x0)

            if not np.isfinite(f0):
                continue

            # Coordinate descent with adaptive step size
            step_size = 0.5
            improvement = True
            coord_iter = 0
            max_coord_iter = calls_per_start // 3

            while improvement and coord_iter < max_coord_iter and self.calls < budget - 5 and time.time() < self.deadline - 5:
                improvement = False
                for d in range(dim):
                    if self.calls >= budget - 5 or time.time() > self.deadline - 5:
                        break
                    for direction in [1, -1]:
                        trial = x0.copy()
                        trial[d] = x0[d] + direction * step_size
                        trial = np.clip(trial, -5, 5)
                        ft = wrapped_func(trial)
                        if ft < f0:
                            f0 = ft
                            x0 = trial.copy()
                            improvement = True
                            break
                    # Adaptive step size
                    if not improvement:
                        step_size *= 0.7
                    else:
                        step_size = min(step_size * 1.2, 2.0)
                coord_iter += 1
                if step_size < 1e-4:
                    break

            # Refine with Nelder-Mead
            if self.calls < budget - 10 and time.time() < self.deadline - 5:
                try:
                    from scipy.optimize import minimize
                    remaining = min(calls_per_start, budget - self.calls - 5)
                    if remaining > 10:
                        maxiter = max(1, remaining // 8)
                        res = minimize(wrapped_func, x0, method='Nelder-Mead',
                                       options={'maxiter': maxiter, 'xatol': 1e-6, 'fatol': 1e-10})
                        if np.isfinite(res.fun) and res.fun < self.best_f:
                            self.best_f = res.fun
                            self.best_x = np.clip(res.x, -5, 5).copy()
                except Exception:
                    pass

            if self.calls < budget - 10 and time.time() < self.deadline - 5:
                # Quick local search around best point found so far
                best_local = self.best_x.copy()
                for d in range(dim):
                    if self.calls >= budget - 5 or time.time() > self.deadline - 5:
                        break
                    for direction in [1, -1]:
                        trial = best_local.copy()
                        trial[d] = best_local[d] + direction * 0.1
                        trial = np.clip(trial, -5, 5)
                        ft = wrapped_func(trial)
                        if ft < self.best_f:
                            self.best_f = ft
                            self.best_x = trial.copy()
                            best_local = trial.copy()

        # Final polish: if we have budget left, do a final local search
        if self.calls < budget - 3 and time.time() < self.deadline - 3:
            try:
                from scipy.optimize import minimize
                remaining = min(30, budget - self.calls - 2)
                if remaining > 5:
                    maxiter = max(1, remaining // 8)
                    res = minimize(wrapped_func, self.best_x, method='Nelder-Mead',
                                   options={'maxiter': maxiter, 'xatol': 1e-7, 'fatol': 1e-12})
                    if np.isfinite(res.fun) and res.fun < self.best_f:
                        self.best_f = res.fun
                        self.best_x = np.clip(res.x, -5, 5).copy()
            except Exception:
                pass

        return self.best_x, self.best_f
