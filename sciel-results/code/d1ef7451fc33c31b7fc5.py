"""Rewritten program with Multi-start Nelder-Mead optimizer."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead optimizer with Halton initialization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.func = func
        self.calls = 0
        self.start_time = time.time()
        self.best_x = None
        self.best_f = float('inf')

        # Wrapper to count calls and enforce time/budget limits
        def eval_func(x):
            if self.calls >= self.budget:
                return float('inf')
            if time.time() - self.start_time > 280:
                return float('inf')
            
            self.calls += 1
            x = np.clip(x, -5, 5)
            try:
                val = func(x)
            except Exception:
                val = float('inf')
            
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            return val

        # 1. Global Search: Halton Sequence
        K = 10
        halton_points = self._halton_sequence(K, self.dim)
        
        initial_points = []
        for p in halton_points:
            f = eval_func(p)
            initial_points.append((f, p))
        
        # Sort by fitness
        initial_points.sort(key=lambda x: x[0])
        
        # 2. Local Search: Nelder-Mead restarts
        max_restarts = 4
        for i in range(min(max_restarts, len(initial_points))):
            if self.calls >= self.budget:
                break
            if time.time() - self.start_time > 280:
                break
                
            start_x = initial_points[i][1]
            self._nelder_mead(start_x, eval_func)
            
        return self.best_x, self.best_f

    def _halton_sequence(self, num_points, dim):
        points = np.zeros((num_points, dim))
        for d in range(dim):
            base = d + 2  # 2, 3, 5, 7, 11...
            for n in range(num_points):
                f = 1.0
                i = n + 1
                while i > 0:
                    f /= base
                    i = int(i / base)
                # Map (0, 1] to [-5, 5]
                points[n, d] = 10.0 * f - 5.0
        return points

    def _nelder_mead(self, x0, eval_func):
        dim = self.dim
        # Initialize simplex
        delta = 0.1
        simplex = np.zeros((dim + 1, dim))
        simplex[0] = x0
        for i in range(1, dim + 1):
            simplex[i] = x0
            simplex[i][i - 1] += delta

        # Evaluate initial simplex
        fvals = np.zeros(dim + 1)
        for i in range(dim + 1):
            fvals[i] = eval_func(simplex[i])

        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5

        max_iter = 60  # Limit iterations to conserve budget

        for _ in range(max_iter):
            if self.calls >= self.budget:
                break
            if time.time() - self.start_time > 280:
                break

            # Sort simplex
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]

            # Convergence check
            if np.max(fvals) - np.min(fvals) < 1e-8:
                break
            if np.max(np.abs(simplex - simplex[0])) < 1e-6:
                break

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = eval_func(xr)

            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                fe = eval_func(xe)
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                # Reflection is good
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                if fr < fvals[-1]:
                    # External contraction
                    xc = centroid + rho * (xr - centroid)
                    fc = eval_func(xc)
                    if fc <= fr:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fvals[i] = eval_func(simplex[i])
                else:
                    # Internal contraction
                    xc = centroid - rho * (centroid - simplex[-1])
                    fc = eval_func(xc)
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fvals[i] = eval_func(simplex[i])

        return self.best_x, self.best_f
# EVOLVE-BLOCK-END
