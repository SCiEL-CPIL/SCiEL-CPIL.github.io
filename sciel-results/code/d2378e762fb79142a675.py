"""Optimizer for BBOB f24 (Lunacek bi-Rastrigin) using Multi-start Nelder-Mead."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead for bi-modal optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, func, x0, budget, rng, deadline):
        """Run Nelder-Mead optimization."""
        n = self.dim
        x = np.array(x0, dtype=float)
        simplex = [x.copy() for _ in range(n + 1)]
        for i in range(n):
            simplex[i + 1][i] += 0.5

        fvals = []
        calls = 0
        for xi in simplex:
            fi = func(xi)
            fvals.append(fi)
            calls += 1
            if time.time() > deadline or calls >= budget:
                break

        best_x = simplex[0].copy()
        best_f = min(fvals)

        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Estimate max iterations based on remaining budget
        # Each iteration costs 1-4 calls. Assume avg 2.
        max_iter = (budget - calls) // 2
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            if calls >= budget:
                break

            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = simplex[0].copy()

            x_c = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = x_c + alpha * (x_c - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls += 1
            
            if calls >= budget:
                break

            if f_r < fvals[0]:
                # Expansion
                x_e = x_c + gamma * (x_r - x_c)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls += 1
                if calls >= budget:
                    break
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                # Contraction
                x_c2 = x_c + rho * (simplex[-1] - x_c)
                x_c2 = np.clip(x_c2, -5, 5)
                f_c2 = func(x_c2)
                calls += 1
                if calls >= budget:
                    break
                if f_c2 < fvals[-1]:
                    simplex[-1] = x_c2
                    fvals[-1] = f_c2
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = func(simplex[i])
                        calls += 1
                        if calls >= budget:
                            break
                    if calls >= budget:
                        break

        return best_x, best_f, calls

    def __call__(self, func):
        rng = np.random.default_rng(42)
        deadline = time.time() + 280
        total_calls = 0
        best_x = None
        best_f = np.inf

        # Phase 1: Random Sampling
        n_init = min(50, self.budget // 10)
        init_points = []
        for i in range(n_init):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            total_calls += 1
            init_points.append((fx, x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        init_points.sort(key=lambda t: t[0])
        
        # Phase 2: Nelder-Mead on best starting points
        remaining = self.budget - total_calls
        n_restarts = min(5, max(1, remaining // 50))
        budget_per_restart = remaining // n_restarts if n_restarts > 0 else 0

        for i in range(n_restarts):
            if time.time() > deadline:
                break
            if total_calls >= self.budget:
                break
            
            if i < len(init_points):
                x0 = init_points[i][1].copy()
            else:
                x0 = best_x.copy()

            x_opt, f_opt, calls_used = self._nelder_mead(
                func, x0, budget_per_restart, rng, deadline)
            
            total_calls += calls_used
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
