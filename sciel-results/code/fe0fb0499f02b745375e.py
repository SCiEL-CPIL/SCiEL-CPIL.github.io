"""Multi-start adaptive (1+1)-ES with restarts and deadline guard for BBOB Weierstrass f16."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-start adaptive (1+1)-ES with restarts for the BBOB Weierstrass (f16) task."""

    def __init__(self, budget, dim):
        self.budget, self.dim = int(budget), int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng()
        deadline = time.time() + 280.0

        n_calls = [0]

        def safe_func(x):
            if n_calls[0] >= budget or time.time() > deadline:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            n_calls[0] += 1
            return float(func(x))

        best_x = np.zeros(dim)
        best_f = np.inf

        # Phase 1: Initial Exploration with LHS
        n_initial = 15
        if n_initial > budget:
            n_initial = max(1, budget // 2)
        
        if n_initial > 0:
            perms = [rng.permutation(n_initial) for _ in range(dim)]
            for i in range(n_initial):
                point = np.zeros(dim)
                for d in range(dim):
                    frac = (perms[d][i] + rng.random()) / n_initial
                    point[d] = -5.0 + 10.0 * frac
                if n_calls[0] < budget and time.time() < deadline:
                    fx = safe_func(point)
                    if fx < best_f:
                        best_f = fx
                        best_x = point.copy()
        
        # Add center point if budget allows
        if n_calls[0] < budget and time.time() < deadline:
            fx = safe_func(np.zeros(dim))
            if fx < best_f:
                best_f = fx
                best_x = np.zeros(dim)

        # Phase 2: Main (1+1)-ES
        # Allocate ~70% of remaining budget to this phase
        budget_phase2 = int((budget - n_calls[0]) * 0.7)
        if budget_phase2 < 10:
            budget_phase2 = max(10, budget - n_calls[0])
            
        sigma = 1.0
        no_improve = 0
        max_no_improve = 30
        
        while n_calls[0] < n_calls[0] + budget_phase2 and time.time() < deadline:
            candidate = best_x + sigma * rng.standard_normal(dim)
            candidate = np.clip(candidate, -5.0, 5.0)
            fx = safe_func(candidate)

            if fx < best_f:
                best_x = candidate.copy()
                best_f = fx
                no_improve = 0
                sigma *= 0.95
            else:
                no_improve += 1
                sigma *= 1.05

            sigma = max(sigma, 1e-4)
            sigma = min(sigma, 5.0)
            
            if no_improve > max_no_improve:
                break

        # Phase 3: Polish (High Precision)
        # Use remaining budget for fine-grained search
        # Smaller sigma, more aggressive reduction
        sigma = max(sigma * 0.5, 0.1)
        max_no_improve = 10
        
        while n_calls[0] < budget and time.time() < deadline:
            candidate = best_x + sigma * rng.standard_normal(dim)
            candidate = np.clip(candidate, -5.0, 5.0)
            fx = safe_func(candidate)

            if fx < best_f:
                best_x = candidate.copy()
                best_f = fx
                no_improve = 0
                sigma *= 0.9
            else:
                no_improve += 1
                sigma *= 1.1

            sigma = max(sigma, 1e-5)
            sigma = min(sigma, 5.0)

            if no_improve > max_no_improve:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
