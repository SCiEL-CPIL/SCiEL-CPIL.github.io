"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead with deterministic initialization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # safety margin before 330s kill
        n = self.dim
        remaining = self.budget
        best_x = None
        best_f = float('inf')

        # Deterministic initial points: corners + center + random
        init_points = []
        # Add center
        init_points.append(np.zeros(n))
        # Add some structured points
        for i in range(n):
            p = np.zeros(n)
            p[i] = 0.0
            init_points.append(p)
        # Add a few random deterministic points
        rng = np.random.default_rng(42)
        for _ in range(10):
            init_points.append(rng.uniform(-4, 4, n))
        
        # Deduplicate
        unique_points = []
        for p in init_points:
            is_dup = False
            for q in unique_points:
                if np.allclose(p, q, atol=1e-10):
                    is_dup = True
                    break
            if not is_dup:
                unique_points.append(p)
        
        # Phase 1: Evaluate initial points
        for x0 in unique_points:
            if remaining <= 0 or time.time() > deadline - 5.0:
                break
            fx0 = func(x0)
            remaining -= 1
            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()
        
        # Phase 2: Local optimization from best starts using Nelder-Mead
        # Use top few starting points for local refinement
        start_candidates = []
        for x0 in unique_points:
            if remaining <= 0 or time.time() > deadline - 10.0:
                break
            fx0 = func(x0)
            remaining -= 1
            start_candidates.append((fx0, x0))
        
        # Sort by function value and pick top candidates for local optimization
        start_candidates.sort(key=lambda t: t[0])
        top_starts = [x for _, x in start_candidates[:min(5, len(start_candidates))]]
        
        for x0 in top_starts:
            if remaining < 10 or time.time() > deadline - 10.0:
                break
            
            # Use Nelder-Mead with limited iterations
            max_iter = min(remaining // (n + 1), 100)
            if max_iter < 5:
                continue
                
            try:
                result = minimize(
                    func, x0, method='Nelder-Mead',
                    options={'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-4,
                            'initial_simplex': None}
                )
                # Count evaluations used
                evals_used = result.nfev
                remaining -= evals_used
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                # If optimization fails, fall back to simple local search
                x = x0.copy()
                fx = func(x)
                remaining -= 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                
                # Simple coordinate descent
                step = 0.5
                for _ in range(min(remaining, 20)):
                    if time.time() > deadline - 5.0:
                        break
                    improved = False
                    for i in range(n):
                        if remaining <= 0:
                            break
                        for direction in [1, -1]:
                            y = x.copy()
                            y[i] = np.clip(y[i] + direction * step, -5, 5)
                            fy = func(y)
                            remaining -= 1
                            if fy < fx:
                                x, fx = y, fy
                                improved = True
                                if fx < best_f:
                                    best_f = fx
                                    best_x = x.copy()
                    if not improved:
                        step *= 0.5
                        if step < 1e-3:
                            break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
