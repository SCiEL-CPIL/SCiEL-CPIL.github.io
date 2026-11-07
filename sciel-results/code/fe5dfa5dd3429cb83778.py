import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Hybrid optimizer: Sobol exploration + (1+1)-ES exploitation + Rastrigin-aware refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # safety margin before 330s kill

        # Track best
        best_x = None
        best_f = float('inf')
        calls = 0

        def eval_func(x):
            nonlocal best_x, best_f, calls
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Sobol quasi-random exploration for uniform basin coverage
        # Use Sobol sequence to systematically cover [-5, 5]^5
        sobol = qmc.Sobol(d=dim, seed=42, scramble=True)
        # Generate points in [0, 1]^5, then map to [-5, 5]^5
        n_sobol = min(50, budget)
        sobol_points = sobol.random(n_sobol) * 10 - 5  # map to [-5, 5]^5

        # Also add the origin as a special point
        sobol_points = np.vstack([np.zeros(dim), sobol_points])
        n_sobol = len(sobol_points)

        # Evaluate Sobol points
        for i in range(n_sobol):
            if calls >= budget or time.time() > deadline:
                break
            eval_func(sobol_points[i])

        # Phase 2: (1+1)-ES exploitation from the best point found
        if best_x is not None and calls < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma_es = 0.5  # start with moderate step for (1+1)-ES

            # Success rate tracking
            window = 15
            recent = [False] * window
            idx = 0

            # Use remaining budget for (1+1)-ES
            max_es_calls = min(300, budget - calls)
            for _ in range(max_es_calls):
                if calls >= budget or time.time() > deadline:
                    break

                y = np.clip(x + sigma_es * rng.standard_normal(dim), -5, 5)
                fy = eval_func(y)

                if fy < fx:
                    x, fx = y, fy
                    recent[idx % window] = True
                else:
                    recent[idx % window] = False
                idx += 1

                # Adapt sigma based on success rate
                if idx >= window:
                    sr = sum(recent) / window
                    if sr > 0.4:
                        sigma_es *= 1.1
                    elif sr < 0.2:
                        sigma_es *= 0.9
                    else:
                        sigma_es *= 1.0

                sigma_es = np.clip(sigma_es, 0.01, 2.0)

        # Phase 3: Rastrigin-aware refinement
        if best_x is not None and calls < budget and time.time() < deadline:
            x_local = best_x.copy()
            f_local = best_f

            # Strategy 1: Coordinate-wise refinement with decreasing step
            step = 0.2
            for _ in range(30):
                if calls >= budget or time.time() > deadline:
                    break
                improved = False
                for d in range(dim):
                    if calls >= budget or time.time() > deadline:
                        break
                    for sign in [1, -1]:
                        if calls >= budget or time.time() > deadline:
                            break
                        y = x_local.copy()
                        y[d] += sign * step
                        y = np.clip(y, -5, 5)
                        fy = eval_func(y)
                        if fy < f_local:
                            x_local, f_local = y, fy
                            improved = True
                            break
                if not improved:
                    step *= 0.5
                    if step < 0.01:
                        break

            # Strategy 2: Try snapping to nearby "integer-like" points
            if calls < budget and time.time() < deadline:
                candidates = []
                for d in range(dim):
                    base = x_local.copy()
                    for offset in [-1, 0, 1]:
                        y = base.copy()
                        y[d] = round(y[d]) + offset
                        y = np.clip(y, -5, 5)
                        candidates.append(y)
                candidates.append(np.clip(np.round(x_local), -5, 5))

                seen = set()
                for y in candidates:
                    if calls >= budget or time.time() > deadline:
                        break
                    key = tuple(np.round(y, 6))
                    if key not in seen:
                        seen.add(key)
                        fy = eval_func(y)
                        if fy < f_local:
                            x_local, f_local = y, fy

            # Strategy 3: Final fine-tuning with very small steps
            if calls < budget and time.time() < deadline:
                x_local = best_x.copy() if best_f < f_local else x_local
                f_local = min(best_f, f_local)
                step = 0.03
                for _ in range(20):
                    if calls >= budget or time.time() > deadline:
                        break
                    improved = False
                    for d in range(dim):
                        if calls >= budget or time.time() > deadline:
                            break
                        for sign in [1, -1]:
                            if calls >= budget or time.time() > deadline:
                                break
                            y = x_local.copy()
                            y[d] += sign * step
                            y = np.clip(y, -5, 5)
                            fy = eval_func(y)
                            if fy < f_local:
                                x_local, f_local = y, fy
                                improved = True
                                break
                    if not improved:
                        step *= 0.5
                        if step < 0.005:
                            break

        return best_x, best_f
