import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # Safety margin
        rng = np.random.default_rng(42)
        eval_count = 0
        best_f = np.inf
        best_x = np.zeros(dim)

        def wrapped(x):
            nonlocal eval_count, best_f, best_x
            if time.time() >= deadline:
                raise TimeoutError("Time limit exceeded")
            if eval_count >= budget:
                raise ValueError("Budget exceeded")
            eval_count += 1
            x = np.asarray(x, dtype=np.float64).flatten()
            x = np.clip(x, -5, 5)
            f_val = float(func(x))
            if np.isfinite(f_val) and f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Halton sequence for global coverage
        def halton(index, base):
            result = 0.0
            f = 1.0
            i = index
            while i > 0:
                f /= base
                result += f * (i % base)
                i //= base
            return result

        def halton_sequence(n, dim):
            points = []
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            for i in range(1, n + 1):
                point = []
                for d in range(dim):
                    point.append(halton(i, primes[d]))
                points.append(point)
            return np.array(points)

        # Generate Halton points mapped to [-5, 5]^dim
        n_halton = min(15, budget // 30)
        if n_halton > 0:
            halton_pts = halton_sequence(n_halton, dim)
            halton_pts = halton_pts * 10.0 - 5.0  # map [0,1] to [-5,5]
            for x0 in halton_pts:
                if eval_count >= budget or time.time() >= deadline:
                    break
                try:
                    wrapped(x0)
                except (TimeoutError, ValueError):
                    break

        # Add origin
        if eval_count < budget and time.time() < deadline:
            try:
                wrapped(np.zeros(dim))
            except (TimeoutError, ValueError):
                pass

        # Phase 2: Nelder-Mead from best point (derivative-free, handles anisotropy)
        if eval_count < budget and time.time() < deadline:
            remaining = budget - eval_count
            maxfev = min(remaining, 300)
            if maxfev > 20:
                try:
                    res = minimize(
                        wrapped,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 200,
                            'maxfev': maxfev,
                            'xatol': 1e-8,
                            'fatol': 1e-12,
                            'adaptive': True
                        }
                    )
                except (TimeoutError, ValueError):
                    pass
                except Exception:
                    pass

        # Phase 3: Coordinate descent polishing
        if eval_count < budget and time.time() < deadline:
            remaining = budget - eval_count
            if remaining > 10:
                step_sizes = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
                for step in step_sizes:
                    if eval_count >= budget or time.time() >= deadline:
                        break
                    for d in range(dim):
                        if eval_count >= budget or time.time() >= deadline:
                            break
                        for direction in [1, -1]:
                            if eval_count >= budget or time.time() >= deadline:
                                break
                            x_try = best_x.copy()
                            x_try[d] += direction * step
                            x_try = np.clip(x_try, -5, 5)
                            try:
                                f_try = wrapped(x_try)
                                if f_try < best_f:
                                    best_f = f_try
                                    best_x = x_try.copy()
                            except (TimeoutError, ValueError):
                                break

        # Fallback: if something went wrong and no calls were made, do random search
        if eval_count == 0:
            for _ in range(min(10, budget)):
                x = rng.uniform(-5, 5, dim)
                try:
                    f_val = float(func(x))
                    eval_count += 1
                    if f_val < best_f:
                        best_f = f_val
                        best_x = x.copy()
                except Exception:
                    break

        return best_x, best_f
