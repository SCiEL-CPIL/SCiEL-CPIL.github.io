"""Optimized BBOB f22 solver using Sobol global search and 1+1 CMA-ES refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Sobol-based global search with 1+1 CMA-ES local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit
        n = self.dim
        budget = self.budget

        best_x = np.zeros(n)
        best_f = float('inf')
        calls = 0

        def try_eval(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return None
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            try:
                f = float(func(x))
            except Exception:
                f = float('inf')
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def cma_es_1plus1(x0, f0, budget_left, sigma0=0.5):
            """
            1+1 CMA-ES for local refinement.
            """
            nonlocal calls, best_x, best_f
            if budget_left < 10:
                return x0, f0

            x = x0.copy()
            f = f0
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            
            # Strategy parameters for 1+1 CMA-ES
            cs = (n + 2) / (n + 8)
            cmu = 2 / ((n + 4) ** 2 + 1)
            cc = (4 + n / 4) / (n + 4 + 2 / cs)
            damps = 1 + 2 * max(0, np.sqrt((n - 1) / (n + 3)) - 1) + cs
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            
            gen = 0

            for _ in range(budget_left):
                if calls >= budget - 2 or time.time() > deadline - 2:
                    break

                # Sample
                z = np.random.standard_normal(n)
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(n)
                    C = np.eye(n)
                
                y = sigma * (L @ z)
                x_trial = x + y
                x_trial = np.clip(x_trial, -5.0, 5.0)

                f_trial = try_eval(x_trial)
                if f_trial is None:
                    break

                if f_trial < f:
                    # Success
                    y = x_trial - x
                    step = y / sigma
                    pc = (1 - cc) * pc + cc * step
                    ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs)) * step
                    
                    norm_pc = np.linalg.norm(pc)
                    # hsig approximation
                    # E||pc|| = sqrt(1 - (1-cc)^(2*gen)) * chi_n
                    # For simplicity and speed, use a threshold based on chi_n
                    hs = 1 if (norm_pc / chi_n < 1.4 + 2/(n+1)) else 0
                    
                    C = (1 - cmu * hs) * C + cmu * hs * np.outer(step, step)
                    C = (C + C.T) / 2  # Ensure symmetry
                    
                    x = x_trial
                    f = f_trial
                    
                    # Update sigma
                    sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
                    sigma = np.clip(sigma, 1e-8, 10.0)
                else:
                    # Failure
                    pass

            return x, f

        # Phase 1: Sobol quasi-random global sampling
        # Use Sobol sequence for uniform coverage
        # We need enough points to cover the space well. 128 points is a good start.
        # Ensure power of 2 for Sobol.
        n_global = 128
        sobol_engine = qmc.Sobol(d=n, scramble=False)
        # Generate points in [0, 1]^n
        points = sobol_engine.random(n_global)
        
        # Map to [-5, 5]^n
        points = points * 10.0 - 5.0
        
        for i in range(n_global):
            if calls >= budget - 50 or time.time() > deadline - 30:
                break
            x = points[i]
            try_eval(x)

        # Add a few random points for diversity if budget allows
        rng = np.random.default_rng(42)
        for _ in range(10):
            if calls >= budget - 50 or time.time() > deadline - 30:
                break
            x = rng.uniform(-5, 5, n)
            try_eval(x)

        # Phase 2: Local refinement with 1+1 CMA-ES from best point
        if best_f < float('inf') and calls < budget - 10:
            remaining = budget - calls
            # Use most of remaining budget for CMA-ES
            cma_budget = min(remaining - 5, 300)
            if cma_budget >= 10:
                x_best, f_best = cma_es_1plus1(best_x, best_f, cma_budget, sigma0=0.5)
                if f_best < best_f:
                    best_x = x_best
                    best_f = f_best

        # Phase 3: If we have budget left, try a second CMA-ES from a different starting point
        if calls < budget - 10 and time.time() < deadline - 10:
            # Find second-best distinct point from the Sobol set or generate new ones
            # Let's just generate a few new random points to find a distinct basin
            candidates = []
            for _ in range(20):
                if calls >= budget - 10 or time.time() > deadline - 10:
                    break
                x = rng.uniform(-5, 5, n)
                f = try_eval(x)
                if f is not None:
                    candidates.append((f, x))

            if candidates:
                candidates.sort(key=lambda t: t[0])
                # Pick the best candidate that is far from current best
                for f_c, x_c in candidates:
                    if np.linalg.norm(x_c - best_x) > 1.0:
                        remaining = budget - calls
                        if remaining >= 10:
                            cma_budget = min(remaining - 5, 100)
                            if cma_budget >= 10:
                                x_try, f_try = cma_es_1plus1(x_c, f_c, cma_budget, sigma0=0.5)
                                if f_try < best_f:
                                    best_x = x_try
                                    best_f = f_try
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
