"""Optimizer for BBOB-style Rosenbrock (f8) on [-5,5]^5."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

class Optimizer:
    """Hybrid optimizer: diverse multi-start + L-BFGS-B + Powell + Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s

        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def safe_call(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            try:
                f = float(func(x))
            except Exception:
                f = np.inf
            
            if not np.isfinite(f):
                f = np.inf
            
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Diverse Global Scan
        # Generate diverse starting points covering the search space
        rng = np.random.default_rng(42)
        start_points = []
        
        # Origin and standard points
        start_points.append(np.zeros(dim))
        start_points.append(np.ones(dim))
        start_points.append(-np.ones(dim))
        
        # Structured grid points at various magnitudes
        for v in [-4, -3, -2, 2, 3, 4]:
            start_points.append(np.full(dim, v))
        
        # Points along axes
        for i in range(dim):
            for v in [-3, -1, 1, 3]:
                pt = np.zeros(dim)
                pt[i] = v
                start_points.append(pt)
        
        # Random points
        for _ in range(30):
            pt = rng.uniform(-4.5, 4.5, size=dim)
            start_points.append(pt)
        
        # Remove duplicates
        seen = set()
        unique_points = []
        for p in start_points:
            key = tuple(np.round(p, 6))
            if key not in seen:
                seen.add(key)
                unique_points.append(p)
        
        # Evaluate start points (limit to 80 calls)
        for p in unique_points:
            if calls >= 80:
                break
            if time.time() > deadline:
                break
            f_try = safe_call(p)
        
        # Find top 3 promising starts
        # Re-evaluate to get scores (or track during evaluation)
        scored_points = []
        for p in unique_points[:40]:  # Check first 40 unique points
            if calls >= 80:
                break
            if time.time() > deadline:
                break
            # We need the function value, but safe_call doesn't return it if budget exceeded
            # Let's just use the best found so far and a few neighbors
            pass

        # Since we don't easily get back the scores from the loop above without refactoring,
        # let's use the current best_x as the primary start, and generate a few perturbations.
        # Actually, let's re-evaluate the top candidates properly.
        # For efficiency, let's just take the best_x found so far as the main start.
        # And add a few random perturbations of it.
        
        main_starts = [best_x.copy()]
        
        # Add a few diverse points based on the initial scan logic (re-using unique_points)
        # We'll pick 2 more diverse points from the unique list that are far from best_x
        candidates = []
        for p in unique_points:
            dist = np.linalg.norm(p - best_x)
            if dist > 1.0:  # Ensure diversity
                candidates.append((dist, p))
        
        candidates.sort(reverse=True)  # Farthest first for diversity
        for dist, p in candidates[:2]:
            main_starts.append(p.copy())
        
        # Phase 2: Local Refinement
        # Allocate remaining budget
        remaining = budget - calls
        
        if remaining > 50:
            # Split budget: 50% for best, 25% for second, 25% for third
            weights = [0.5, 0.25, 0.25]
            for i, x_start in enumerate(main_starts):
                if calls >= budget - 20 or time.time() > deadline:
                    break
                if i >= len(weights):
                    break
                
                alloc = int(remaining * weights[i])
                if alloc < 20:
                    continue
                
                # 2a: L-BFGS-B
                try:
                    res_lbfgs = minimize(
                        safe_call,
                        x_start.copy(),
                        method='L-BFGS-B',
                        bounds=[(-5.0, 5.0)] * dim,
                        options={
                            'maxiter': 100,
                            'maxfun': alloc // 2,
                            'ftol': 1e-14,
                            'gtol': 1e-12,
                            'disp': False
                        }
                    )
                    if res_lbfgs.fun < best_f:
                        best_x = res_lbfgs.x.copy()
                        best_f = res_lbfgs.fun
                except Exception:
                    pass

                # 2b: Powell's Method (if budget remains)
                if calls < budget - 10 and time.time() < deadline:
                    try:
                        res_powell = minimize(
                            safe_call,
                            best_x.copy(),
                            method='Powell',
                            bounds=[(-5.0, 5.0)] * dim,
                            options={
                                'maxiter': 50,
                                'maxfev': min(alloc // 2, budget - calls - 5),
                                'disp': False
                            }
                        )
                        if res_powell.fun < best_f:
                            best_x = res_powell.x.copy()
                            best_f = res_powell.fun
                    except Exception:
                        pass

        # Phase 3: Final Nelder-Mead Polish
        # Use a custom or scipy Nelder-Mead for final fine-tuning
        if calls < budget - 10 and time.time() < deadline:
            try:
                res_nm = minimize(
                    safe_call,
                    best_x.copy(),
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'maxfev': min(50, budget - calls - 5),
                        'xatol': 1e-8,
                        'fatol': 1e-12,
                        'adaptive': True,
                        'disp': False
                    }
                )
                if res_nm.fun < best_f:
                    best_x = res_nm.x.copy()
                    best_f = res_nm.fun
            except Exception:
                pass

        return best_x, best_f
