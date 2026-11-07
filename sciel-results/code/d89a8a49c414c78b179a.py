import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Use 280s to be safe against system overhead/IO (limit is 300s)
        deadline = start_time + 280.0
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        candidates = []

        def check_budget():
            nonlocal calls_used
            if calls_used >= self.budget:
                return False
            if time.time() > deadline:
                return False
            return True

        def safe_call(x):
            nonlocal calls_used, best_f, best_x, candidates
            if not check_budget():
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            candidates.append((fx, x.copy()))
            return fx

        # Phase 1: Global Exploration
        # Revert to 40 points (30 Sobol + 10 Random) as 100 was too many
        n_sobol = 30
        n_random = 10
        
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
        sobol_points = sampler.random(n_sobol)
        X_sobol = sobol_points * 10.0 - 5.0
        
        rng = np.random.default_rng(42)
        X_random = rng.uniform(-5.0, 5.0, size=(n_random, self.dim))
        
        X_initial = np.vstack([X_sobol, X_random])
        
        for x in X_initial:
            if not check_budget():
                break
            safe_call(x)

        if best_x is None:
            # Fallback
            x = np.zeros(self.dim)
            safe_call(x)
            if best_x is None:
                return np.zeros(self.dim), 0.0

        # Phase 2: Local Refinement with Powell's Method (Derivative-Free)
        # Sort candidates by function value
        candidates.sort(key=lambda t: t[0])
        
        # Select top 3 unique-ish points (matching parent config)
        top_candidates = []
        for fx, x in candidates:
            if len(top_candidates) >= 3:
                break
            is_unique = True
            for _, x_prev in top_candidates:
                if np.linalg.norm(x - x_prev) < 0.5:
                    is_unique = False
                    break
            if is_unique:
                top_candidates.append((fx, x))
        
        # Powell's method is derivative-free and often more efficient for low-dim
        # It does not accept bounds, but func clips, so it's safe.
        
        for fx0, x0 in top_candidates:
            if not check_budget() or time.time() > deadline - 10.0:
                break
            
            def objective(x):
                fx = safe_call(x)
                if fx is None:
                    raise Exception("Budget exceeded")
                return fx

            try:
                # Powell is robust for this problem type
                res = minimize(
                    objective,
                    x0,
                    method='Powell',
                    options={
                        'maxiter': 200,
                        'ftol': 1e-12,
                        'xtol': 1e-12
                    }
                )
            except Exception:
                pass

        # Phase 3: Final Polish with Nelder-Mead (Derivative-Free)
        # If budget remains, do one more short run from the global best
        if check_budget() and time.time() < deadline - 5.0:
            def objective_polish(x):
                fx = safe_call(x)
                if fx is None:
                    raise Exception("Budget exceeded")
                return fx

            try:
                # Nelder-Mead with small initial simplex
                # Default simplex might be too large or small
                # Let scipy handle it, but limit iterations
                res = minimize(
                    objective_polish,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 100,
                        'xatol': 1e-12,
                        'fatol': 1e-12
                    }
                )
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(self.dim)
            if check_budget():
                safe_call(best_x)

        return best_x, best_f
