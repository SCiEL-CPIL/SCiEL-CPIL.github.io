import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return best_f
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def run_nm(x0, max_iter, xatol=1e-8, fatol=1e-10):
            nonlocal calls
            if calls >= budget or time.time() >= deadline:
                return
            try:
                def objective(x):
                    return evaluate(x)
                
                minimize(
                    objective,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iter,
                        'xatol': xatol,
                        'fatol': fatol,
                        'adaptive': True
                    }
                )
            except Exception:
                pass

        def run_lbfgsb(x0, max_iter, gtol=1e-10):
            nonlocal calls
            if calls >= budget or time.time() >= deadline:
                return
            try:
                def objective(x):
                    return evaluate(x)
                
                minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * n,
                    options={
                        'maxiter': max_iter,
                        'gtol': gtol,
                        'maxls': 50
                    }
                )
            except Exception:
                pass

        # Phase 1: Diverse Exploration using Sobol Quasi-Random Sequence
        # We want to cover the space systematically. 
        # Let's allocate ~60% of the budget for exploration.
        num_exploration = 15
        calls_per_exploration = max(10, int((budget * 0.6) / num_exploration))
        
        # Generate Sobol points
        # We need at least 2^k points for good coverage, but 15 is small.
        # We can generate a small Sobol set and map it to [-4.8, 4.8]
        # Also include origin and a few random points for robustness.
        
        sobol_sampler = qmc.Sobol(d=n, scramble=True, seed=42)
        # Generate 2^4 = 16 points, take first 14 (excluding 0 and 1 corners if we want interior, 
        # but Sobol includes corners. Let's just generate 16 and use them).
        sobol_points = sobol_sampler.random(16)
        
        # Map [0,1] to [-4.8, 4.8]
        sobol_points = sobol_points * 9.6 - 4.8
        
        # Create start list
        starts = [np.zeros(n)] # Always start at origin
        for p in sobol_points[:num_exploration-1]:
            starts.append(p)
            
        # Add a couple of random points to ensure diversity if Sobol misses something
        while len(starts) < num_exploration:
            starts.append(rng.uniform(-4.8, 4.8, size=n))

        for x0 in starts:
            if calls >= budget or time.time() >= deadline:
                break
            
            # 1. L-BFGS-B for fast convergence on smooth Griewank part
            # Allocate half the per-exploration budget to L-BFGS-B
            lbfgs_iter = max(5, calls_per_exploration // 2)
            run_lbfgsb(x0, max_iter=lbfgs_iter, gtol=1e-8)
            
            # 2. Nelder-Mead for polishing and handling Rosenbrock valley
            # Use the best point found by L-BFGS-B (or x0 if L-BFGS-B failed/stalled)
            if calls < budget and time.time() < deadline:
                # If best_x improved, start NM from best_x, else from x0
                nm_start = best_x if best_x is not None else x0
                nm_iter = max(5, calls_per_exploration // 2)
                run_nm(nm_start, max_iter=nm_iter, xatol=1e-8, fatol=1e-10)

        # Phase 2: Polish with high-precision methods
        if best_x is not None and calls < budget and time.time() < deadline:
            remaining = budget - calls
            
            # L-BFGS-B with tighter tolerance
            if remaining > 10:
                run_lbfgsb(best_x, max_iter=min(remaining, 50), gtol=1e-12)
            
            # Final NM polish with very tight tolerance
            if best_x is not None and calls < budget and time.time() < deadline:
                remaining = budget - calls
                if remaining > 5:
                    run_nm(best_x, max_iter=remaining, xatol=1e-12, fatol=1e-14)

        return best_x, best_f
