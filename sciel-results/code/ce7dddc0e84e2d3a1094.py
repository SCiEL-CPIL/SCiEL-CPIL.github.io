"""Exploit quadratic structure of Discus via BFGS optimization with robust initial exploration."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Optimizer that exploits the structure of BBOB f11 (Discus)
    using Nelder-Mead, which handles non-smoothness and flat directions well."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # safety margin before 330s limit
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal best_x, best_f, calls
            if calls >= budget:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initial Exploration
        # We need to find the "needle" direction.
        # Evaluate origin and points along axes.
        evaluate(np.zeros(dim))
        
        # Axis sweeps to identify the flat direction
        for axis in range(dim):
            for r in [1.0, 2.0, 3.0, 4.0]:
                if calls >= budget or time.time() > deadline:
                    break
                x_pos = np.zeros(dim)
                x_pos[axis] = r
                evaluate(x_pos)
                
                x_neg = np.zeros(dim)
                x_neg[axis] = -r
                if calls < budget and time.time() < deadline:
                    evaluate(x_neg)
            if calls >= budget or time.time() > deadline:
                break

        # Random sweeps to catch rotated features
        n_random = 20
        for _ in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            evaluate(x)

        # Phase 2: Nelder-Mead Optimization
        # Nelder-Mead is derivative-free and robust to the non-smooth max() term
        # and flat directions present in the Discus function.
        if best_x is not None and calls < budget:
            # Define a callback to stop if we hit budget or deadline
            def nm_callback(xk):
                return calls >= budget or time.time() > deadline

            try:
                # Nelder-Mead options:
                # maxiter: max number of iterations
                # xatol, fatol: tolerances for convergence
                # We set maxiter high enough but rely on maxfev/budget via callback
                res = minimize(
                    evaluate, 
                    best_x, 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': 200, 
                        'xatol': 1e-4, 
                        'fatol': 1e-4,
                        'initial_simplex': None # Let scipy choose default
                    },
                    callback=nm_callback
                )
                # Update best if Nelder-Mead found something better
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        # Phase 3: Polish with a second Nelder-Mead run or local refinement
        # If we have budget left, try a local search around the best point
        if calls < budget - 20 and time.time() < deadline:
            # Try a smaller step local search or another NM run with tighter bounds
            # NM can sometimes get stuck in a wide simplex. 
            # Let's try a few random restarts from the best point with small perturbations
            # and run NM on them if budget allows.
            n_restarts = 3
            for i in range(n_restarts):
                if calls >= budget - 10 or time.time() > deadline:
                    break
                x_start = best_x + rng.normal(0, 0.5, dim)
                x_start = np.clip(x_start, -5.0, 5.0)
                
                # Only run if this start is not too far away or if we expect improvement
                # Actually, just run it. NM is fast.
                try:
                    res2 = minimize(
                        evaluate, 
                        x_start, 
                        method='Nelder-Mead', 
                        options={
                            'maxiter': 50, 
                            'xatol': 1e-3, 
                            'fatol': 1e-3
                        },
                        callback=nm_callback
                    )
                    if res2.fun < best_f:
                        best_f = res2.fun
                        best_x = res2.x
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
