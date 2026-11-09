"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Hybrid optimizer: (1+1)-ES with adaptive restarts and Nelder-Mead refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Initial point: Center of the box is a strong heuristic for rotated Rosenbrock
        x = np.zeros(n)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls_made = 1

        sigma = 2.0
        no_improve = 0
        max_no_improve = 40

        # Phase 1: (1+1)-ES Global Search
        while calls_made < budget:
            if time.time() > deadline:
                break

            # Generate candidate
            y = x + sigma * rng.standard_normal(n)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls_made += 1

            if fy < fx:
                # Accept step
                x = y
                fx = fy
                no_improve = 0
                
                # Update global best
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()

                # Increase sigma on success to explore larger regions
                sigma *= 1.25
            else:
                # Reject step
                no_improve += 1
                # Decrease sigma on failure to refine local search
                sigma *= 0.9

                # Restart logic if stuck
                if no_improve >= max_no_improve:
                    # Strategy: First try large jumps from the best point found so far
                    restarted = False
                    for _ in range(3):
                        if calls_made >= budget or time.time() > deadline:
                            break
                        candidate = best_x + 3.0 * rng.standard_normal(n)
                        candidate = np.clip(candidate, -5, 5)
                        cf = func(candidate)
                        calls_made += 1
                        if cf < best_fx:
                            best_fx = cf
                            best_x = candidate.copy()
                            x = best_x.copy()
                            fx = best_fx
                            no_improve = 0
                            sigma = 2.0
                            restarted = True
                            break
                    
                    # If large jumps didn't help, try a full random restart
                    if not restarted:
                        x_new = rng.uniform(-5, 5, n)
                        fx_new = func(x_new)
                        calls_made += 1
                        if fx_new < best_fx:
                            best_fx = fx_new
                            best_x = x_new.copy()
                            x = x_new.copy()
                            fx = fx_new
                        else:
                            # Keep current x but reset parameters to try again
                            pass 
                        no_improve = 0
                        sigma = 2.0

        # Phase 2: Local Refinement with Nelder-Mead
        # This is crucial for Rosenbrock to squeeze out the last bit of performance
        if calls_made < budget and time.time() < deadline - 10:
            try:
                from scipy.optimize import minimize

                remaining_calls = budget - calls_made
                
                # Nelder-Mead can be expensive in terms of calls per iteration.
                # Limit iterations to ensure we don't exceed budget.
                # Typically NM uses ~ (dim+1) calls per iteration.
                max_iter = max(1, remaining_calls // (n + 2))
                
                def nm_wrapper(z):
                    nonlocal calls_made
                    if calls_made >= budget:
                        return best_fx
                    if time.time() > deadline:
                        return best_fx
                    calls_made += 1
                    return func(np.clip(z, -5, 5))

                # Start NM from the best point found by ES
                res = minimize(
                    nm_wrapper,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iter,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )

                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
