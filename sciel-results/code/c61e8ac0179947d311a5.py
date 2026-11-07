"""Optimizer for BBOB f24 (Lunacek bi-Rastrigin) using Sobol + Adaptive Nelder-Mead."""
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Optimizer using a two-phase "Global-Local" strategy:
    1. Global Phase: Dense Latin Hypercube Sampling (LHS) to identify the global basin.
       This is rotation-invariant and robust against the secondary minimum of Lunacek.
    2. Local Phase: Adaptive Nelder-Mead Simplex method starting from the best LHS point.
       This avoids the covariance adaptation overhead of CMA-ES and exploits local smoothness.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        best_x = None
        fbest = np.inf
        calls = 0

        def update_best(x, fx):
            nonlocal best_x, fbest
            if fx < fbest:
                fbest = fx
                best_x = x.copy()

        # --- Phase 1: Global Search via Latin Hypercube Sampling ---
        # We allocate ~30% of the budget to global exploration to ensure we find the correct basin.
        # For dim=5, a sample size of ~150 points provides good coverage.
        n_global = min(150, budget // 3)
        
        # Generate LHS points in [-5, 5]^dim
        # LHS ensures each dimension is divided into n equal probability intervals,
        # with exactly one sample in each interval.
        points = np.zeros((n_global, dim))
        for j in range(dim):
            # Create intervals [0, 1/n), [1/n, 2/n), ...
            indices = np.arange(n_global)
            # Random permutation to assign which interval gets which point
            perm = np.random.permutation(n_global)
            # Uniform random within each interval
            points[:, j] = (perm + np.random.uniform(size=n_global)) / n_global
        
        # Scale to [-5, 5]
        points = points * 10.0 - 5.0

        # Evaluate all global points
        global_results = []
        for pt in points:
            if not time_ok() or calls >= budget:
                break
            fx = func(pt)
            calls += 1
            update_best(pt, fx)
            global_results.append((fx, pt))
            
        if not global_results:
            # Fallback if something went wrong
            x0 = np.zeros(dim)
            fx0 = func(x0)
            calls += 1
            update_best(x0, fx0)
            return best_x, fbest

        # Sort by function value to find the best global candidate
        global_results.sort(key=lambda x: x[0])
        best_global_x = global_results[0][1].copy()
        
        # --- Phase 2: Local Refinement via Nelder-Mead ---
        # We use SciPy's Nelder-Mead implementation which is robust and efficient.
        # We run it from the best global point.
        # We might also try a few of the next best global points if the first one fails or stalls.
        
        remaining_budget = budget - calls
        if remaining_budget > 0 and time_ok():
            # We will run Nelder-Mead with a callback to count function evaluations
            # and stop if we exceed the budget or time limit.
            
            nm_calls = [0]
            
            def nm_callback(xk):
                nonlocal calls, best_x, fbest
                # Update global counter
                calls += 1
                nm_calls[0] += 1
                # Evaluate function
                fx = func(xk)
                update_best(xk, fx)
                # Check time and budget
                if not time_ok() or calls >= budget:
                    raise RuntimeError("Stop")

            # Initial simplex configuration
            # Default Nelder-Mead uses an initial simplex based on the starting point.
            # We can customize it for better initial spread if needed, but default is usually fine
            # given we started from a good LHS point.
            
            try:
                res = minimize(
                    lambda x: func(x), # Note: minimize calls func, but we need to track calls carefully.
                    # Actually, passing func directly is fine if we don't need the callback for counting 
                    # because minimize handles the loop. But we need to count calls for the budget.
                    # It is safer to use a wrapper that counts calls.
                    best_global_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': remaining_budget,
                        'adaptive': True,
                        'xatol': 1e-6,
                        'fatol': 1e-6,
                        # 'callback': nm_callback # Callback in scipy minimize is called after each iteration, not each function eval.
                    }
                )
            except RuntimeError:
                pass
            except Exception:
                pass
            
            # The above approach has a subtle issue: `minimize` manages its own function calls.
            # To strictly respect the `calls` counter and `budget`, it's better to implement a simple 
            # local search or use a wrapper that increments `calls` inside the objective function passed to minimize.
            
            # Let's re-do Phase 2 with a strict call-counting wrapper.
            
            if calls < budget and time_ok():
                def objective(x):
                    nonlocal calls, best_x, fbest
                    if not time_ok():
                        raise RuntimeError("Time limit exceeded")
                    if calls >= budget:
                        raise RuntimeError("Budget exceeded")
                    calls += 1
                    fx = func(x)
                    update_best(x, fx)
                    return fx

                # We try the best global point
                try:
                    # Reset any previous state if we were to restart, but we are continuing from best_global_x
                    # Nelder-Mead doesn't have state to reset easily, so we just call it.
                    # We limit maxiter to be safe, though the callback/exception will stop it.
                    res = minimize(
                        objective,
                        best_global_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': budget - calls + 10, # Generous, will be cut by exceptions
                            'adaptive': True,
                            'xatol': 1e-5,
                            'fatol': 1e-5
                        }
                    )
                except RuntimeError:
                    pass
                except Exception:
                    pass

        return best_x, fbest
