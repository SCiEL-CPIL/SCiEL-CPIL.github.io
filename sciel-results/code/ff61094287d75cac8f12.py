"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Sobol Scout + Powell + (1+1)-ES refinement for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        best_x = None
        best_f = float('inf')
        eval_count = 0
        dim = self.dim
        budget = self.budget

        def wrapped_func(x):
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget:
                raise Exception("Budget exceeded")
            if time.time() > deadline:
                raise Exception("Time exceeded")
            
            x = np.asarray(x, dtype=float)
            fx = func(x)
            eval_count += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            return fx

        # Phase 1: Global Search
        # Use Sobol sequence for better space coverage than random in low dims
        try:
            from scipy.stats import qmc
            n_scout = 80
            sampler = qmc.Sobol(d=dim, scramble=True)
            samples = sampler.random(n=n_scout)
            # Map from [0, 1]^dim to [-5, 5]^dim
            samples = samples * 10.0 - 5.0
            scout_points = list(samples)
        except ImportError:
            # Fallback to random if qmc is not available
            n_scout = 80
            scout_points = np.random.uniform(-5, 5, size=(n_scout, dim)).tolist()

        # Add strategic points
        scout_points.append(np.zeros(dim))
        scout_points.append(np.ones(dim))
        scout_points.append(-np.ones(dim))

        for p in scout_points:
            if time.time() > deadline or eval_count >= budget:
                break
            try:
                fp = wrapped_func(p)
            except Exception:
                break

        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = func(best_x)
                eval_count += 1
            except Exception:
                pass

        remaining = budget - eval_count
        if remaining > 10:
            stop_flag = {'stop': False}
            
            def callback(xk):
                if time.time() > deadline or eval_count >= budget:
                    stop_flag['stop'] = True
                    return True
                return False

            # Phase 2: Powell's method for initial refinement
            try:
                res = minimize(
                    wrapped_func, 
                    best_x, 
                    method='Powell', 
                    options={
                        'xtol': 1e-8, 
                        'ftol': 1e-10,
                        'maxiter': 200,
                        'maxfev': min(remaining, 150)
                    },
                    callback=callback
                )
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                pass

            # Phase 3: (1+1)-ES refinement to escape shallow local minima
            remaining = budget - eval_count
            if remaining > 10 and time.time() < deadline:
                # (1+1)-ES with adaptive step size
                x = best_x.copy()
                f = best_f
                sigma = 0.1  # Initial step size
                
                # Run (1+1)-ES for a portion of remaining budget
                max_es_evals = min(remaining, 100)
                es_evals = 0
                
                while es_evals < max_es_evals and eval_count < budget and time.time() < deadline:
                    # Generate offspring
                    z = np.random.randn(dim)
                    x_trial = x + sigma * z
                    
                    # Evaluate
                    try:
                        f_trial = wrapped_func(x_trial)
                        es_evals += 1
                    except Exception:
                        break
                    
                    # Accept if better
                    if f_trial < f:
                        x = x_trial
                        f = f_trial
                        sigma *= 1.05  # Increase step size on success
                    else:
                        sigma *= 0.95  # Decrease step size on failure
                    
                    # Update global best
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                    
                    # Check for convergence
                    if sigma < 1e-6:
                        break

            # Phase 4: Final Powell polish
            remaining = budget - eval_count
            if remaining > 5 and time.time() < deadline:
                stop_flag['stop'] = False
                try:
                    res = minimize(
                        wrapped_func, 
                        best_x, 
                        method='Powell', 
                        options={
                            'xtol': 1e-10, 
                            'ftol': 1e-12,
                            'maxiter': 100,
                            'maxfev': remaining
                        },
                        callback=callback
                    )
                    
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                        
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
