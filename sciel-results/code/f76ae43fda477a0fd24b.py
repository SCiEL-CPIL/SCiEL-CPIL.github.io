"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as so

class Optimizer:
    """Differential Evolution + Nelder-Mead Polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget
        
        call_count = 0
        best_x = np.zeros(dim)
        best_fx = np.inf
        lock = np.zeros(1) # Simple lock for thread safety if needed, but single thread here

        def counting_func(x):
            nonlocal call_count, best_x, best_fx
            # Check time
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            
            # Clip to box
            x = np.clip(x, -5.0, 5.0)
            
            # Evaluate
            fx = func(x)
            call_count += 1
            
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
                
            return fx

        # Phase 1: Differential Evolution (Global)
        # Budget allocation: ~70% for DE, ~30% for Polish
        de_budget = int(budget * 0.7)
        
        # DE Parameters
        # popsize * maxiter <= de_budget
        # For 5D, popsize=10 is reasonable.
        popsize = 10
        maxiter = de_budget // popsize
        if maxiter < 1:
            maxiter = 1
            
        try:
            res = so.differential_evolution(
                counting_func,
                bounds=[(-5.0, 5.0)] * dim,
                popsize=popsize,
                maxiter=maxiter,
                tol=1e-8,
                seed=42,
                init='latinhypercube',
                polish=False # We will do our own polish to control budget
            )
        except TimeoutError:
            pass
        except Exception as e:
            # If DE fails, we still have best_x from counting_func
            pass

        # Phase 2: Nelder-Mead Polish (Local)
        # Use remaining budget
        remaining_budget = budget - call_count
        if remaining_budget > 10 and time.time() < deadline:
            # Nelder-Mead
            # maxiter for NM is number of function evaluations roughly
            # Set maxiter to remaining_budget to be safe
            try:
                res_nm = so.minimize(
                    counting_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': remaining_budget, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                # Update best if NM found better
                if res_nm.fun < best_fx:
                    best_fx = res_nm.fun
                    best_x = res_nm.x
            except TimeoutError:
                pass
            except Exception:
                pass

        return best_x, best_fx

# EVOLVE-BLOCK-END
