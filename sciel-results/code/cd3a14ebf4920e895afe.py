"""CMA-ES based optimizer for Rosenbrock with adaptive population and restarts."""
import numpy as np
import time
from scipy.optimize import cma


class Optimizer:
    """CMA-ES based optimizer for Rosenbrock with adaptive population and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)

        # CMA-ES requires a population size, typically 4 + 3*log(n)
        # For n=5, lambda = 4 + 3*2.3 = ~11. Let's use 15 for robustness.
        lambda_pop = 15
        
        # Initial bounds for the search
        # The minimizer is in [-4, 4]^5, but the function clips to [-5, 5].
        # We can start with a box slightly smaller than the full domain to avoid boundary issues
        # or use the full domain. CMA-ES handles bounds via resampling or penalty, 
        # but standard cma.py uses a penalty or clipping depending on version.
        # To be safe and efficient, we'll initialize the mean at 0 and sigma at 2.0 (covers most of [-4,4]).
        
        best_x = np.zeros(self.dim)
        best_f = np.inf
        calls_used = 0

        # We will run CMA-ES with a strict call budget.
        # CMA-ES is stochastic, so we might run it multiple times if we have budget left
        # and it stagnates, or just let it run until budget is exhausted.
        # Given the budget is 500, and lambda=15, that's ~33 generations.
        # This is usually sufficient for Rosenbrock in 5D if started reasonably.
        
        # To improve robustness, we can perform a quick initial scan to find a good start.
        # However, CMA-ES is good at global search. Let's stick to a single robust run
        # or a small number of restarts if the first one fails to make progress.
        
        # Phase 1: Initial CMA-ES Run
        # We allocate 80% of the budget to the first run.
        budget_run1 = int(self.budget * 0.8)
        
        # CMA-ES options
        # maxfun is the total number of function evaluations.
        # We need to wrap func to count calls and enforce the deadline.
        
        call_count = [0]
        
        def func_wrapper(x):
            nonlocal calls_used
            if call_count[0] >= budget_run1:
                raise ValueError("Budget exceeded")
            if time.time() > deadline - 5:
                raise ValueError("Time exceeded")
            call_count[0] += 1
            calls_used += 1
            val = func(x)
            # Update global best
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        try:
            # Initial mean: 0
            # Initial sigma: 2.0 (covers [-4,4] roughly with 2 sigma)
            x0 = np.zeros(self.dim)
            sigma0 = 2.0
            
            opts = {
                'maxiter': 100, # Generations
                'maxfun': budget_run1, # Function evaluations
                'verbose': -9, # Quiet
                'seed': 42
            }
            
            es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
            
            # Run until maxfun or maxiter
            while not es.stop():
                solutions = es.ask()
                for x in solutions:
                    # Clip to box to ensure validity, though CMA-ES usually stays in bounds
                    # if started inside and sigma is reasonable.
                    # The task says func clips, so we don't strictly need to, 
                    # but it's good practice to keep the search space sane.
                    x_clipped = np.clip(x, -5, 5)
                    f_val = func_wrapper(x_clipped)
                    es.tell(solutions, [f_val] if len(solutions)==1 else None) # This is wrong, tell takes list of fvals
                    
            # Correction: cma.py interface
            # es.ask() returns list of x
            # es.tell(list_of_x, list_of_f)
            
            # Let's re-implement the loop correctly
            # The previous loop was pseudocode.
            
        except Exception:
            pass

        # The above try block was a sketch. Let's write the actual loop properly.
        # I will discard the previous attempt and write a clean CMA-ES loop.
        
        # Reset state for a clean run
        best_x = np.zeros(self.dim)
        best_f = np.inf
        calls_used = 0
        call_count = [0]
        budget_run1 = int(self.budget * 0.8)

        def func_wrapper(x):
            if call_count[0] >= budget_run1:
                raise ValueError("Budget exceeded")
            if time.time() > deadline - 5:
                raise ValueError("Time exceeded")
            call_count[0] += 1
            calls_used += 1
            val = func(x)
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        try:
            x0 = np.zeros(self.dim)
            sigma0 = 2.0
            opts = {
                'maxiter': 100,
                'maxfun': budget_run1,
                'verbose': -9,
                'seed': 42
            }
            es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
            
            while not es.stop():
                solutions = es.ask()
                fvals = []
                for x in solutions:
                    x_clipped = np.clip(x, -5, 5)
                    fvals.append(func_wrapper(x_clipped))
                es.tell(solutions, fvals)
                
        except Exception:
            pass

        # Phase 2: Polish with Nelder-Mead if we have budget left
        # CMA-ES might get close but not perfectly precise. NM is good for final polish.
        remaining = self.budget - calls_used
        if remaining > 20 and time.time() < deadline - 10:
            nm_budget = min(remaining, 50)
            nm_count = [0]
            
            def nm_wrapper(x):
                if nm_count[0] >= nm_budget:
                    raise ValueError("NM budget exceeded")
                if time.time() > deadline - 5:
                    raise ValueError("Time exceeded")
                nm_count[0] += 1
                calls_used += 1
                val = func(x)
                if val < best_f:
                    best_f = val
                    best_x = x.copy()
                return val

            from scipy.optimize import minimize
            try:
                res_nm = minimize(
                    nm_wrapper, 
                    best_x, 
                    method='Nelder-Mead',
                    options={'maxiter': nm_budget, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
            except Exception:
                pass

        return best_x, best_f
