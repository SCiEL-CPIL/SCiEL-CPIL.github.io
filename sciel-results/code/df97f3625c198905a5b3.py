import numpy as np
import time
from cma import CMAEvolutionStrategy

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin
        
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made
            if time.time() > deadline:
                return np.inf
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            try:
                val = func(x)
                if np.isnan(val) or np.isinf(val):
                    return 1e10
                return val
            except Exception:
                return 1e10

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Multi-Start CMA-ES Strategy
        # We divide the budget into several phases.
        # Phase 1: Small population, wide spread (Global exploration)
        # Phase 2: Medium population, tighter spread (Local exploitation)
        # Phase 3: Large population, very tight spread (Polishing)
        
        # Configuration for phases: (pop_size, sigma_init, max_iter_est)
        # Total budget is 500.
        # Let's try 3 phases.
        
        # Phase 1: 10% budget, pop 10, sigma 2.0
        # Phase 2: 30% budget, pop 20, sigma 1.0
        # Phase 3: 60% budget, pop 30, sigma 0.5
        
        # However, CMA-ES overhead is high. Let's use a simpler allocation.
        # We will run CMA-ES with a specific maxiter limit based on remaining budget.
        
        # Initial start: Center of the box
        x0 = np.zeros(dim)
        
        # Run 1: Broad search
        pop1 = 10
        sigma1 = 2.5
        # CMA-ES typically needs ~ (10 + 2 * sqrt(dim/pop)) * dim * ln(1/tol) iterations
        # For dim=5, pop=10, this is roughly 10-20 iterations.
        # Each iteration costs 'pop' function calls.
        # So 20 iterations * 10 calls = 200 calls.
        # Let's limit maxiter to ensure we don't blow the budget.
        
        remaining = budget - calls_made
        if remaining > pop1 * 5: # At least 5 iterations
            # Estimate max iterations
            max_iter1 = max(5, (remaining // 2) // pop1) # Use half remaining budget for phase 1
            try:
                es1 = CMAEvolutionStrategy(x0, sigma1, {'popsize': pop1, 'maxiter': max_iter1, 'bounds': [-5, 5]})
                es1.ask
                while not es1.stop() and calls_made < budget and time.time() < deadline:
                    solutions = es1.ask()
                    fitvals = [safe_call(s) for s in solutions]
                    es1.tell(solutions, fitvals)
                    # Update best
                    min_idx = np.argmin(fitvals)
                    if fitvals[min_idx] < best_f:
                        update_best(solutions[min_idx], fitvals[min_idx])
                        x0 = solutions[min_idx].copy() # Update start for next phase
            except Exception:
                pass

        remaining = budget - calls_made
        if remaining > 10 and time.time() < deadline:
            # Run 2: Focused search around best found
            pop2 = 20
            sigma2 = 1.0
            max_iter2 = max(5, remaining // pop2)
            try:
                es2 = CMAEvolutionStrategy(x0, sigma2, {'popsize': pop2, 'maxiter': max_iter2, 'bounds': [-5, 5]})
                while not es2.stop() and calls_made < budget and time.time() < deadline:
                    solutions = es2.ask()
                    fitvals = [safe_call(s) for s in solutions]
                    es2.tell(solutions, fitvals)
                    min_idx = np.argmin(fitvals)
                    if fitvals[min_idx] < best_f:
                        update_best(solutions[min_idx], fitvals[min_idx])
                        x0 = solutions[min_idx].copy()
            except Exception:
                pass

        remaining = budget - calls_made
        if remaining > 5 and time.time() < deadline:
            # Run 3: Polishing
            pop3 = 30
            sigma3 = 0.2
            max_iter3 = max(2, remaining // pop3)
            try:
                es3 = CMAEvolutionStrategy(x0, sigma3, {'popsize': pop3, 'maxiter': max_iter3, 'bounds': [-5, 5]})
                while not es3.stop() and calls_made < budget and time.time() < deadline:
                    solutions = es3.ask()
                    fitvals = [safe_call(s) for s in solutions]
                    es3.tell(solutions, fitvals)
                    min_idx = np.argmin(fitvals)
                    if fitvals[min_idx] < best_f:
                        update_best(solutions[min_idx], fitvals[min_idx])
                        x0 = solutions[min_idx].copy()
            except Exception:
                pass

        return best_x, best_f
