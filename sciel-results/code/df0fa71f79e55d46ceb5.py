"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

class Optimizer:
    """Global (Differential Evolution) + Local (Powell) Hybrid Optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = np.inf
        self.calls_made = 0
        self.start_time = 0.0
        self.deadline = 0.0

    def __call__(self, func):
        self.start_time = time.time()
        # 300s limit, use 285s for safety margin
        self.deadline = self.start_time + 285.0
        
        rng = np.random.default_rng(42)
        
        # Strategy:
        # 1. Global Phase: Differential Evolution (DE) to find the basin of attraction.
        #    DE is robust to rotations and can handle the narrow valley better than simple random search.
        #    We allocate ~60% of the budget to this phase.
        # 2. Local Phase: Powell's method (derivative-free, good for Rosenbrock) to polish.
        #    We allocate ~40% of the budget to this phase.
        
        # Wrapper to count calls and enforce budget/time
        def wrapped_func(x):
            self.calls_made += 1
            if self.calls_made > self.budget:
                return 1e10
            if time.time() > self.deadline:
                return 1e10
            
            x_clipped = np.clip(x, -5, 5)
            val = func(x_clipped)
            
            if val < self.best_f:
                self.best_f = val
                self.best_x = x_clipped.copy()
                
            return val

        # Phase 1: Differential Evolution
        # DE requires a population size. 
        # For dim=5, a population of 10-20 is reasonable.
        # Let's estimate iterations. 
        # Budget for DE: 300 calls.
        # Pop size: 12.
        # Iterations: ~25.
        
        de_budget = int(self.budget * 0.6)
        
        # DE options:
        # maxiter: limit iterations
        # popsize: population size
        # tol: convergence tolerance
        
        # We need to be careful with maxiter. 
        # In scipy DE, maxiter is the number of generations.
        # Each generation uses roughly popsize evaluations (plus some for mutations/crossovers).
        # So max_iter = de_budget // 15 is a safe estimate.
        
        max_iter_de = max(5, de_budget // 15)
        
        try:
            res_de = scipy.optimize.differential_evolution(
                wrapped_func,
                bounds=[(-5, 5)] * self.dim,
                seed=42,
                maxiter=max_iter_de,
                popsize=12,
                tol=1e-8,
                init='latinhypercube' # Good for coverage
            )
            
            if res_de.fun < self.best_f:
                self.best_f = res_de.fun
                self.best_x = res_de.x
        except Exception:
            pass

        # Phase 2: Powell's Method
        # Start from the best point found by DE.
        # Powell is very effective on Rosenbrock once in the valley.
        
        if self.calls_made < self.budget and time.time() < self.deadline:
            if self.best_x is None:
                self.best_x = np.zeros(self.dim)
            
            # Add a small perturbation to avoid exact local minima if DE got stuck?
            # Or just start at best_x. Powell handles local search well.
            x0 = self.best_x.copy()
            
            # Powell options:
            # maxiter: iterations.
            # In dim 5, Powell might take 20-50 iterations to converge tightly.
            # Each iteration involves several function calls (line searches).
            # Let's say ~10 calls per iteration on average? 
            # Remaining budget: ~200.
            # Max iter: 200 // 10 = 20.
            
            remaining_budget = self.budget - self.calls_made
            max_iter_powell = max(5, remaining_budget // 10)
            
            try:
                res_powell = scipy.optimize.minimize(
                    wrapped_func,
                    x0,
                    method='Powell',
                    options={'maxiter': max_iter_powell, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                
                if res_powell.fun < self.best_f:
                    self.best_f = res_powell.fun
                    self.best_x = res_powell.x
            except Exception:
                pass
        
        # Phase 3: Optional Polish with Nelder-Mead if budget remains
        # Sometimes Powell stops a bit early. NM can do a final tight squeeze.
        if self.calls_made < self.budget - 10 and time.time() < self.deadline:
            x0 = self.best_x.copy()
            
            remaining_budget = self.budget - self.calls_made
            max_iter_nm = max(5, remaining_budget // 6)
            
            try:
                res_nm = scipy.optimize.minimize(
                    wrapped_func,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': max_iter_nm, 'xatol': 1e-9, 'fatol': 1e-9}
                )
                
                if res_nm.fun < self.best_f:
                    self.best_f = res_nm.fun
                    self.best_x = res_nm.x
            except Exception:
                pass

        return self.best_x, self.best_f

# EVOLVE-BLOCK-END
