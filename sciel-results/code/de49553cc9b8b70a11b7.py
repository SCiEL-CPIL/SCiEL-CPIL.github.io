import numpy as np
import scipy.optimize
import time

class Optimizer:
    """
    Differential Evolution + L-BFGS-B optimizer for rotated Rosenbrock (f9).
    Uses DE for global exploration to handle unknown rotation/shift,
    followed by high-precision L-BFGS-B polishing for local convergence.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0
        self.best_x = None
        self.best_f = float('inf')
        self.start_time = None

    def __call__(self, func):
        self.start_time = time.time()
        self.eval_count = 0
        self.best_x = np.zeros(self.dim)
        self.best_f = float('inf')

        # Custom exception to stop optimization early
        class BudgetExceeded(Exception):
            pass

        def wrapped_func(x):
            # Check time limit with safety margin
            if time.time() - self.start_time > 280:
                raise BudgetExceeded()
            
            # Check call budget
            if self.eval_count >= self.budget:
                raise BudgetExceeded()

            # Clip input to box [-5, 5]
            x = np.clip(x, -5, 5)
            
            self.eval_count += 1
            try:
                fx = float(func(x))
            except Exception:
                fx = 1e10  # Large penalty for errors

            # Update global best
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            
            return fx

        # Phase 1: Differential Evolution for global exploration
        # DE uses population-based diversity to handle unknown rotation/shift
        # without relying on random restarts
        
        # Reserve budget for polishing
        de_budget = int(self.budget * 0.75)  # Use 75% of budget for DE
        polish_budget = self.budget - de_budget
        
        # Track evaluations during DE
        de_eval_count = 0
        
        def de_func(x):
            nonlocal de_eval_count
            if de_eval_count >= de_budget:
                raise BudgetExceeded()
            if time.time() - self.start_time > 260:
                raise BudgetExceeded()
            
            de_eval_count += 1
            return wrapped_func(x)
        
        # DE parameters
        # Population size: 10 * dim = 50 individuals
        # Max iterations: de_budget / pop_size
        pop_size = 10 * self.dim
        max_iter_de = max(5, int(de_budget / pop_size))
        
        # Initial population bounds
        bounds = [(-5, 5)] * self.dim
        
        try:
            result_de = scipy.optimize.minimize(
                de_func,
                np.zeros(self.dim),
                method='differential_evolution',
                bounds=bounds,
                options={
                    'maxiter': max_iter_de,
                    'popsize': pop_size,
                    'tol': 1e-8,
                    'mutate': (0.5, 1.0),  # (F, CR)
                    'recombination': 0.9,
                    'init': 'latinhypercube',
                    'seed': 42,  # Deterministic
                }
            )
        except (BudgetExceeded, Exception):
            # If DE fails or runs out of budget, use current best
            pass
        
        # Phase 2: Polish with L-BFGS-B using remaining budget
        # Start from the best point found by DE
        x_start = self.best_x.copy()
        
        # Check if we have enough budget for polishing
        if self.budget - self.eval_count > 5 and time.time() - self.start_time < 270:
            try:
                result_polish = scipy.optimize.minimize(
                    wrapped_func,
                    x_start,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': 100,
                        'ftol': 1e-14,
                        'gtol': 1e-10,
                        'maxls': 50
                    }
                )
            except (BudgetExceeded, Exception):
                pass
        
        return self.best_x, self.best_f
