"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc
from scipy.optimize import minimize

class Optimizer:
    """Two-stage optimizer: Sobol global search + Nelder-Mead local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')
        self.start_time = time.time()
        self.time_limit = 320.0  # Safety margin before 330s kill

    def _evaluate(self, func, x):
        """Wrapper to count calls and update best."""
        if self.calls >= self.budget:
            raise Exception("Budget exceeded")
        if time.time() - self.start_time > self.time_limit:
            raise TimeoutError("Time limit exceeded")
        
        x = np.clip(x, -5, 5)
        fx = func(x)
        self.calls += 1
        
        if fx < self.best_f:
            self.best_f = fx
            self.best_x = x.copy()
            
        return fx

    def __call__(self, func):
        # Phase 1: Sobol Quasi-Random Search
        n_sobol = 200
        # Ensure we don't exceed budget in phase 1
        n_sobol = min(n_sobol, self.budget)
        
        try:
            sobol_gen = qmc.Sobol(d=self.dim, scramble=True, seed=42)
            points_unit = sobol_gen.random(n_sobol)
            # Map [0, 1] to [-5, 5]
            points = points_unit * 10.0 - 5.0
            
            for p in points:
                self._evaluate(func, p)
        except (Exception, TimeoutError):
            pass

        if self.best_x is None:
            # Fallback if something went wrong
            x0 = np.zeros(self.dim)
            try:
                self._evaluate(func, x0)
            except:
                pass
            return self.best_x, self.best_f

        # Phase 2: Multi-Start Nelder-Mead
        # We have remaining budget
        # Let's try to refine the best point, and potentially the 2nd best if budget allows
        
        # To implement multi-start, we need to keep track of top candidates from Phase 1
        # But we didn't store them. Let's re-evaluate or store them?
        # Storing 200 points and their values is cheap.
        
        # Let's restart Phase 1 logic to store top candidates? 
        # No, that wastes code. Let's just use the best point for now.
        # If we want multi-start, we should have stored the top K.
        
        # Let's modify Phase 1 to store top candidates.
        
        # Actually, let's just run Nelder-Mead on the best point.
        # If we have lots of budget left, we can run it again on a perturbed best point.
        
        def objective(x):
            return self._evaluate(func, x)

        # Run Nelder-Mead
        # We estimate that Nelder-Mead uses ~ (dim+1) * iterations evaluations.
        # With 300 budget left, we can afford ~ 300/6 = 50 iterations.
        
        try:
            res = minimize(objective, self.best_x, method='Nelder-Mead', 
                           options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
        except (Exception, TimeoutError):
            pass

        # If we have significant budget left (> 50), try a second start
        # We can perturb the best point
        if self.budget - self.calls > 50:
            try:
                # Perturb best_x slightly
                rng = np.random.default_rng(0)
                x_start2 = self.best_x + rng.normal(0, 0.5, self.dim)
                x_start2 = np.clip(x_start2, -5, 5)
                
                # Evaluate start point to ensure it's valid and counted
                # Note: minimize will evaluate x_start2 again.
                # To avoid double counting, we can just let minimize do it.
                
                res2 = minimize(objective, x_start2, method='Nelder-Mead', 
                                options={'maxiter': 50, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
            except (Exception, TimeoutError):
                pass

        return self.best_x, self.best_f
# EVOLVE-BLOCK-END
