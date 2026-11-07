import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """Deterministic Sobol global search followed by (1+1)-ES local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        
        # State tracking
        best_x = None
        best_f = np.inf
        calls_made = 0

        # Helper to check time and update best
        def check_and_update(x, f):
            nonlocal best_x, best_f, calls_made
            if f < best_f:
                best_f = f
                best_x = x.copy()
            calls_made += 1
            return time.time() > deadline

        # Phase 1: Deterministic Global Search using Sobol Sequence
        # Allocate 40% of budget for global exploration
        n_global = int(budget * 0.4)
        
        # Generate Sobol sequence points
        # We need enough points to cover n_global evaluations
        # Sobol sequence is deterministic and space-filling
        sampler = qmc.Sobol(d=dim, scramble=True)
        # Number of points must be a power of 2 for Sobol
        n_points = 1
        while n_points < n_global:
            n_points *= 2
        
        # Generate points in [0, 1]^dim
        sobol_points = sampler.random(n_points)
        
        # Map to [-5, 5]^dim
        sobol_points = sobol_points * 10.0 - 5.0
        
        for i in range(n_points):
            if time.time() > deadline or calls_made >= budget:
                break
            x = sobol_points[i]
            fx = func(x)
            if check_and_update(x, fx):
                break
        
        if best_x is None:
            # Fallback if no points were evaluated (should not happen)
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made = 1

        # Phase 2: Local Refinement with (1+1)-ES
        # Use remaining budget for local optimization
        # (1+1)-ES is robust for low dimensions and adapts step size well
        
        # Parameters for (1+1)-ES
        sigma = 1.0  # Initial step size
        c_sigma = 1.0 / (dim ** 0.5 + 3)  # Adaptation rate for sigma
        c_c = 1.0 / (dim + 2)  # Adaptation rate for mean (if we had cumulative step, but (1+1) is simpler)
        
        # For (1+1)-ES, we just adapt sigma based on success rate
        # Simple adaptive rule: if success, increase sigma slightly; if fail, decrease
        # Or use the standard CMA-ES style sigma update if we implement a proper (1+1) with path history
        # Let's use a simpler, robust (1+1)-ES with restarts
        
        # We will run multiple restarts of (1+1)-ES from the best point found so far,
        # or from new random points if stuck.
        
        # To keep it simple and effective within the remaining budget:
        # We'll run a single (1+1)-ES instance from the global best, 
        # but if it stagnates, we restart from a new Sobol point or random point.
        
        # Actually, a better strategy for the remaining 60% is to continue with a 
        # more intensive local search. Let's use a simple (1+1)-ES with 
        # step size adaptation.
        
        x_mean = best_x.copy()
        sigma = 1.0
        
        # We'll use a window to track success rate for sigma adaptation
        # Success if new point is better than current best
        # This is a simple (1+1)-ES
        
        # To avoid getting stuck in local minima of the Büche-Rastrigin function,
        # we should restart if no improvement for a certain number of steps.
        
        stagnation_limit = 20  # Steps without improvement
        steps_since_improvement = 0
        
        while calls_made < budget and time.time() < deadline:
            # Generate candidate
            z = np.random.standard_normal(dim)
            x_trial = x_mean + sigma * z
            # Clip to box
            x_trial = np.clip(x_trial, -5, 5)
            
            f_trial = func(x_trial)
            calls_made += 1
            
            if f_trial < best_f:
                best_f = f_trial
                best_x = x_trial.copy()
                x_mean = x_trial.copy()
                steps_since_improvement = 0
                # Increase sigma slightly on success? 
                # Standard (1+1) often keeps sigma constant or adapts based on success rate.
                # Let's adapt sigma: if success, sigma *= 1.1; else sigma *= 0.9
                # But this can oscillate. 
                # Better: use a simple rule.
                sigma *= 1.05
            else:
                steps_since_improvement += 1
                sigma *= 0.95
                
            # Ensure sigma doesn't get too small or too large
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            # Check for stagnation
            if steps_since_improvement > stagnation_limit:
                # Restart from a new random point or the global best?
                # If we restart from global best, we might just loop.
                # Let's restart from a new random point to escape local minima.
                x_mean = np.random.uniform(-5, 5, dim)
                sigma = 1.0
                steps_since_improvement = 0
                # Evaluate the new start?
                # We already have a budget, so we can evaluate it.
                # But we just generated x_trial and it was worse.
                # Let's just set x_mean to a new random point and continue.
                # The next iteration will generate a trial around this new point.
                # This effectively resets the local search.
                
                # Optional: Evaluate the new start to see if it's good?
                # No, let the ES find it.
                
                if time.time() > deadline or calls_made >= budget:
                    break

        return best_x, best_f
