"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Black-box optimizer for DTLZ1 using random restart local search.
    Strategy:
    1. Generate random starting points in the search space.
    2. For each starting point, fix the 'position' variables (first n_obj-1) and optimize the 'distance' variables (rest) to minimize the objectives.
       Since we don't know which are position/distance, we assume the standard DTLZ1 structure: 
       x[0:n_obj-1] are position, x[n_obj-1:] are distance.
       We minimize the sum of objectives or a scalarization to drive the distance variables to the front.
       Actually, for DTLZ1, minimizing g(x_d) is key. g is usually the sum of squares of distance variables.
       We can use Nelder-Mead to minimize the first objective (or sum of objectives) with respect to the distance variables, 
       keeping position variables fixed. This drives the point onto the Pareto front.
    3. Evaluate the resulting points.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # Assume standard DTLZ1 structure:
        # n_pos = n_obj - 1 (position variables)
        # n_dist = dim - n_pos (distance variables)
        n_pos = n_obj - 1
        n_dist = dim - n_pos

        # Budget allocation:
        # We have `self.budget` evaluations.
        # Each local search takes multiple evaluations.
        # Let's estimate evaluations per restart.
        # Nelder-Mead in n_dist dimensions takes roughly (n_dist + 1) * 20 evaluations for a good convergence.
        # If n_dist = 5, that's ~120 evals per restart. Too many for 200 budget.
        # We need a cheaper local search or fewer restarts.
        # Let's use a simple random hill climbing or a very limited Nelder-Mead.
        # Or, we can use the knowledge that for DTLZ1, the optimal distance variables are often at the bounds (e.g., 0).
        # But to be robust, let's do a few random steps.
        
        # Alternative: Use the fact that we have 200 evals.
        # If we do 10 restarts, we have 20 evals per restart.
        # In 5 dimensions, 20 evals is very few for Nelder-Mead.
        
        # Let's try a different approach:
        # 1. Sample random points.
        # 2. For each point, try to improve it by randomly perturbing the distance variables and keeping the better one.
        # This is a simple stochastic local search.
        
        # Number of restarts
        n_restarts = min(50, self.budget // 4)  # At least 4 evals per restart
        evals_per_restart = self.budget // n_restarts
        
        candidates = []
        
        for i in range(n_restarts):
            if time.time() > deadline - 10:
                break
                
            # Random starting point
            x = lo + (hi - lo) * np.random.rand(dim)
            
            # Fix position variables, optimize distance variables
            # We will do a simple random search on the distance variables
            best_x = x.copy()
            best_f = np.array(problem(best_x))
            evals_used = 1
            
            # Try to improve by perturbing distance variables
            for _ in range(evals_per_restart - 1):
                if time.time() > deadline - 5:
                    break
                
                # Perturb distance variables
                x_trial = best_x.copy()
                x_trial[n_pos:] += (hi[n_pos:] - lo[n_pos:]) * np.random.rand(n_dist) * 0.1
                x_trial = np.clip(x_trial, lo, hi)
                
                f_trial = np.array(problem(x_trial))
                evals_used += 1
                
                # Check if trial is better (non-dominated)
                # For simplicity, let's check if it dominates best_x or if best_x dominates trial
                # If neither, we might keep best_x to stay on the front?
                # Actually, we want to drive to the front. 
                # A simple heuristic: minimize the sum of objectives? No, that's scalarization.
                # Let's just keep the point if it is non-dominated by best_x, or if it has a smaller sum.
                # To ensure we are on the front, we should minimize the distance to the front.
                # In DTLZ1, the front is at g=0.
                # Let's assume the first objective is related to g? No, f1 = (1+g)*...
                # So minimizing f1 + f2 + f3 might not drive g to 0 if position is bad.
                
                # Better: Use the fact that we want to minimize g.
                # We don't know g. But we know that for a fixed position, the Pareto optimal point has g=0.
                # So we want to find x_d such that the resulting point is on the front.
                # A point is on the front if it is non-dominated by any other point with the same position.
                # This is hard to check without more evals.
                
                # Let's stick to a simpler heuristic:
                # Keep the trial if it dominates the current best, or if the current best does not dominate the trial and the trial has a smaller sum of objectives.
                # This tends to move towards the front.
                
                if np.all(f_trial <= best_f) and np.any(f_trial < best_f):
                    best_x = x_trial
                    best_f = f_trial
                elif not (np.all(best_f <= f_trial) and np.any(best_f < f_trial)):
                    # If neither dominates, keep the one with smaller sum (scalarization heuristic)
                    if np.sum(f_trial) < np.sum(best_f):
                        best_x = x_trial
                        best_f = f_trial

            candidates.append(best_x)
            
            # We have used evals_used evaluations.
            # We need to track total evaluations to not exceed budget.
            # The problem statement says "More than 200 evaluations ... makes the whole result invalid".
            # So we must be careful.
            # My loop above uses `evals_per_restart` evaluations per restart.
            # n_restarts * evals_per_restart <= budget.
            
        # Evaluate the final candidates? 
        # No, we already evaluated them during the search.
        # But we need to return the points? 
        # "The return value of __call__ is ignored."
        # "Every returned objective vector is recorded."
        # This implies the evaluator intercepts `problem(x)` calls.
        # So we just need to make sure we call `problem` for the points we want to score.
        # We have called `problem` for all intermediate points and the final best_x.
        # The evaluator will record all of them.
        # We don't need to re-evaluate.
        
        return candidates
# EVOLVE-BLOCK-END
