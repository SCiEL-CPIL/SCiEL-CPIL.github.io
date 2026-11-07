"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    """Hybrid LHS + Nelder-Mead Multi-objective Optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin against 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # 1. Latin Hypercube Sampling for global coverage
        # Use about 50% of budget for LHS
        n_lhs = max(10, self.budget // 2)
        
        # Generate LHS points
        # We want points in [0, 1] then scale to [lo, hi]
        # For each dimension, divide [0, 1] into n_lhs intervals
        # Pick one random point from each interval, then shuffle for each dimension
        lhs_points = np.zeros((n_lhs, dim))
        for j in range(dim):
            # Create intervals
            intervals = np.linspace(0, 1, n_lhs + 1)
            # Random point in each interval
            # np.random.uniform(low, high)
            # We can use np.random.rand(n_lhs) * (interval_width) + interval_start
            # Or simpler: np.random.uniform(0, 1, n_lhs) -> sort? No, LHS is specific.
            # Standard LHS:
            # 1. Divide [0,1] into n equal intervals.
            # 2. For each dimension, generate one random number in each interval.
            # 3. Shuffle the numbers for each dimension independently.
            
            # Generate random offsets within intervals
            # Interval i is [i/n, (i+1)/n]
            # Point = (i + u) / n where u ~ U(0,1)
            u = np.random.rand(n_lhs)
            points = (np.arange(n_lhs) + u) / n_lhs
            # Shuffle
            np.random.shuffle(points)
            lhs_points[:, j] = points
            
        # Scale to bounds
        X_lhs = lo + (hi - lo) * lhs_points
        
        # Evaluate LHS points
        F = np.zeros((n_lhs, n_obj))
        eval_count = 0
        for i in range(n_lhs):
            if time.time() > deadline:
                break
            try:
                F[i] = problem(X_lhs[i])
                eval_count += 1
            except Exception:
                F[i] = np.full(n_obj, 1e6) # Penalty
                
        # If we ran out of time or budget, return what we have
        if eval_count >= self.budget or time.time() > deadline:
             # Filter out invalid points if necessary, but problem clips so they should be valid
             # We just return the last X and F? The contract says "return value of __call__ is ignored"
             # But "Every returned objective vector is recorded". 
             # Wait, the contract says: "problem(x) ... returns 3 objective values."
             # "Every returned objective vector is recorded".
             # This implies the evaluator records the output of `problem(x)` calls.
             # So I don't need to return the points. I just need to make the calls.
             # The return value of __call__ is ignored.
             return None

        # 2. Local Refinement with Nelder-Mead on non-dominated points
        # Find non-dominated points from LHS
        dominated = np.zeros(n_lhs, dtype=bool)
        for i in range(n_lhs):
            for j in range(n_lhs):
                if i == j: continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[i] = True
                    break
        
        non_dom_idx = np.where(~dominated)[0]
        
        # Sort non-dominated points by sum of objectives (simple heuristic to pick best candidates)
        # We want to refine the most promising ones
        if len(non_dom_idx) > 0:
            sums = np.sum(F[non_dom_idx], axis=1)
            # Pick top K non-dominated points to refine
            k_refine = min(5, len(non_dom_idx))
            best_idx_in_nd = np.argsort(sums)[:k_refine]
            start_idx = non_dom_idx[best_idx_in_nd]
            
            # Remaining budget
            remaining_budget = self.budget - eval_count
            
            # Nelder-Mead requires a scalar objective. 
            # We can minimize the Tchebycheff or a weighted sum, but weights are tricky.
            # Alternatively, we can run NM on each objective separately? No, that collapses the front.
            # A common trick is to minimize a weighted sum with random weights, 
            # but that only finds specific points on the front.
            # Given the small budget left, maybe a simple local random search (hill climbing) 
            # is more robust and less prone to getting stuck in a single objective's basin.
            
            # Let's do a simple local random walk / hill climbing on the non-dominated points.
            # For each of the top k_refine points, try small perturbations.
            
            # Allocate remaining budget to these points
            steps_per_point = max(1, remaining_budget // k_refine)
            
            for idx in start_idx:
                x_current = X_lhs[idx].copy()
                f_current = F[idx].copy()
                
                for step in range(steps_per_point):
                    if eval_count >= self.budget or time.time() > deadline:
                        break
                    
                    # Random perturbation
                    # Step size decreases slightly? Or fixed small step.
                    # Let's use a step size relative to the range
                    step_size = 0.05 * (hi - lo)
                    delta = np.random.uniform(-1, 1, dim) * step_size
                    x_new = x_current + delta
                    x_new = np.clip(x_new, lo, hi)
                    
                    try:
                        f_new = problem(x_new)
                        eval_count += 1
                        
                        # Check if new point dominates current
                        if np.all(f_new <= f_current) and np.any(f_new < f_current):
                            x_current = x_new
                            f_current = f_new
                            X_lhs[idx] = x_current
                            F[idx] = f_current
                        # Check if current dominates new? If so, just discard new.
                        # We don't maintain a population here, just improving the seed point.
                        
                    except Exception:
                        pass

        return None
# EVOLVE-BLOCK-END
