"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG4 with disconnected Pareto front.
    
    Uses a 2D parameter grid for the shape variables (needed for 3-objective
    WFG4) with local coordinate descent to optimize the remaining 10 variables.
    This captures the disconnected regions of the Pareto front more effectively.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 285.0  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        X_list = []
        F_list = []
        used = 0
        
        def eval_count():
            return used
        
        def is_deadline():
            return time.time() > deadline
        
        def try_eval(x):
            """Evaluate x, increment used counter. Returns f or None if over budget."""
            nonlocal used
            if used >= self.budget or time.time() > deadline:
                return None
            x_clipped = np.clip(x, lo, hi)
            f = problem(x_clipped)
            used += 1
            return f
        
        # WFG4 with 3 objectives has a 2D Pareto front parameterized by 2 shape variables.
        # The disconnected nature means we need to sample a 2D grid of shape parameters.
        # For WFG4, the first 2 variables (x[0], x[1]) act as shape parameters,
        # and the remaining 10 variables should be optimized to minimize the "distance" term.
        
        # Sample a 2D grid of shape parameters
        # Use a moderate grid size to balance coverage and refinement budget
        n_grid = 8  # 8x8 = 64 initial points
        n_refine_per_point = 3  # 3 refinement steps per point
        
        # Shape parameter ranges - WFG4 shape variables typically in [0, 1]
        # But let's use the actual bounds
        s1_vals = np.linspace(lo[0], hi[0], n_grid)
        s2_vals = np.linspace(lo[1], hi[1], n_grid)
        
        # Phase 1: Evaluate 2D grid of shape parameters with remaining vars at midpoints
        for s1 in s1_vals:
            for s2 in s2_vals:
                if used >= self.budget or time.time() > deadline:
                    break
                
                x = np.empty(self.dim)
                x[0] = s1
                x[1] = s2
                for d in range(2, self.dim):
                    x[d] = (lo[d] + hi[d]) / 2.0
                
                f = try_eval(x)
                if f is None:
                    break
                X_list.append(x.copy())
                F_list.append(f)
        
        if not X_list:
            return np.array([])
        
        X = np.array(X_list)
        F = np.array(F_list)
        
        # Phase 2: Local coordinate descent on the remaining 10 variables for each point
        # This is critical for WFG4 because the optimal values of the non-shape
        # variables depend on the shape parameter values.
        
        remaining_budget = self.budget - used
        if remaining_budget > 0 and len(X) > 0:
            # Process points in order, spending limited budget on each
            max_steps_per_point = max(1, remaining_budget // len(X))
            max_steps_per_point = min(max_steps_per_point, 5)  # cap at 5 steps per point
            
            for idx in range(len(X)):
                if used >= self.budget or time.time() > deadline:
                    break
                
                x = X[idx].copy()
                f = F[idx].copy()
                f_sum = np.sum(f)
                
                # Coordinate descent on dimensions 2 through 11
                steps_done = 0
                for d in range(2, self.dim):
                    if used >= self.budget or time.time() > deadline or steps_done >= max_steps_per_point:
                        break
                    
                    # Try several step sizes
                    best_x = x
                    best_f = f
                    best_f_sum = f_sum
                    
                    for step_frac in [0.1, -0.1, 0.05, -0.05, 0.2, -0.2]:
                        if used >= self.budget or time.time() > deadline:
                            break
                        
                        x_trial = x.copy()
                        x_trial[d] = x[d] + step_frac * span[d]
                        
                        f_trial = try_eval(x_trial)
                        if f_trial is None:
                            break
                        
                        f_sum_trial = np.sum(f_trial)
                        
                        # Accept if it improves the scalar sum
                        if f_sum_trial < best_f_sum:
                            best_x = x_trial
                            best_f = f_trial
                            best_f_sum = f_sum_trial
                            steps_done += 1
                    
                    if best_x is not x:
                        x = best_x
                        f = best_f
                        f_sum = best_f_sum
                
                # Update with improved point
                X[idx] = x
                F[idx] = f
        
        return X
# EVOLVE-BLOCK-END
