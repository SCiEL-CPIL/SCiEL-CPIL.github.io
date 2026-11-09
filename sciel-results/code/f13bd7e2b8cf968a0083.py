import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s margin before 300s limit
        
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        eval_count = 0
        
        def safe_func(x):
            nonlocal best_x, best_f, eval_count
            if time.time() > deadline:
                raise TimeoutError
            if eval_count >= budget:
                raise BudgetExceeded
            f_val = func(x)
            eval_count += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Global Sampling
        # Use a mix of uniform and LHS-like sampling for better coverage
        n_samples = 20
        for i in range(n_samples):
            x = rng.uniform(-5, 5, n)
            safe_func(x)
            
        if best_x is None:
            return np.zeros(n), 0.0

        # Phase 2: Adaptive Coordinate Search
        # We maintain a step size for each dimension.
        # We try to move along each axis. If we find a better point, we adjust the step.
        # We use a simple line search heuristic: 
        # 1. Try x +/- step[i]
        # 2. If better, move there and increase step slightly or keep
        # 3. If worse, shrink step
        # To be more efficient, we can use a parabolic fit if we have 3 points.
        
        # Initialize steps based on initial variance or just a fixed small value
        # Given the box is [-5,5], a reasonable initial step is 0.5
        steps = np.full(n, 0.5)
        
        # We will run this until budget is exhausted or no progress is made for a while
        no_improve_count = 0
        max_no_improve = 500 # Generations (iterations of the loop)
        
        # To handle the multi-modal nature, if we stall, we might need to jump.
        # But first, let's exploit.
        
        while eval_count < budget:
            if time.time() > deadline:
                break
                
            improved = False
            for i in range(n):
                if eval_count >= budget or time.time() > deadline:
                    break
                    
                # Try moving in positive direction
                x_plus = best_x.copy()
                x_plus[i] += steps[i]
                if np.any(x_plus < -5) or np.any(x_plus > 5):
                    # Clip or skip? Clipping is safer to stay in bounds
                    x_plus = np.clip(x_plus, -5, 5)
                    
                f_plus = safe_func(x_plus)
                
                if f_plus < best_f:
                    best_f = f_plus
                    best_x = x_plus
                    improved = True
                    # If we improved, we might want to take a bigger step or continue in this direction
                    # Simple strategy: keep step same, or increase slightly if we were far
                    # Let's just keep it for now, the loop will try again
                else:
                    # Try negative direction
                    x_minus = best_x.copy()
                    x_minus[i] -= steps[i]
                    if np.any(x_minus < -5) or np.any(x_minus > 5):
                        x_minus = np.clip(x_minus, -5, 5)
                        
                    f_minus = safe_func(x_minus)
                    
                    if f_minus < best_f:
                        best_f = f_minus
                        best_x = x_minus
                        improved = True
                    else:
                        # No improvement in this direction, shrink step
                        steps[i] *= 0.7
                        if steps[i] < 1e-4:
                            steps[i] = 1e-4

            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    # Stagnation detected. 
                    # Strategy: If we are stuck, maybe we are in a local minimum.
                    # Lunacek has two global minima (basins). 
                    # If we haven't found the global one, we might be stuck.
                    # Let's try a random jump to a different region.
                    if eval_count < budget - 50:
                        # Jump to a random point
                        x_new = rng.uniform(-5, 5, n)
                        f_new = safe_func(x_new)
                        if f_new < best_f:
                            best_x = x_new.copy()
                            best_f = f_new
                            # Reset steps to explore new basin
                            steps = np.full(n, 0.5)
                            no_improve_count = 0
                        else:
                            # If the jump didn't help, we are likely in a deep local min or global.
                            # Shrink steps further to ensure we are at the bottom of the local min
                            steps *= 0.5
                            if np.all(steps < 1e-5):
                                break
                    else:
                        break
                        
        return best_x, best_f

    def _cov_to_b_d(self, C, n):
        # Kept for interface compatibility, though not used in this strategy
        try:
            L = np.linalg.cholesky(C)
            return L, np.ones(n)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            return eigvecs, np.sqrt(eigvals)
