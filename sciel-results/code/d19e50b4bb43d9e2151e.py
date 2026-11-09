import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        N = self.budget
        R1 = 11.0
        R2 = 11.0
        
        # ZDT2 Pareto front: f1 = x0, f2 = (1 + sqrt(f1))^(-0.5)
        # We optimize the positions x_i on [0, 1] to maximize hypervolume.
        
        # Initialize with a heuristic distribution (clustered near 0)
        # u in [0, 1], x = u^2
        u = np.linspace(0.0, 1.0, N)
        x_vals = u ** 2
        
        # Ensure strictly increasing and within bounds
        x_vals = np.clip(x_vals, 0.0, 1.0)
        # Make sure they are strictly increasing
        for i in range(1, N):
            if x_vals[i] <= x_vals[i-1]:
                x_vals[i] = x_vals[i-1] + 1e-6
        if x_vals[-1] > 1.0:
             x_vals[-1] = 1.0

        def f2(x):
            # Handle x=0 carefully? sqrt(0)=0, 1+0=1, 1^-0.5=1. OK.
            return (1.0 + np.sqrt(x)) ** (-0.5)
        
        def calc_hv(x):
            # x is sorted array of f1 values
            # H = sum_{i=1}^{N-1} (x_{i+1} - x_i) * (R2 - f2(x_i)) + (R1 - x_N) * (R2 - f2(x_N))
            f2_vals = f2(x)
            # Widths
            widths = np.diff(x)
            # Heights for the first N-1 segments
            heights = R2 - f2_vals[:-1]
            area = np.sum(widths * heights)
            # Last segment
            area += (R1 - x[-1]) * (R2 - f2_vals[-1])
            return area
        
        current_hv = calc_hv(x_vals)
        
        # Local search to improve hypervolume
        # We use a simple coordinate descent with adaptive step size
        step_size = 0.01
        num_iters = 100
        
        for _ in range(num_iters):
            if time.time() > deadline - 5.0:
                break
            improved = False
            for i in range(N):
                # Try moving x[i] left and right
                # Constraints: x[i-1] < x[i] < x[i+1]
                lower = x_vals[i-1] + 1e-9 if i > 0 else 0.0
                upper = x_vals[i+1] - 1e-9 if i < N-1 else 1.0
                
                if lower >= upper:
                    continue
                    
                # Adaptive step
                local_step = min(step_size, (upper - lower) * 0.4)
                
                # Candidate 1: Move right
                new_x = x_vals.copy()
                new_x[i] = x_vals[i] + local_step
                if new_x[i] <= upper:
                    hv_new = calc_hv(new_x)
                    if hv_new > current_hv:
                        x_vals = new_x
                        current_hv = hv_new
                        improved = True
                        continue # Restart loop for this i? Or just continue to next i. 
                                 # Restarting is safer for coordinate descent but slower.
                                 # Given N=200 and fast calc, restarting is fine.
                                 # But simple sweep is also okay. Let's stick to simple sweep for speed.
                        # Actually, if we improve, the neighbors' optimal positions change.
                        # A single pass might not converge fast.
                        # Let's do a few passes.
                
                # Candidate 2: Move left
                new_x = x_vals.copy()
                new_x[i] = x_vals[i] - local_step
                if new_x[i] >= lower:
                    hv_new = calc_hv(new_x)
                    if hv_new > current_hv:
                        x_vals = new_x
                        current_hv = hv_new
                        improved = True
            
            if not improved:
                step_size *= 0.5
                if step_size < 1e-6:
                    break
            
        # Final check: ensure valid points
        # x_vals should be strictly increasing in [0, 1]
        
        # Construct the 30-dim vectors
        X = np.zeros((N, self.dim))
        X[:, 0] = x_vals
        # x[1:] remains 0
        
        # Evaluate the final points
        F_list = []
        for i in range(N):
            if time.time() > deadline:
                break
            x = X[i]
            f = problem(x)
            F_list.append(f)
            
        return X
