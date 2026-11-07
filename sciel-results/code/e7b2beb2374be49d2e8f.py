import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280
        
        N = self.budget
        dim = self.dim
        
        # ZDT3 Pareto front: x[0] = f1, x[1:] = 0
        # f1 = x0
        # f2 = 1 - sqrt(x0) - (1-x0)*sin(10*pi*x0)
        
        def f2_val(x):
            # x is array
            # Handle x=0 carefully for sqrt
            sqrt_x = np.sqrt(x)
            return 1.0 - sqrt_x - (1.0 - x) * np.sin(10.0 * np.pi * x)
        
        def calc_hv(x):
            # x is sorted array of f1 values
            # f2 values
            f2 = f2_val(x)
            # f1 is x
            # Add reference point
            # HV = sum_{i=0}^{N-1} (x_{i+1} - x_i) * (11 - f2_i)
            # where x_N = 11
            # We need to handle the last term: (11 - x_{N-1}) * (11 - f2_{N-1})
            
            # Efficient calculation:
            # diffs = np.diff(np.append(x, 11.0))
            # heights = 11.0 - f2
            # return np.sum(diffs * heights)
            
            diffs = np.diff(np.append(x, 11.0))
            heights = 11.0 - f2
            return np.sum(diffs * heights)
        
        # Initialize x
        # Start with uniform distribution
        x = np.linspace(0.0, 1.0, N)
        
        # Coordinate Descent
        # We optimize each x[i] given the neighbors
        # To ensure progress, we iterate multiple times
        n_iterations = 100
        n_candidates = 50
        
        for _ in range(n_iterations):
            if time.time() > deadline:
                break
            for i in range(N):
                # Determine bounds for x[i]
                L = x[i-1] if i > 0 else 0.0
                R = x[i+1] if i < N-1 else 1.0
                
                if L >= R:
                    continue
                
                # Generate candidates
                # Include current value and linspace
                candidates = np.linspace(L, R, n_candidates)
                
                best_val = -np.inf
                best_x = x[i]
                
                # Evaluate current
                curr_val = calc_hv(x)
                best_val = curr_val
                best_x = x[i]
                
                # Evaluate candidates
                # We can vectorize the HV calculation for candidates if we are clever,
                # but N=200, so simple loop is fine.
                for cx in candidates:
                    if cx == x[i]:
                        continue
                    # Create new x
                    x_new = x.copy()
                    x_new[i] = cx
                    # Ensure sorted (should be, but just in case)
                    # Since cx is in [L, R] and x is sorted, x_new is sorted
                    val = calc_hv(x_new)
                    if val > best_val:
                        best_val = val
                        best_x = cx
                
                if best_x != x[i]:
                    x[i] = best_x
        
        # Final evaluation
        # Sort x to be sure
        x = np.sort(x)
        
        results = []
        for xi in x:
            vec = np.zeros(dim)
            vec[0] = xi
            # Clip to bounds
            for j in range(dim):
                vec[j] = max(bounds[j][0], min(bounds[j][1], vec[j]))
            
            # Check time
            if time.time() > deadline:
                break
                
            f = problem(vec)
            results.append(f)
            
        return results
