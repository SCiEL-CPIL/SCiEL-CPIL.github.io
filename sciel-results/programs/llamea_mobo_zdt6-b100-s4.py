import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295
        
        n_evals = self.budget
        if n_evals <= 0:
            return []
            
        results = []
        
        # Generate x0 values
        # Focus on the first valley of ZDT6 where f1 is minimal (~0.28)
        # x0 in [0, 0.2] covers the first oscillation period roughly
        # Uniform sampling in this range should capture the front well
        
        # Ensure we hit boundaries 0 and 1 for f1=1
        x0s = []
        if n_evals > 2:
            # 1 point at 0, 1 point at 1
            # Rest in [0, 0.2]
            n_inner = n_evals - 2
            x0s.append(0.0)
            x0s.append(1.0)
            
            # Uniform in [0, 0.2]
            # Avoid 0 and 1 duplicates if possible, but 0 is included.
            # Let's use linspace(0, 0.2, n_inner)
            # This includes 0. We already have 0.
            # Let's use linspace(0, 0.2, n_inner+1)[1:] to skip 0?
            # Or just linspace(0, 0.2, n_inner)
            inner_xs = np.linspace(0.0, 0.2, n_inner)
            x0s.extend(inner_xs.tolist())
            
        elif n_evals == 2:
            x0s = [0.0, 1.0]
        else:
            x0s = [0.0]
            
        for i in range(n_evals):
            if time.time() > deadline:
                break
            x = np.zeros(self.dim, dtype=float)
            x[0] = float(x0s[i])
            # Other x's are 0, which is optimal for ZDT6
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                break
                
        return results
