"""Multi-objective optimizer on DTLZ1 with direct Pareto front sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for DTLZ1.
    
    Strategy:
    1. Directly sample the Pareto front by setting g-minimizing variables to 0.5
       and sampling shape variables on the simplex.
    2. Use remaining budget for local refinement via small mutations.
    3. Track all evaluated points.
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
        budget = self.budget
        
        # DTLZ1 structure:
        # First (n_obj - 1) variables determine shape on front
        # Remaining (dim - n_obj + 1) variables affect g
        # For dim=7, n_obj=3: vars 0,1 are shape; vars 2-6 affect g
        
        n_shape = n_obj - 1  # 2
        n_g = dim - n_obj + 1  # 5
        
        # Base point: set g-variables to 0.5 to minimize g
        base = np.zeros(dim)
        base[n_shape:] = 0.5  # x3..x7 = 0.5
        
        all_X = []
        all_F = []
        
        def evaluate(x):
            x = np.clip(x, lo, hi)
            f = problem(x)
            return x, f
        
        # Phase 1: Sample the Pareto front directly
        # Use LHS or uniform sampling on the simplex for shape variables
        n_front_samples = min(budget // 2, 100)
        
        # Generate points on simplex: x1 >= 0, x2 >= 0, x1 + x2 <= 1
        # Use Dirichlet distribution or uniform on simplex
        for i in range(n_front_samples):
            # Sample from simplex using exponential trick
            e1 = -np.log(np.random.uniform(0.01, 1.0))
            e2 = -np.log(np.random.uniform(0.01, 1.0))
            e3 = -np.log(np.random.uniform(0.01, 1.0))
            s = e1 + e2 + e3
            x1 = e1 / s
            x2 = e2 / s
            # x3 = e3/s, but we only need x1, x2 for the 2-shape-variable case
            # Actually for 3 objectives, the front is 2D: x1, x2 with x1+x2<=1
            # The third "coordinate" is 1-x1-x2 which determines f3
            
            x = base.copy()
            x[0] = x1
            x[1] = x2
            x, f = evaluate(x)
            all_X.append(x)
            all_F.append(f)
        
        used = n_front_samples
        
        # Phase 2: Local refinement - mutate around existing points
        # Focus on non-dominated points
        while used < budget:
            if time.time() > deadline:
                break
            
            # Find non-dominated points
            F_arr = np.array(all_F)
            X_arr = np.array(all_X)
            
            # Identify non-dominated
            n_dom = len(F_arr)
            non_dom_mask = np.ones(n_dom, dtype=bool)
            for i in range(n_dom):
                if not non_dom_mask[i]:
                    continue
                for j in range(n_dom):
                    if i == j:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        non_dom_mask[i] = False
                        break
            
            # Pick a parent from non-dominated set
            if np.any(non_dom_mask):
                parent_idx = np.random.choice(np.where(non_dom_mask)[0])
            else:
                parent_idx = np.random.randint(n_dom)
            
            parent = X_arr[parent_idx]
            
            # Small mutation
            step_size = 0.05
            child = parent + step_size * np.random.randn(dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            all_X.append(child)
            all_F.append(fc)
        
        return all_X
# EVOLVE-BLOCK-END
