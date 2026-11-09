def construct():
        import sys
        import numpy as np
        from math import gcd
        
        N = 256
        # 1. Try MIP
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            from scipy.sparse import csr_matrix
            
            # Build conflict pairs
            conflicts = []
            for i in range(1, N+1):
                for j in range(i+1, N+1):
                    if gcd(i, j) > 1:
                        conflicts.append((i-1, j-1)) # 0-indexed for matrix
            
            if not conflicts:
                 return list(range(1, N+1))
            
            # Number of constraints
            M = len(conflicts)
            # Number of variables
            n_vars = N
            
            # Build sparse matrix
            rows = []
            cols = []
            data = []
            for k, (i, j) in enumerate(conflicts):
                rows.append(k)
                cols.append(i)
                data.append(1)
                rows.append(k)
                cols.append(j)
                data.append(1)
                
            A = csr_matrix((data, (rows, cols)), shape=(M, n_vars))
            
            # Objective: maximize sum(i * x_i)
            # milp minimizes c @ x, so we use c = -[1, 2, ..., 256]
            c = -np.arange(1, N+1)
            
            # Constraints: x_i + x_j <= 1
            constraints = LinearConstraint(A, ub=1)
            
            # Bounds: 0 <= x_i <= 1
            bounds = Bounds(lb=0, ub=1)
            
            # Integrality: all integer
            integrality = np.ones(n_vars)
            
            res = milp(c, constraints=constraints, bounds=bounds, integrality=integrality)
            
            if res.success:
                # Extract solution
                # res.x is the solution vector
                selected = [i+1 for i, val in enumerate(res.x) if val > 0.5]
                # Verify strictly
                # Check types
                return [int(x) for x in selected]
            else:
                raise RuntimeError("MIP failed")
                
        except Exception as e:
            # Fallback to Heuristic
            pass
            
        # Heuristic: Greedy
        selected = []
        for i in range(N, 0, -1):
            is_coprime = True
            for s in selected:
                if gcd(i, s) != 1:
                    is_coprime = False
                    break
            if is_coprime:
                selected.append(i)
                
        # Local Search Improvement
        # Try swapping or adding
        # This is a basic improvement
        improved = True
        while improved:
            improved = False
            current_sum = sum(selected)
            # Try to add any number not in selected
            for i in range(1, N+1):
                if i in selected: continue
                if all(gcd(i, s) == 1 for s in selected):
                    selected.append(i)
                    improved = True
                    break
            # Try swapping one out for multiple in? Or just simple swaps
            # For simplicity, just the add loop above is often enough for this structure
            # But let's do a simple 1-for-1 swap check if time permits
            # Given the "largest first" greedy is strong, simple add is good.
            
        return [int(x) for x in selected]
