import time
import sys
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    target_diffs = set(range(1, 31))
    
    # Lower bound for M
    # A Sidon set of size k has at most ~sqrt(2M) elements? 
    # Actually, if max is M, size k satisfies k(k-1)/2 <= M.
    # We need to cover 30 differences.
    # If we have k elements, we have k(k-1)/2 differences.
    # We need at least 30 distinct differences from the set {1..30}.
    # So k(k-1)/2 >= 30 => k >= 9.
    # For k=9, minimal M is roughly 9^2 = 81? 
    # Let's check known values. 
    # Perfect difference sets exist for k = n^2+n+1 with diameter n^2+n.
    # For n=2, k=7, diameter 6? No.
    # Let's just iterate M from 50 upwards.
    
    for M in range(50, 1000):
        if time.time() > deadline:
            break
            
        # Variables:
        # x_0 ... x_M  (M+1 vars)
        # z_{i,d} for d=1..M, i=0..M-d
        # Total z vars = sum_{d=1}^M (M-d+1) = M(M+1)/2
        
        num_x = M + 1
        num_z = M * (M + 1) // 2
        
        total_vars = num_x + num_z
        
        # Objective: minimize sum(x)
        c = np.zeros(total_vars)
        c[:num_x] = 1.0
        
        # Bounds
        lb = np.zeros(total_vars)
        ub = np.ones(total_vars)
        # x_0 must be 1
        lb[0] = 1.0
        ub[0] = 1.0
        
        # Integrality: all variables are binary
        integrality = np.ones(total_vars)
        
        # Constraints
        # We build linear constraints A_ub @ x <= b_ub
        
        # Let's use LinearConstraint for clarity, but building sparse matrix is faster.
        # We will build the matrix row by row.
        
        # Rows:
        # 1. z_{i,d} - x_i <= 0
        # 2. z_{i,d} - x_{i+d} <= 0
        # 3. -z_{i,d} + x_i + x_{i+d} <= 1  (equivalent to z >= x_i + x_{i+d} - 1)
        # 4. sum(z_{i,d}) = 1 for d in 1..30
        # 5. sum(z_{i,d}) <= 1 for d in 31..M
        
        # Total rows:
        # 3 * num_z + M (for sums)
        num_rows = 3 * num_z + M
        
        A = np.zeros((num_rows, total_vars))
        b_ub = np.zeros(num_rows)
        b_lb = np.zeros(num_rows)
        
        row_idx = 0
        z_var_offset = num_x
        
        # Map (i, d) to z variable index
        # z_{i,d} is the (i)th variable in the block for difference d.
        # Block for d has size M-d+1.
        # Index of block d in z-variables: sum_{k=1}^{d-1} (M-k+1)
        
        def get_z_idx(i, d):
            # Number of z vars for diffs < d
            # sum_{k=1}^{d-1} (M - k + 1)
            # = sum_{j=M-d+2}^{M} j
            # = (M-d+2 + M) * (d-1) / 2
            # Let's precompute offsets for speed
            pass # We'll compute in loop
            
        # Precompute z offsets for each d
        z_offsets = [0] * (M + 1)
        offset = 0
        for d in range(1, M + 1):
            z_offsets[d] = offset
            offset += M - d + 1
            
        for d in range(1, M + 1):
            num_pairs = M - d + 1
            base_z = z_offsets[d] + z_var_offset
            
            for i in range(num_pairs):
                # z_idx
                z_idx = base_z + i
                x_i_idx = i
                x_j_idx = i + d
                
                # Constraint 1: z_{i,d} - x_i <= 0
                A[row_idx, z_idx] = 1.0
                A[row_idx, x_i_idx] = -1.0
                b_ub[row_idx] = 0.0
                row_idx += 1
                
                # Constraint 2: z_{i,d} - x_{i+d} <= 0
                A[row_idx, z_idx] = 1.0
                A[row_idx, x_j_idx] = -1.0
                b_ub[row_idx] = 0.0
                row_idx += 1
                
                # Constraint 3: -z_{i,d} + x_i + x_{i+d} <= 1
                A[row_idx, z_idx] = -1.0
                A[row_idx, x_i_idx] = 1.0
                A[row_idx, x_j_idx] = 1.0
                b_ub[row_idx] = 1.0
                row_idx += 1
            
            # Constraint 4/5: Sum of z_{i,d}
            if d <= 30:
                # Sum = 1
                for i in range(num_pairs):
                    A[row_idx, base_z + i] = 1.0
                b_ub[row_idx] = 1.0
                b_lb[row_idx] = 1.0
            else:
                # Sum <= 1
                for i in range(num_pairs):
                    A[row_idx, base_z + i] = 1.0
                b_ub[row_idx] = 1.0
                b_lb[row_idx] = -np.inf
            row_idx += 1
            
        # Create LinearConstraint
        # A @ vars >= b_lb and A @ vars <= b_ub
        # Note: scipy milp uses A_ub @ x <= b_ub and A_eq @ x == b_eq
        # LinearConstraint allows both bounds.
        
        lc = LinearConstraint(A, b_lb, b_ub)
        bounds = Bounds(lb, ub)
        
        # Solve
        try:
            res = milp(c, constraints=lc, integrality=integrality, bounds=bounds)
        except Exception:
            continue
            
        if res.status == 0: # Optimal
            x_vals = res.x[:num_x]
            sol = [i for i in range(M + 1) if x_vals[i] > 0.5]
            
            # Verify solution
            if 0 not in sol:
                continue
            if max(sol) != M:
                # This can happen if the solver finds a solution with max < M? 
                # But we iterate M upwards. If a solution with max < M existed, 
                # we would have found it in a previous iteration.
                # However, if the solver is weak, it might return a valid set with max < M.
                # That's fine, we can just return it.
                pass
                
            diffs = set()
            valid_sidon = True
            for i in range(len(sol)):
                for j in range(i + 1, len(sol)):
                    d = sol[j] - sol[i]
                    if d in diffs:
                        valid_sidon = False
                        break
                    diffs.add(d)
                if not valid_sidon:
                    break
                    
            if not valid_sidon:
                continue
                
            if not target_diffs.issubset(diffs):
                continue
                
            if len(sol) > 32:
                continue
                
            return sol
            
    # Fallback
    return [0]

