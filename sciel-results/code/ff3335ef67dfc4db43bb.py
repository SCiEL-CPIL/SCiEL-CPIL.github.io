import numpy as np
import time
import random

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Safety margin
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def find_decomp(target_rank, max_restarts=1000):
        """
        Finds a decomposition of rank target_rank for n x m * m x p.
        Returns dict with U, V, W or None.
        """
        # Values for U and V
        # Bias towards integers and half-integers
        vals = [-1, -0.5, 0, 0.5, 1]
        
        num_eqs = n * m * p
        num_vars_W = p * n * target_rank
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization for U and V
            # Use a mix of sparse and dense
            if random.random() < 0.5:
                # Sparse
                u_vals = [0, 0, 0, 1, -1]
                v_vals = [0, 0, 0, 1, -1]
            else:
                # Dense
                u_vals = vals
                v_vals = vals
                
            U_rand = np.array([[random.choice(u_vals) for _ in range(target_rank)] for _ in range(n * m)])
            V_rand = np.array([[random.choice(v_vals) for _ in range(target_rank)] for _ in range(m * p)])
            
            # Check if U and V are too sparse or all zero
            if np.all(U_rand == 0) or np.all(V_rand == 0):
                continue
            
            # Build linear system for W
            # Equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            A = np.zeros((num_eqs, num_vars_W))
            b = np.ones(num_eqs)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        for r in range(target_rank):
                            u_val = U_rand[i * m + j, r]
                            v_val = V_rand[j * p + k, r]
                            if u_val != 0 and v_val != 0:
                                # Variable index for W[k*n+i, r]
                                # W is (p*n, rank). Flattened: W_flat[idx] = W[k*n+i, r]
                                # Let's flatten W as [k*n+i, r] -> index (k*n+i)*rank + r
                                var_idx = (k * n + i) * target_rank + r
                                A[eq_idx, var_idx] = u_val * v_val
            
            try:
                # Solve least squares
                w_sol, residuals, rank_a, s = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                continue
            
            # Check if solution is close to half-integers
            # Round to nearest half-integer
            w_rounded = np.round(w_sol * 2) / 2.0
            
            # Verify with rounded W
            # Reconstruct U, V, W as lists for verification
            U_list = [[U_rand[i * m + j, r] for r in range(target_rank)] for i in range(n) for j in range(m)]
            V_list = [[V_rand[j * p + k, r] for r in range(target_rank)] for j in range(m) for k in range(p)]
            W_list = [[w_rounded[(k * n + i) * target_rank + r] for r in range(target_rank)] for k in range(p) for i in range(n)]
            
            if verify(U_list, V_list, W_list, target_rank):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Try decreasing ranks from 35 down to 20
    best_result = schoolbook()
    best_rank = n * m * p
    
    # We know schoolbook is 36.
    # Try to find better.
    # Start from 35, 34, ...
    for target_rank in range(35, 19, -1):
        if time.time() > deadline:
            break
        
        # Allocate more restarts for higher ranks? Or lower?
        # Lower ranks are harder, so maybe more restarts?
        # But time is limited.
        # Let's use a fixed number of restarts per rank, but adjust based on time.
        result = find_decomp(target_rank, max_restarts=500)
        if result is not None:
            best_result = result
            best_rank = target_rank
            # Continue to try lower ranks if time allows
            # But if we find a low rank, it's likely the minimum.
            # However, we should try to go lower.
            pass
    
    return best_result
# EVOLVE-BLOCK-END
