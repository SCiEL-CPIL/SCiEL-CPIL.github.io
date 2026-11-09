import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Schoolbook solution as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    # Pre-compute equation indices for verification
    num_eqs_total = n * n * p * p * m * m
    
    I = np.zeros(num_eqs_total, dtype=int)
    K = np.zeros(num_eqs_total, dtype=int)
    I2 = np.zeros(num_eqs_total, dtype=int)
    K2 = np.zeros(num_eqs_total, dtype=int)
    J = np.zeros(num_eqs_total, dtype=int)
    J2 = np.zeros(num_eqs_total, dtype=int)
    
    idx = 0
    for i in range(n):
        for k in range(p):
            for i2 in range(n):
                for k2 in range(p):
                    for j in range(m):
                        for j2 in range(m):
                            I[idx] = i
                            K[idx] = k
                            I2[idx] = i2
                            K2[idx] = k2
                            J[idx] = j
                            J2[idx] = j2
                            idx += 1
                            
    b_full = np.zeros(num_eqs_total)
    mask = (I2 == I) & (K2 == K) & (J == J2)
    b_full[mask] = 1.0
    
    U_idx = I2 * m + J
    V_idx = J2 * p + K2
    W_col_base = K * n + I

    def verify_solution(U, V, W):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        R = W.shape[1]
        r_range = np.arange(R)
        
        eqs = np.arange(num_eqs_total)
        u_rows = U_idx[eqs]
        v_rows = V_idx[eqs]
        w_cols = W_col_base[eqs]
        
        col_indices = r_range[None, :] * (p*n) + w_cols[:, None]
        u_vals = U[u_rows, :]
        v_vals = V[v_rows, :]
        prod_vals = u_vals * v_vals
        
        A_verify = np.zeros((num_eqs_total, num_eqs_total))
        row_indices = np.arange(num_eqs_total)[:, None]
        A_verify[row_indices, col_indices] = prod_vals
        
        residual = A_verify @ W.reshape(-1) - b_full
        return np.max(np.abs(residual)) < 1e-9

    # Strategy: Try structured decompositions
    # The schoolbook has rank 84. We'll try to find lower rank by:
    # 1. Random structured search with better initialization
    # 2. Local optimization from schoolbook
    
    # First, try a more systematic random search with verification
    for R in range(rank_sb - 1, 75, -1):
        if time.time() > deadline - 5.0:
            break
            
        # Limit trials based on remaining time
        remaining_time = deadline - 5.0 - time.time()
        max_trials = max(10, int(remaining_time / 2.0))  # ~2s per trial estimate
        
        for trial in range(max_trials):
            if time.time() > deadline - 5.0:
                    break
                    
            # Generate random U, V with entries in {-1, 0, 1}
            # Bias towards sparsity for better conditioning
            U = np.random.randint(-1, 2, size=(n*m, R))
            V = np.random.randint(-1, 2, size=(m*p, R))
            
            # Solve the linear system for W
            num_vars = p * n * R
            
            # Use a subset of equations for speed, then verify
            if num_vars <= num_eqs_total:
                eq_indices = np.random.choice(num_eqs_total, num_vars, replace=False)
            else:
                eq_indices = np.arange(num_eqs_total)
                
            b_sub = b_full[eq_indices]
            
            eqs = eq_indices
            u_rows = U_idx[eqs]
            v_rows = V_idx[eqs]
            w_cols = W_col_base[eqs]
            
            r_range = np.arange(R)
            col_indices = r_range[None, :] * (p*n) + w_cols[:, None]
            
            u_vals = U[u_rows, :]
            v_vals = V[v_rows, :]
            prod_vals = u_vals * v_vals
            
            A_sub = np.zeros((len(eq_indices), num_vars))
            row_indices = np.arange(len(eq_indices))[:, None]
            A_sub[row_indices, col_indices] = prod_vals
            
            try:
                # Use lstsq for robustness
                w_sol, residuals, rank, s = np.linalg.lstsq(A_sub, b_sub, rcond=None)
            except:
                continue
                
            # Check if solution is valid (integer or half-integer)
            w2 = w_sol * 2
            w2_rounded = np.round(w2)
            if not np.allclose(w2, w2_rounded, atol=1e-8):
                continue
                
            # Full verification
            W_sol = w_sol.reshape((p*n, R))
            if verify_solution(U, V, W_sol):
                if R < best_rank:
                    best_rank = R
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W_sol.copy()
                    break

    # Final polish: try to reduce rank further with local search
    if best_rank < rank_sb and time.time() < deadline - 10.0:
        # Try to eliminate columns from the found solution
        for _ in range(3):  # Try a few times
            if time.time() > deadline - 10.0:
                break
            current_rank = best_W.shape[1]
            for col_to_remove in range(current_rank):
                if time.time() > deadline - 10.0:
                    break
                # Try removing this column
                U_test = np.delete(best_U, col_to_remove, axis=1)
                V_test = np.delete(best_V, col_to_remove, axis=1)
                W_test = np.delete(best_W, col_to_remove, axis=1)
                
                if verify_solution(U_test, V_test, W_test):
                    best_U = U_test
                    best_V = V_test
                    best_W = W_test
                    best_rank -= 1
                    break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
