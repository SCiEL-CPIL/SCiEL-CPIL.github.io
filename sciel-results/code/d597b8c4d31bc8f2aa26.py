import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
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

    # Precompute equation indices
    # Equations indexed by (i, k, i2, k2, j, j2)
    eq_i, eq_k, eq_i2, eq_k2, eq_j, eq_j2 = np.meshgrid(
        range(n), range(p), range(n), range(p), range(m), range(m),
        indexing='ij'
    )
    eq_i = eq_i.ravel()
    eq_k = eq_k.ravel()
    eq_i2 = eq_i2.ravel()
    eq_k2 = eq_k2.ravel()
    eq_j = eq_j.ravel()
    eq_j2 = eq_j2.ravel()
    
    num_eqs = len(eq_i)
    
    # Precompute b
    b = np.zeros(num_eqs)
    b[(eq_i == eq_i2) & (eq_k == eq_k2) & (eq_j == eq_j2)] = 1.0
    
    # Precompute column offset for W position: k*n + i
    w_col_offset = eq_k * n + eq_i  # shape (num_eqs,)
    
    # Precompute U and V row indices
    u_row_idx = eq_i2 * m + eq_j  # shape (num_eqs,)
    v_row_idx = eq_j2 * p + eq_k2  # shape (num_eqs,)
    
    # Search for lower rank
    for R in range(rank_sb - 1, 69, -1):
        if time.time() > deadline - 5.0:
            break
        
        max_trials = 100
        found = False
        
        for trial in range(max_trials):
            if time.time() > deadline - 5.0:
                break
            
            U = np.random.randint(-1, 2, size=(n*m, R))
            V = np.random.randint(-1, 2, size=(m*p, R))
            
            # Build A vectorized
            # A shape: (num_eqs, 21*R)
            # For each r: A[:, r*21 + w_col_offset] = U[u_row_idx, r] * V[v_row_idx, r]
            A = np.zeros((num_eqs, 21 * R))
            
            # Get the relevant U and V values for all equations
            # U[u_row_idx, :] has shape (num_eqs, R)
            # V[v_row_idx, :] has shape (num_eqs, R)
            U_vals = U[u_row_idx, :]  # (num_eqs, R)
            V_vals = V[v_row_idx, :]  # (num_eqs, R)
            coeffs = U_vals * V_vals  # (num_eqs, R), element-wise
            
            # Now scatter: for each r, A[:, r*21 + w_col_offset] = coeffs[:, r]
            # We can do this by creating a 3D array and reshaping
            # Or loop over R (which is ~70-85, acceptable)
            for r in range(R):
                A[:, r * 21 + w_col_offset] = coeffs[:, r]
            
            # Solve
            try:
                w, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
                
            if len(residuals) > 0 and residuals[0] > 1e-9:
                continue
            
            # Check half-integer
            w2 = w * 2
            w2_rounded = np.round(w2)
            if np.allclose(w2, w2_rounded, atol=1e-9):
                W_sol = w.reshape((21, R))
                if R < best_rank:
                    best_rank = R
                    best_U = U
                    best_V = V
                    best_W = W_sol
                    found = True
                    break
        
        if found:
            pass
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
