import numpy as np
import time
import sys
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
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

    num_eqs = n*n*p*p*m*m
    
    # Pre-compute indices for equations
    # Equation index: (i, k, i2, k2, j, j2)
    I = np.zeros(num_eqs, dtype=int)
    K = np.zeros(num_eqs, dtype=int)
    I2 = np.zeros(num_eqs, dtype=int)
    K2 = np.zeros(num_eqs, dtype=int)
    J = np.zeros(num_eqs, dtype=int)
    J2 = np.zeros(num_eqs, dtype=int)
    
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
                            
    # Pre-compute b
    b = np.zeros(num_eqs)
    mask = (I2 == I) & (K2 == K) & (J == J2)
    b[mask] = 1.0
    
    # Pre-compute row indices for scattering
    # The number of columns in A is (p*n)*R.
    # The scatter columns depend on R.
    
    for R in range(rank_sb - 1, 69, -1):
        if time.time() > deadline - 5.0:
            break
            
        max_trials = 50 # Increase trials since construction is faster
        found = False
        
        for trial in range(max_trials):
            if time.time() > deadline - 5.0:
                break
                
            U = np.random.randint(-1, 2, size=(n*m, R))
            V = np.random.randint(-1, 2, size=(m*p, R))
            
            # Vectorized A construction
            U_vals = U[I2*m + J, :] # shape (num_eqs, R)
            V_vals = V[J2*p + K2, :] # shape (num_eqs, R)
            Prod = U_vals * V_vals # shape (num_eqs, R)
            
            # Scatter Prod into A
            # A shape: (num_eqs, (p*n)*R)
            A = np.zeros((num_eqs, (p*n)*R))
            
            # Column indices for each (row, r)
            # Col = r * (p*n) + (K[row]*n + I[row])
            # We need to create an array of shape (num_eqs, R)
            # Cols[idx, r] = r * (p*n) + (K[idx]*n + I[idx])
            
            # Use broadcasting
            r_idx = np.arange(R)
            base_cols = K * n + I # shape (num_eqs,)
            # Cols = base_cols[:, None] + r_idx[None, :] * (p*n)
            Cols = base_cols[:, None] + r_idx[None, :] * (p*n) # shape (num_eqs, R)
            
            rows_flat = np.repeat(np.arange(num_eqs), R)
            cols_flat = Cols.flatten()
            vals_flat = Prod.flatten()
            
            A[rows_flat, cols_flat] = vals_flat
            
            try:
                w, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
                
            if len(residuals) > 0 and residuals[0] > 1e-9:
                continue
                
            w2 = w * 2
            w2_rounded = np.round(w2)
            if np.allclose(w2, w2_rounded, atol=1e-9):
                W_sol = w.reshape((p*n, R))
                
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
