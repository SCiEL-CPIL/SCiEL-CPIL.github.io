import numpy as np
import time
import random
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_schoolbook = [[0] * rank_schoolbook for _ in range(n * m)]
    V_schoolbook = [[0] * rank_schoolbook for _ in range(m * p)]
    W_schoolbook = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j][col] = 1
                V_schoolbook[j * p + k][col] = 1
                W_schoolbook[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_schoolbook
    best_V = V_schoolbook
    best_W = W_schoolbook

    # We will search for ranks from 59 down to 50.
    # Vectorized approach for solving W and verification.
    for r in range(59, 49, -1):
        if time.time() > deadline:
            break
            
        # Try multiple random initializations
        # Increase attempts since vectorization is faster
        for attempt in range(1000):
            if time.time() > deadline:
                break
                
            # Generate random U and V with entries in {-1, 0, 1}
            U = np.random.randint(-1, 2, (n * m, r))
            V = np.random.randint(-1, 2, (m * p, r))
            
            W = np.zeros((p * n, r))
            valid = True
            
            # Solve for W for each (i, k)
            for i in range(n):
                for k in range(p):
                    # Build matrix A (m x r)
                    # A[j, c] = U[i*m + j, c] * V[j*p + k, c]
                    # Vectorized construction
                    U_block = U[i * m : (i + 1) * m, :]
                    V_block = V.reshape(m, p, r)[:, k, :]
                    A = U_block * V_block
                    
                    # Solve A * w = 1
                    try:
                        w, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        w_rounded = np.round(w * 2) / 2
                        if np.allclose(A @ w_rounded, np.ones(m), atol=1e-9):
                            W[k * n + i, :] = w_rounded
                        else:
                            valid = False
                            break
                    except:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Vectorized verification
                # T_{ijk} = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
                # We want T_{ijk} = 1 for all i, j, k.
                # Compute T as a tensor
                # U is (n*m, r), V is (m*p, r), W is (p*n, r)
                # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
                U_t = U.reshape(n, m, r)
                V_t = V.reshape(m, p, r)
                W_t = W.reshape(p, n, r)
                
                # Compute sum over c: U[i,j,c] * V[j,k,c] * W[k,i,c]
                # This is a contraction over c.
                # First, compute U[i,j,c] * V[j,k,c] -> (n, m, p, r)
                # This is large: 3*4*5*59 ~ 3540.
                # Then multiply by W[k,i,c] and sum over c.
                
                # Efficient way:
                # For each c, compute outer product? No.
                # Use einsum: 'ijc,jkc,kic->ijk'
                T = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t, optimize=True)
                
                if np.allclose(T, 1.0, atol=1e-9):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If no better rank found, return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
