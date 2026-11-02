import numpy as np
import time
import random

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s limit
    
    # Target tensor
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    target = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target[u_idx, v_idx, w_idx] = 1.0

    target_flat = target.flatten()
    
    # Fallback: Schoolbook (Rank 36)
    best_rank = n * m * p
    best_U = np.zeros((dim_u, best_rank), dtype=int)
    best_V = np.zeros((dim_v, best_rank), dtype=int)
    best_W = np.zeros((dim_w, best_rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j, col] = 1
                best_V[j*p+k, col] = 1
                best_W[k*n+i, col] = 1
                col += 1

    # Try to find a lower rank decomposition using Greedy Linear Solve
    # We fix U and V randomly, solve for W, check if W is "nice" (half-integer) and exact.
    
    # Known lower bounds suggest rank <= 36. 
    # We'll try ranks from 24 upwards (or lower if we dare, but 24 is safe for 3x3x4? 
    # 3x3x3 is 21. 3x3x4 is likely 24-28. Let's start at 24).
    
    for R in range(24, best_rank + 1):
        if time.time() > deadline:
            break
            
        # Number of restarts depends on time remaining
        time_remaining = deadline - time.time()
        # Estimate cost per restart: ~1ms. Allow 500 restarts if 0.5s left?
        # Let's do 100 restarts per rank, checking time frequently.
        max_restarts = 100
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Random integer U and V
            # Small integers to keep numbers manageable
            U = np.random.randint(-2, 3, size=(dim_u, R))
            V = np.random.randint(-2, 3, size=(dim_v, R))
            
            # Build linear system A * W_flat = b
            # Equation: sum_r U[i,r] * V[j,r] * W[k,r] = target[i,j,k]
            # Unknowns: W[k,r] for k in 0..dim_w-1, r in 0..R-1
            # Total unknowns: dim_w * R
            # Total equations: dim_u * dim_v * dim_w
            
            num_eqs = dim_u * dim_v * dim_w
            num_unk = dim_w * R
            
            A = np.zeros((num_eqs, num_unk))
            b = target_flat.copy()
            
            # Populate A
            # Index mapping for W: W[k, r] -> index k*R + r
            # Equation index: (i, j, k) -> idx = (i * dim_v + j) * dim_w + k
            for i in range(dim_u):
                for j in range(dim_v):
                    for k in range(dim_w):
                        eq_idx = (i * dim_v + j) * dim_w + k
                        # Sum over r: U[i,r] * V[j,r] * W[k,r]
                        for r in range(R):
                            A[eq_idx, k * R + r] = U[i, r] * V[j, r]
            
            # Solve least squares
            try:
                W_flat, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            except Exception:
                continue
                
            W = W_flat.reshape(dim_w, R)
            
            # Check if W is close to half-integers
            W_rounded = np.round(W * 2) / 2.0
            
            # Check exactness with rounded W
            # Reconstruct tensor
            T_recon = np.einsum('ir,jr,kr->ijk', U, V, W_rounded)
            
            if np.allclose(T_recon, target, atol=1e-9):
                # Valid decomposition found!
                # Convert to lists
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W_rounded.tolist()
                }

    # Fallback to schoolbook
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
