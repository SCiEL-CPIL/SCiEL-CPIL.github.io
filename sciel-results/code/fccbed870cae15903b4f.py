"""Rank-minimal exact tensor decomposition for 3x4x4 matrix multiplication."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 4, 4]
DEADLINE = time.time() + 140  # safety margin before 150s kill

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12
    
    # Build target tensor T
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Fallback: Schoolbook
    best_rank = n * m * p
    best_U = np.zeros((dim_u, best_rank))
    best_V = np.zeros((dim_v, best_rank))
    best_W = np.zeros((dim_w, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j, col] = 1
                best_V[j*p+k, col] = 1
                best_W[k*n+i, col] = 1
                col += 1

    def check_decomposition(U, V, W, tol=1e-9):
        T_rec = np.zeros((dim_u, dim_v, dim_w))
        for r in range(U.shape[1]):
            T_rec += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        return np.allclose(T_rec, T, atol=tol)

    def snap_to_half_int(arr):
        return np.round(arr * 2) / 2

    def als_step(U, V, W, iters=50):
        """
        Perform ALS iterations.
        U: (dim_u, R), V: (dim_v, R), W: (dim_w, R)
        """
        R = U.shape[1]
        # Precompute indices for building the linear system
        # We solve for U given V, W
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # Vectorize T to (dim_u, dim_v*dim_w)
        T_flat = T.reshape(dim_u, dim_v * dim_w)
        
        for _ in range(iters):
            if time.time() > DEADLINE - 10:
                break
            
            # Solve for U
            # Variables: U[a, r] for a in [0, dim_u), r in [0, R)
            # Eq for (a, b, c): sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
            # We can solve for each a independently?
            # Yes! The equation for T[a,b,c] only involves U[a,:].
            # So for each a, we have a linear system:
            # sum_r U[a,r] (V[:,r] * W[:,r])_{(b,c)} = T[a, b, c]
            # Let Z_r = V[:,r] * W[:,r] (size dim_v * dim_w)
            # Then sum_r U[a,r] Z_r[(b,c)] = T_flat[a, (b,c)]
            # This is a least squares problem: U[a,:] @ Z_flat = T_flat[a,:]
            # Where Z_flat is (R, dim_v*dim_w)
            
            Z_flat = np.zeros((R, dim_v * dim_w))
            for r in range(R):
                Z_flat[r, :] = (V[:, r] * W[:, r]).flatten()
            
            # Solve U[a, :] = argmin || U[a, :] @ Z_flat - T_flat[a, :] ||^2
            # This is (Z_flat.T @ U[a, :].T) = Z_flat.T @ T_flat[a, :].T
            # U[a, :] = (Z_flat.T @ Z_flat)^-1 Z_flat.T @ T_flat[a, :].T
            # Use lstsq for stability
            for a in range(dim_u):
                # Z_flat is (R, dim_v*dim_w). We want U[a] (size R)
                # Z_flat.T @ U[a] = T_flat[a]
                # This is underdetermined if R > dim_v*dim_w? No, R is small.
                # Z_flat.T is (dim_v*dim_w, R).
                # We want x (size R) such that A x = b.
                # A = Z_flat.T, b = T_flat[a]
                try:
                    U[a, :], _, _, _ = np.linalg.lstsq(Z_flat.T, T_flat[a], rcond=None)
                except:
                    pass

            # Solve for V
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For fixed b: sum_r V[b,r] (U[:,r] * W[:,r])_{(a,c)} = T[:, b, c]
            # Z_r = U[:,r] * W[:,r] (size dim_u * dim_w)
            Z_flat = np.zeros((R, dim_u * dim_w))
            for r in range(R):
                Z_flat[r, :] = (U[:, r] * W[:, r]).flatten()
            
            for b in range(dim_v):
                try:
                    V[b, :], _, _, _ = np.linalg.lstsq(Z_flat.T, T[:, b, :].flatten(), rcond=None)
                except:
                    pass

            # Solve for W
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For fixed c: sum_r W[c,r] (U[:,r] * V[:,r])_{(a,b)} = T[a, b, c]
            # Z_r = U[:,r] * V[:,r] (size dim_u * dim_v)
            Z_flat = np.zeros((R, dim_u * dim_v))
            for r in range(R):
                Z_flat[r, :] = (U[:, r] * V[:, r]).flatten()
            
            for c in range(dim_w):
                try:
                    W[c, :], _, _, _ = np.linalg.lstsq(Z_flat.T, T[:, :, c].flatten(), rcond=None)
                except:
                    pass
        
        return U, V, W

    # Try ranks from 47 down to 30
    found = False
    for R in range(47, 29, -1):
        if time.time() > DEADLINE - 20:
            break
        
        for restart in range(10):
            if time.time() > DEADLINE - 15:
                break
            
            # Random initialization
            U = np.random.randn(dim_u, R)
            V = np.random.randn(dim_v, R)
            W = np.random.randn(dim_w, R)
            
            # Run ALS
            U, V, W = als_step(U, V, W, iters=20)
            
            # Check if real solution is valid
            if check_decomposition(U, V, W, tol=1e-8):
                # Try to snap
                U_s = snap_to_half_int(U)
                V_s = snap_to_half_int(V)
                W_s = snap_to_half_int(W)
                
                if check_decomposition(U_s, V_s, W_s, tol=1e-9):
                    best_rank = R
                    best_U, best_V, best_W = U_s, V_s, W_s
                    found = True
                    break
                else:
                    # If real is valid but snap fails, we might still use it if we can't find integer one?
                    # But contract says integer/half-integer. 
                    # If we can't snap, we can't use this rank safely.
                    # However, if we don't find any integer solution, we return 48.
                    # Let's store the best real solution just in case, but prioritize integer.
                    pass
            
            # If we found a valid integer solution, break restart loop
            if found:
                break
        
        if found:
            break

    # If no low rank integer solution found, return schoolbook
    U_out = best_U
    V_out = best_V
    W_out = best_W

    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
