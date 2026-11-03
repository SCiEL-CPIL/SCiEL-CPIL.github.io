import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    # Build the target tensor T
    # T[u, v, w] = 1 if u=(i,j), v=(j,k), w=(k,i)
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    def snap_half_int(mat):
        return 0.5 * np.round(2 * mat)

    def is_exact(U, V, W):
        recon = np.einsum('ur,vr,wr->uvw', U, V, W)
        return np.allclose(recon, T, atol=1e-9)

    def is_exact_frac(U, V, W):
        # Use fractions for exact check
        Uf = [[Fraction(x).limit_denominator(10000) for x in row] for row in U]
        Vf = [[Fraction(x).limit_denominator(10000) for x in row] for row in V]
        Wf = [[Fraction(x).limit_denominator(10000) for x in row] for row in W]
        
        n_m, m_p, p_n = T.shape
        # Check a subset of entries for speed, but ideally all
        # Actually, let's just check the max error with fractions if possible
        # But fractions are slow. Let's stick to float check with tight tolerance for speed,
        # and maybe a final exact check on the best candidate.
        # For now, is_exact is sufficient if we snap to half-integers.
        return is_exact(U, V, W)

    # ALS Solver with better initialization and regularization
    def als_solve(T, rank, n_iter=1000, tol=1e-14, reg=1e-8, seed=None):
        n_m, m_p, p_n = T.shape
        rng = np.random.default_rng(seed)
        
        # Initialize with random normal
        U = rng.normal(0, 1, (n_m, rank))
        V = rng.normal(0, 1, (m_p, rank))
        W = rng.normal(0, 1, (p_n, rank))
        
        prev_err = np.inf
        for it in range(n_iter):
            if time.time() > deadline:
                break
                
            # Update U
            # T_flat = U @ B.T
            B = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(m_p * p_n, rank)
            T_flat = T.reshape(n_m, m_p * p_n)
            # Solve U @ B.T = T_flat
            # U = T_flat @ B @ (B.T @ B + reg*I)^-1
            try:
                BTB = B.T @ B + reg * np.eye(rank)
                U = T_flat @ B @ np.linalg.inv(BTB)
            except:
                U = np.linalg.lstsq(B.T, T_flat.T, rcond=None)[0].T
                
            # Update V
            A = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(n_m * p_n, rank)
            T2 = T.transpose(1, 0, 2).reshape(m_p, n_m * p_n)
            try:
                ATA = A.T @ A + reg * np.eye(rank)
                V = T2 @ A @ np.linalg.inv(ATA)
            except:
                V = np.linalg.lstsq(A.T, T2.T, rcond=None)[0].T
                
            # Update W
            C = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(n_m * m_p, rank)
            T3 = T.transpose(2, 0, 1).reshape(p_n, n_m * m_p)
            try:
                CTC = C.T @ C + reg * np.eye(rank)
                W = T3 @ C @ np.linalg.inv(CTC)
            except:
                W = np.linalg.lstsq(C.T, T3.T, rcond=None)[0].T
            
            # Calculate error
            recon = np.einsum('ur,vr,wr->uvw', U, V, W)
            err = np.linalg.norm(recon - T)
            if abs(prev_err - err) < tol:
                break
            prev_err = err
            
        return U, V, W, err

    # Try ranks from 16 to 25
    # Known lower bound is 16 (Strassen-like or specific for 3x3x4?)
    # Actually, for 3x3x3 it's 16. For 3x3x4, it might be different.
    # Let's search 15 to 25.
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None

    # Use multiple seeds for diversity
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
    
    for r in range(15, 26):
        if time.time() > deadline - 5:
            break
            
        found = False
        for seed in seeds:
            if time.time() > deadline - 5:
                break
                
            U, V, W, err = als_solve(T, r, n_iter=300, seed=seed)
            
            # Snap to half-integers
            U_snap = snap_half_int(U)
            V_snap = snap_half_int(V)
            W_snap = snap_half_int(W)
            
            if is_exact(U_snap, V_snap, W_snap):
                best_rank = r
                best_U = U_snap
                best_V = V_snap
                best_W = W_snap
                found = True
                break
        
        if found:
            break

    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Return schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
