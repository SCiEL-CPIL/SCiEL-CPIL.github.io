"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START

def construct():
    """
    Returns a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 3x3 matrix by a 3x5 matrix.
    """
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    n, m, p = DIMENSIONS
    N, M, P = n, m, p  # aliases for clarity

    # --- 1. Base: Schoolbook ---
    rank_school = N * M * P  # 45
    best_rank = rank_school
    best_U = [[0] * rank_school for _ in range(N * M)]
    best_V = [[0] * rank_school for _ in range(M * P)]
    best_W = [[0] * rank_school for _ in range(P * N)]
    c = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                best_U[i * M + j][c] = 1
                best_V[j * P + k][c] = 1
                best_W[k * N + i][c] = 1
                c += 1

    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    # T[(i,j), (j2,k), (k2,i2)] = 1 if j==j2 and i==i2 and k==k2
    # W index: k*n+i, so output index c = k*n+i means i=c%n, k=c//n
    T = np.zeros((N*M, M*P, P*N), dtype=float)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                a_idx = i*M + j
                b_idx = j*P + k
                c_idx = k*N + i  # W[k*n+i]
                T[a_idx, b_idx, c_idx] = 1.0

    def try_rank(r, max_iters=300, n_restarts=5):
        """Try to find CP decomposition of rank r."""
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            # Random initialization
            U = np.random.randn(N*M, r)
            V = np.random.randn(M*P, r)
            W = np.random.randn(P*N, r)
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                # Update U: for each a, U[a,:] should satisfy
                # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # Fix V,W, solve for U:
                # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # Reshape: T[a] is (m*p, p*n), and we need T[a] = sum_r U[a,r] * outer(V[:,r], W[:,r])
                # So T[a] = sum_r U[a,r] * V[:,r] W[:,r]^T
                # This is a linear system: T[a] (flattened) = U[a,:] @ (V ⊙ W)
                # where (V ⊙ W)[r, (b,c)] = V[b,r]*W[c,r]
                
                # Khatri-Rao product of V and W: shape (m*p*p*n, r) -> transpose to (r, m*p*p*n)
                # Actually, let's do it per-index for clarity but that's slow.
                # Use: for each a, T[a] = U[a,:] @ K where K[r, (b,c)] = V[b,r]*W[c,r]
                K = np.einsum('br,cr->bcr', V, W).reshape(r, -1).T  # (m*p*p*n, r)
                # T_flat[a] = T[a].reshape(-1)
                T_flat = T.reshape(N*M, -1)  # (n*m, m*p*p*n)
                # Solve: T_flat @ K.T = U @ K @ K.T ... no
                # T_flat[a, :] @ K.T = sum_c T[a,c] K[c,r] = sum_c sum_s U[a,s] V[c_b,s] W[c_c,s] K[c,r]
                # This isn't straightforward. Let me use ALS properly.
                
                # Standard ALS: 
                # Fix V and W, update U:
                # For each a: minimize ||T[a] - sum_r U[a,r] outer(V[:,r], W[:,r])||^2
                # This gives: U[a,:] = argmin sum_{b,c} (T[a,b,c] - sum_r U[a,r] V[b,r] W[c,r])^2
                # Normal equations: U[a,:] @ (sum_{b,c} V[b,r]W[c,r] V[b,r']W[c,r']) = sum_{b,c} T[a,b,c] V[b,r']W[c,r']
                # Let M = sum_{b,c} outer(V[b,:]*W[c,:], V[b,:]*W[c,:])  -- this is r x r
                # Actually M[r1,r2] = sum_b sum_c V[b,r1]W[c,r1] V[b,r2]W[c,r2]
                # And rhs[a,r] = sum_b,c T[a,b,c] V[b,r] W[c,r]
                
                # Compute M (r x r):
                # M[r1,r2] = sum_b sum_c V[b,r1] W[c,r1] V[b,r2]W[c,r2]
                # = sum_b (sum_c V[b,r1]W[c,r1] V[b,r2]W[c,r2])
                # Let's compute: for each b,c, contribution to M is outer(v_w, v_w) where v_w[r] = V[b,r]*W[c,r]
                # M = sum_{b,c} outer(V[b,:]*W[c,:], V[b,:]*W[c,:])
                # = (V^T @ V) ⊙ (W^T @ W)? No...
                # M[r1,r2] = sum_b sum_c V[b,r1]V[b,r2] W[c,r1]W[c,r2]
                # = (sum_b V[b,r1]V[b,r2]) * (sum_c W[c,r1]W[c,r2])
                # = (V^T V)[r1,r2] * (W^T W)[r1,r2]
                M = (V.T @ V) * (W.T @ W)  # element-wise product, r x r
                rhs = np.einsum('abc,br,cr->ar', T, V, W)  # (n*m, r)
                try:
                    U = np.linalg.solve(M.T, rhs.T).T
                except np.linalg.LinAlgError:
                    U = np.linalg.lstsq(M.T, rhs.T, rcond=None)[0].T
                
                # Update V similarly:
                # Fix U and W, update V:
                # M_v[r1,r2] = sum_a sum_c U[a,r1]W[c,r1] U[a,r2]W[c,r2] = (U^T U)[r1,r2] * (W^T W)[r1,r2]
                M_v = (U.T @ U) * (W.T @ W)
                rhs_v = np.einsum('abc,ar,cr->br', T, U, W)
                try:
                    V = np.linalg.solve(M_v.T, rhs_v.T).T
                except np.linalg.LinAlgError:
                    V = np.linalg.lstsq(M_v.T, rhs_v.T, rcond=None)[0].T
                
                # Update W:
                M_w = (U.T @ U) * (V.T @ V)
                rhs_w = np.einsum('abc,ar,br->cr', T, U, V)
                try:
                    W = np.linalg.solve(M_w.T, rhs_w.T).T
                except np.linalg.LinAlgError:
                    W = np.linalg.lstsq(M_w.T, rhs_w.T, rcond=None)[0].T
                
                # Check convergence
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.linalg.norm(T - T_recon) / np.linalg.norm(T)
                if err < 1e-10:
                    return U, V, W, True
            
        return None, None, None, False
    
    # Try ranks from low to high
    # For 3x3x3, best known is 21. For 3x3x5, try starting from 18 or so.
    # Actually, let's be more systematic. The trivial rank is 45.
    # Try a range of ranks.
    
    best_result = None
    for r in range(18, 46):
        if time.time() > deadline:
            break
        U, V, W, success = try_rank(r, max_iters=200, n_restarts=3)
        if success:
            best_result = (r, U, V, W)
            break
    
    if best_result is None:
        # Fallback to schoolbook
        r = N * M * P
        U = np.zeros((N*M, r))
        V = np.zeros((M*P, r))
        W = np.zeros((P*N, r))
        col = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    U[i*M+j, col] = 1
                    V[j*P+k, col] = 1
                    W[k*N+i, col] = 1
                    col += 1
        r = N * M * P
    else:
        r, U, V, W = best_result
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
