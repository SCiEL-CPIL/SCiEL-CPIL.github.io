import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Construct the target tensor T as a flattened matrix or 3D array
    # T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    # Flatten T to a matrix for ALS?
    # T[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
    # This is a tensor train / CP decomposition.
    
    def check_solution(U, V, W):
        # Check if U, V, W exactly reconstruct T
        # T_recon[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
        # We can compute this efficiently
        # Temp = U @ V.T -> (nm, mp) ? No.
        # Let's use einsum or explicit loops for small sizes
        # T_recon = np.einsum('ur,vr,wr->uvw', U, V, W)
        T_recon = np.einsum('ur,vr,wr->uvw', U, V, W)
        if not np.allclose(T_recon, T, atol=1e-9):
            return False
        
        # Check if entries are integer or half-integer
        # Multiply by 2 and check if close to integer
        if not np.allclose(U * 2, np.round(U * 2), atol=1e-9):
            return False
        if not np.allclose(V * 2, np.round(V * 2), atol=1e-9):
            return False
        if not np.allclose(W * 2, np.round(W * 2), atol=1e-9):
            return False
            
        # Round to half-integers
        U_int = np.round(U * 2) / 2.0
        V_int = np.round(V * 2) / 2.0
        W_int = np.round(W * 2) / 2.0
        
        # Convert to lists
        return True, U_int.tolist(), V_int.tolist(), W_int.tolist()

    def try_rank(R, max_iters=1000, seed=0):
        np.random.seed(seed)
        U = np.random.randn(nm, R)
        V = np.random.randn(mp, R)
        W = np.random.randn(pn, R)
        
        # Normalize to avoid scaling issues
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        for it in range(max_iters):
            # Update U
            # T[u, :, :] = sum_r U[u, r] (V[:, r] outer W[:, r])
            # Let M_v_w = sum_r V[v, r] W[w, r] -> (mp, pn)
            # T[u, v, w] = sum_r U[u, r] M_v_w[r, ?, ?] -> No
            # T[u, v, w] = sum_r U[u, r] * (sum_s V[v, s] W[w, s] delta_{rs})? No.
            # T[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
            # For fixed r, the term is U[u, r] * (V[:, r] outer W[:, r])
            # So T[u, :, :] = sum_r U[u, r] (V[:, r] outer W[:, r])
            # This is a linear system for U[u, :].
            # Let B_r = V[:, r] outer W[:, r] -> (mp, pn)
            # T[u, :, :] = sum_r U[u, r] B_r
            # Vectorize T[u, :, :] to t_u (mp*pn,)
            # Vectorize B_r to b_r (mp*pn,)
            # t_u = sum_r U[u, r] b_r
            # This is a least squares problem: t = U @ B.T ?
            # Let B_mat be (mp*pn, R) where column r is b_r
            # Then T_flat (nm, mp*pn) = U (nm, R) @ B_mat.T (R, mp*pn)
            
            # Construct B_mat
            # B_mat[:, r] = (V[:, r] * W[:, r]).flatten() ? No.
            # V[:, r] is (mp, 1), W[:, r] is (pn, 1).
            # Outer product is (mp, pn).
            # Flatten to (mp*pn,).
            B_mat = np.zeros((mp * pn, R))
            for r in range(R):
                B_mat[:, r] = np.outer(V[:, r], W[:, r]).flatten()
            
            T_flat = T.reshape(nm, mp * pn)
            # Solve U @ B_mat.T = T_flat
            # U = T_flat @ pinv(B_mat.T)
            # Use lstsq for stability
            try:
                U_new, residuals, rank_u, s = np.linalg.lstsq(B_mat.T, T_flat.T, rcond=None)
                U_new = U_new.T
            except:
                return None
                
            # Update V
            # T[:, v, w] = sum_r (U[:, r] outer W[:, r]) * V[v, r]
            # Let C_r = U[:, r] outer W[:, r] -> (nm, pn)
            # T[:, v, w] = sum_r V[v, r] C_r
            # Vectorize T[:, v, :] to t_v (nm*pn,)
            # C_mat (nm*pn, R) where column r is c_r
            C_mat = np.zeros((nm * pn, R))
            for r in range(R):
                C_mat[:, r] = np.outer(U_new[:, r], W[:, r]).flatten()
            
            T_flat_v = T.transpose(1, 0, 2).reshape(mp, nm * pn) # (mp, nm*pn)
            try:
                V_new, residuals, rank_v, s = np.linalg.lstsq(C_mat.T, T_flat_v.T, rcond=None)
                V_new = V_new.T
            except:
                return None
                
            # Update W
            # T[u, v, w] = sum_r (U[:, r] outer V[:, r]) * W[w, r]
            # Let D_r = U[:, r] outer V[:, r] -> (nm, mp)
            # T[u, v, w] = sum_r W[w, r] D_r
            # Vectorize T[:, :, w] to t_w (nm*mp,)
            # D_mat (nm*mp, R) where column r is d_r
            D_mat = np.zeros((nm * mp, R))
            for r in range(R):
                D_mat[:, r] = np.outer(U_new[:, r], V_new[:, r]).flatten()
            
            T_flat_w = T.transpose(2, 0, 1).reshape(pn, nm * mp) # (pn, nm*mp)
            try:
                W_new, residuals, rank_w, s = np.linalg.lstsq(D_mat.T, T_flat_w.T, rcond=None)
                W_new = W_new.T
            except:
                return None
                
            U, V, W = U_new, V_new, W_new
            
            # Check convergence / exactness
            # If error is small, we might have a solution
            T_recon = np.einsum('ur,vr,wr->uvw', U, V, W)
            err = np.linalg.norm(T_recon - T)
            if err < 1e-9:
                res = check_solution(U, V, W)
                if res:
                    return res[1], res[2], res[3]
                    
        return None

    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    # Try ranks from low to high
    # Lower bound: max(nm, mp, pn) is not a bound.
    # Schoolbook is 90.
    # Let's try from 50 upwards.
    for R in range(50, 91):
        if time.time() > deadline:
            break
        # Try a few random seeds
        for seed in range(10):
            if time.time() > deadline:
                break
            res = try_rank(R, max_iters=500, seed=seed)
            if res:
                U, V, W = res
                return {"U": U, "V": V, "W": W}
                
    # Fallback to schoolbook
    rank = nm * mp
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
