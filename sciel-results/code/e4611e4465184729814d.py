import time
import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = 3, 4, 4
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Dimensions
    nm = n * m  # 12
    mp = m * p  # 16
    pn = p * n  # 12
    
    # Construct the tensor T of shape (nm, mp, pn)
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i) else 0
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify(U, V, W, tol=1e-6):
        """
        Verify if U, V, W decompose T exactly.
        U: (nm, rank)
        V: (mp, rank)
        W: (pn, rank)
        """
        rank = U.shape[1]
        # Compute sum of outer products
        # This is equivalent to checking if T == np.einsum('ak,bk,ck->abc', U, V, W)
        # But to save memory/time, we can check slice by slice or use einsum
        # For rank 20, size is small.
        try:
            T_recon = np.einsum('ak,bk,ck->abc', U, V, W, optimize=True)
            if np.allclose(T_recon, T, atol=tol, rtol=0):
                return True
        except Exception:
            pass
        return False

    def round_to_half_int(X):
        """
        Round entries to nearest integer or half-integer.
        """
        return np.round(X * 2) / 2.0

    best_rank = nm * mp * pn  # Schoolbook 48
    best_U, best_V, best_W = None, None, None
    
    # Initialize schoolbook
    U_sb = np.zeros((nm, best_rank))
    V_sb = np.zeros((mp, best_rank))
    W_sb = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a, col] = 1.0
                V_sb[b, col] = 1.0
                W_sb[c, col] = 1.0
                col += 1
    
    if verify(U_sb, V_sb, W_sb):
        best_U, best_V, best_W = U_sb, V_sb, W_sb
    else:
        raise ValueError("Schoolbook verification failed")

    # Try to find lower rank decompositions
    # Known rank for 3x4x4 is 20.
    # We will try ranks from 20 to 47.
    
    def try_rank(r, max_iters=500, n_restarts=5):
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization
            scale = 1.0
            U = np.random.randn(nm, r) * scale
            V = np.random.randn(mp, r) * scale
            W = np.random.randn(pn, r) * scale
            
            # ALS iterations
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Update U
                # T[a, b, c] = sum_k U[a, k] V[b, k] W[c, k]
                # For fixed V, W, we solve for U[:, k]
                # Let M_k = V[:, k] * W[:, k] (outer product, shape mp x pn)
                # T[a, :, :] = sum_k U[a, k] M_k
                # This is a linear system for each a?
                # Actually, T is a 3-tensor. We can reshape T to (nm, mp*pn).
                # Let T_2d = T.reshape(nm, mp*pn)
                # Let M_k_2d = M_k.reshape(mp*pn)
                # T_2d[a, :] = sum_k U[a, k] M_k_2d
                # This is a least squares problem: T_2d = U @ M_mat
                # where M_mat is (r, mp*pn) with rows M_k_2d.
                # U = T_2d @ pinv(M_mat)
                
                M_mat = np.zeros((r, mp * pn))
                for k in range(r):
                    M_k = np.outer(V[:, k], W[:, k])
                    M_mat[k, :] = M_k.flatten()
                
                T_2d = T.reshape(nm, mp * pn)
                try:
                    U, residuals, rank_u, s = np.linalg.lstsq(M_mat.T, T_2d.T, rcond=None)
                    U = U.T
                except Exception:
                    pass

                # Update V
                # T[a, b, c] = sum_k U[a, k] V[b, k] W[c, k]
                # For fixed U, W, solve for V[:, k]
                # Let N_k = U[:, k] * W[:, k] (outer product, shape nm x pn)
                # T[:, b, :] = sum_k V[b, k] N_k
                # T_2d_v = T.transpose(1, 0, 2).reshape(mp, nm*pn)
                # N_mat is (r, nm*pn) with rows N_k.flatten()
                # V = T_2d_v @ pinv(N_mat)
                
                N_mat = np.zeros((r, nm * pn))
                for k in range(r):
                    N_k = np.outer(U[:, k], W[:, k])
                    N_mat[k, :] = N_k.flatten()
                
                T_2d_v = T.transpose(1, 0, 2).reshape(mp, nm * pn)
                try:
                    V, residuals, rank_v, s = np.linalg.lstsq(N_mat.T, T_2d_v.T, rcond=None)
                    V = V.T
                except Exception:
                    pass

                # Update W
                # T[a, b, c] = sum_k U[a, k] V[b, k] W[c, k]
                # For fixed U, V, solve for W[:, k]
                # Let P_k = U[:, k] * V[:, k] (outer product, shape nm x mp)
                # T[:, :, c] = sum_k W[c, k] P_k
                # T_2d_w = T.transpose(2, 0, 1).reshape(pn, nm*mp)
                # P_mat is (r, nm*mp) with rows P_k.flatten()
                # W = T_2d_w @ pinv(P_mat)
                
                P_mat = np.zeros((r, nm * mp))
                for k in range(r):
                    P_k = np.outer(U[:, k], V[:, k])
                    P_mat[k, :] = P_k.flatten()
                
                T_2d_w = T.transpose(2, 0, 1).reshape(pn, nm * mp)
                try:
                    W, residuals, rank_w, s = np.linalg.lstsq(P_mat.T, T_2d_w.T, rcond=None)
                    W = W.T
                except Exception:
                    pass

            # After ALS, round to half-integers
            U_r = round_to_half_int(U)
            V_r = round_to_half_int(V)
            W_r = round_to_half_int(W)
            
            if verify(U_r, V_r, W_r):
                return U_r, V_r, W_r, r
                
            # Also try rounding to integers
            U_i = np.round(U)
            V_i = np.round(V)
            W_i = np.round(W)
            
            if verify(U_i, V_i, W_i):
                return U_i, V_i, W_i, r

        return None, None, None, None

    # Try ranks 20, 21, ... up to 47
    for r in range(20, 48):
        if time.time() > deadline:
            break
            
        U_new, V_new, W_new, rank_found = try_rank(r)
        if rank_found is not None:
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank = rank_found
            break  # Found the lowest possible in this loop, can stop if we expect 20 to be min

    # If we didn't find anything better, return schoolbook
    if best_rank == nm * mp * pn:
        return {
            'U': best_U.tolist(),
            'V': best_V.tolist(),
            'W': best_W.tolist()
        }
    else:
        return {
            'U': best_U.tolist(),
            'V': best_V.tolist(),
            'W': best_W.tolist()
        }
