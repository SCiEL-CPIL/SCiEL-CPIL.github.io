import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Build the target tensor T
    # T[u, v, w] = 1 if u=(i,j), v=(j,k), w=(k,i)
    # u in [0, n*m), v in [0, m*p), w in [0, p*n)
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    # Function to check if a decomposition is exact
    def is_exact(U, V, W):
        # Compute sum_r U[:, r] * V[:, r] * W[:, r]
        # This is a tensor reconstruction
        # We can do this by contracting along the rank dimension
        # Reshape U to (n*m, 1, 1, r), V to (1, m*p, 1, r), W to (1, 1, p*n, r)
        # Then sum over r
        # Or simply:
        # result[u, v, w] = sum_r U[u, r] * V[v, r] * W[w, r]
        # We can use einsum
        recon = np.einsum('ur,vr,wr->uvw', U, V, W)
        # Check if recon matches T
        # Since T is 0 or 1, and we snap to half-integers, exact match should be close to 0 error
        return np.allclose(recon, T, atol=1e-9)

    # Function to snap to half-integers
    def snap_half_int(mat):
        return 0.5 * np.round(2 * mat)

    # ALS solver
    def als_solve(T, rank, n_iter=500, tol=1e-12):
        n_m, m_p, p_n = T.shape
        U = np.random.randn(n_m, rank) * 0.1
        V = np.random.randn(m_p, rank) * 0.1
        W = np.random.randn(p_n, rank) * 0.1
        
        prev_err = np.inf
        for it in range(n_iter):
            # Update U
            # T_uvw = sum_r U_ur V_vr W_wr
            # For fixed V, W, U is linear.
            # We want U such that U @ (V^T * W) approx T
            # Let's flatten V and W to form a matrix
            # V_flat = V.T (rank x m_p)
            # W_flat = W.T (rank x p_n)
            # The term for U is: U @ (V.T * W.T) where * is element-wise? No.
            # T[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
            # Let M_vw = V[v, :] * W[w, :] (outer product of rows? No, element-wise product of vectors)
            # Actually, for a fixed u, T[u, :, :] = sum_r U[u, r] (V[:, r] * W[:, r])
            # So T_u (flattened) = U[u, :] @ (V * W).T ?
            # Let B = V * W (element-wise product). Shape (m_p * p_n, rank).
            # B[idx, r] = V[idx // p_n, r] * W[idx % p_n, r]
            # Then T_flat[u, idx] = sum_r U[u, r] B[idx, r]
            # So T_flat = U @ B.T
            # U = T_flat @ B.T @ (B @ B.T)^-1
            
            B = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(m_p * p_n, rank)
            T_flat = T.reshape(n_m, m_p * p_n)
            
            # Solve U @ B.T = T_flat
            # U = T_flat @ (B.T @ B)^-1 B.T ? No, U is (n_m, rank), B.T is (rank, m_p*p_n)
            # T_flat = U @ B.T
            # U = T_flat @ B @ (B.T @ B)^-1 ?
            # Let's use lstsq
            # T_flat (n_m, m_p*p_n) = U (n_m, rank) @ B.T (rank, m_p*p_n)
            # This is equivalent to B (m_p*p_n, rank).T @ U.T = T_flat.T
            # B.T @ U.T = T_flat.T
            # U.T = (B.T)^-1 T_flat.T
            # U = (T_flat @ B) @ (B.T @ B)^-1
            try:
                U = np.linalg.lstsq(B.T, T_flat.T, rcond=None)[0].T
            except:
                pass
                
            # Update V
            A = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(n_m * p_n, rank)
            # T_flat2 is T reshaped to (m_p, n_m * p_n)
            # T[u, v, w] -> T2[v, u * p_n + w]
            T2 = T.transpose(1, 0, 2).reshape(m_p, n_m * p_n)
            try:
                V = np.linalg.lstsq(A.T, T2.T, rcond=None)[0].T
            except:
                pass
                
            # Update W
            C = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(n_m * m_p, rank)
            # T3 is T reshaped to (p_n, n_m * m_p)
            # T[u, v, w] -> T3[w, u * m_p + v]
            T3 = T.transpose(2, 0, 1).reshape(p_n, n_m * m_p)
            try:
                W = np.linalg.lstsq(C.T, T3.T, rcond=None)[0].T
            except:
                pass
            
            # Calculate error
            recon = np.einsum('ur,vr,wr->uvw', U, V, W)
            err = np.linalg.norm(recon - T)
            if abs(prev_err - err) < tol:
                break
            prev_err = err
            
        return U, V, W, err

    # Try ranks from 20 to 35
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    start_time = time.time()
    deadline = start_time + 120  # Keep 30s margin
    
    # We can try a few random restarts for each rank
    for r in range(20, 36):
        if time.time() > deadline:
            break
            
        found = False
        for restart in range(5):
            if time.time() > deadline:
                break
                
            U, V, W, err = als_solve(T, r, n_iter=200)
            
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
