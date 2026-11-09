import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Target ranks to try, in increasing order
    # Schoolbook is n*m*p = 36.
    # We try to beat this.
    target_ranks = list(range(18, 36))
    
    best_result = None
    best_rank = n * m * p  # 36
    
    # Schoolbook decomposition as fallback
    U_school = np.zeros((n * m, best_rank))
    V_school = np.zeros((m * p, best_rank))
    W_school = np.zeros((p * n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    # Helper to check exactness with half-integers
    def check_exact(U, V, W, rank):
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        tol = 1e-6
        
        # Check half-integer property
        if not np.allclose(2 * U, np.round(2 * U), atol=tol):
            return False
        if not np.allclose(2 * V, np.round(2 * V), atol=tol):
            return False
        if not np.allclose(2 * W, np.round(2 * W), atol=tol):
            return False
            
        # Snap to half-integers
        U_r = np.round(2 * U) / 2
        V_r = np.round(2 * V) / 2
        W_r = np.round(2 * W) / 2
        
        # Reconstruct T[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c]
        # U_r is (n*m, rank) -> reshape to (n, m, rank)
        # V_r is (m*p, rank) -> reshape to (m, p, rank)
        # W_r is (p*n, rank) -> reshape to (p, n, rank)
        
        U_3d = U_r.reshape(n, m, rank)
        V_3d = V_r.reshape(m, p, rank)
        W_3d = W_r.reshape(p, n, rank)
        
        T_recon = np.zeros((n, m, p))
        # Use einsum for speed and correctness
        # T[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c]
        # Note: W is indexed [k, i, c] in the tensor definition W[k*n+i, c].
        # So W_3d[k, i, c] is correct.
        
        # We can compute this by iterating c or using einsum
        # np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        try:
            T_recon = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        except Exception:
            return False
            
        T_target = np.ones((n, m, p))
        
        if not np.allclose(T_recon, T_target, atol=tol):
            return False
            
        return True

    def snap_to_half(U, V, W):
        return np.round(2 * U) / 2, np.round(2 * V) / 2, np.round(2 * W) / 2

    # Corrected ALS Solver
    def als_decomposition(rank, max_iter=50, tol=1e-8):
        # Initialize with random half-integers to help with snapping later?
        # Or just random floats.
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        T = np.ones((n, m, p))
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
                
            # Reshape to 3D for easier indexing
            U_3d = U.reshape(n, m, rank)
            V_3d = V.reshape(m, p, rank)
            W_3d = W.reshape(p, n, rank)
            
            # Update U
            U_new_3d = np.zeros_like(U_3d)
            for i in range(n):
                for j in range(m):
                    # M_ij[c, k] = V[j, k, c] * W[k, i, c]
                    # V_3d[j, :, :] is (p, rank)
                    # W_3d[:, i, :] is (p, rank)
                    v_j = V_3d[j, :, :]  # (p, rank)
                    w_i = W_3d[:, i, :]  # (p, rank)
                    # M_ij[c, k] = v_j[k, c] * w_i[k, c]
                    # We want U[i, j, c] such that sum_c U[i, j, c] * (v_j[:, c] * w_i[:, c]) = T[i, j, :]
                    # Let x = U[i, j, :] (rank,)
                    # Let A[c, k] = v_j[k, c] * w_i[k, c] -> A is (rank, p) if we transpose v_j*w_i
                    # v_j * w_i is (p, rank). Transpose is (rank, p).
                    A = (v_j * w_i).T  # (rank, p)
                    # x @ A = T[i, j, :]
                    # A.T @ x = T[i, j, :]
                    try:
                        sol = np.linalg.lstsq(A.T, T[i, j, :], rcond=None)[0]
                        U_new_3d[i, j, :] = sol
                    except:
                        U_new_3d[i, j, :] = U_3d[i, j, :]
                        
            U = U_new_3d.reshape(n * m, rank)
            
            # Update V
            U_3d = U.reshape(n, m, rank)
            V_new_3d = np.zeros_like(V_3d)
            for j in range(m):
                for k in range(p):
                    # N_jk[c, i] = U[i, j, c] * W[k, i, c]
                    # U_3d[:, j, :] is (n, rank)
                    # W_3d[k, :, :] is (n, rank)
                    u_j = U_3d[:, j, :]  # (n, rank)
                    w_k = W_3d[k, :, :]  # (n, rank)
                    N = (u_j * w_k).T  # (rank, n)
                    # V[j, k, :] @ N = T[:, j, k]
                    # N.T @ V[j, k, :] = T[:, j, k]
                    try:
                        sol = np.linalg.lstsq(N.T, T[:, j, k], rcond=None)[0]
                        V_new_3d[j, k, :] = sol
                    except:
                        V_new_3d[j, k, :] = V_3d[j, k, :]
                        
            V = V_new_3d.reshape(m * p, rank)
            
            # Update W
            U_3d = U.reshape(n, m, rank)
            V_3d = V.reshape(m, p, rank)
            W_new_3d = np.zeros_like(W_3d)
            for k in range(p):
                for i in range(n):
                    # P_ki[c, j] = U[i, j, c] * V[j, k, c]
                    # U_3d[i, :, :] is (m, rank)
                    # V_3d[:, k, :] is (m, rank)
                    u_i = U_3d[i, :, :]  # (m, rank)
                    v_k = V_3d[:, k, :]  # (m, rank)
                    P = (u_i * v_k).T  # (rank, m)
                    # W[k, i, :] @ P = T[i, :, k]
                    # P.T @ W[k, i, :] = T[i, :, k]
                    try:
                        sol = np.linalg.lstsq(P.T, T[i, :, k], rcond=None)[0]
                        W_new_3d[k, i, :] = sol
                    except:
                        W_new_3d[k, i, :] = W_3d[k, i, :]
                        
            W = W_new_3d.reshape(p * n, rank)
            
            # Check convergence
            U_3d = U.reshape(n, m, rank)
            V_3d = V.reshape(m, p, rank)
            W_3d = W.reshape(p, n, rank)
            T_recon = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
            err = np.linalg.norm(T_recon - T)
            if err < tol:
                break
                
        return U, V, W

    # Try to find a decomposition for each rank
    for r in target_ranks:
        if time.time() > deadline:
            break
            
        if r >= best_rank:
            break
            
        # Try multiple random restarts
        found = False
        for restart in range(5): # Reduced restarts to fit time, relying on better ALS
            if time.time() > deadline:
                break
                
            U, V, W = als_decomposition(r, max_iter=30)
            
            # Snap to half-integers
            U_snap, V_snap, W_snap = snap_to_half(U, V, W)
            
            if check_exact(U_snap, V_snap, W_snap, r):
                best_rank = r
                best_result = {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                found = True
                break
            else:
                # If not exact, try a simple local search on the snapped values
                # This is a structural addition: Discrete Local Search
                U_ls = U_snap.copy()
                V_ls = V_snap.copy()
                W_ls = W_snap.copy()
                
                # Simple hill climbing: try flipping a random entry by 0.5
                # This is very slow, so we limit it
                if time.time() < deadline - 5:
                    for ls_iter in range(20):
                        if time.time() > deadline:
                            break
                        # Pick random matrix and entry
                        mat = np.random.choice([U_ls, V_ls, W_ls])
                        row = np.random.randint(mat.shape[0])
                        col = np.random.randint(mat.shape[1])
                        
                        old_val = mat[row, col]
                        mat[row, col] += 0.5
                        
                        if check_exact(U_ls, V_ls, W_ls, r):
                            best_rank = r
                            best_result = {"U": U_ls.tolist(), "V": V_ls.tolist(), "W": W_ls.tolist()}
                            found = True
                            break
                        else:
                            mat[row, col] = old_val
                            
                    if found:
                        break

    return best_result
