import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    start = time.time()
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    dim_u = n * m  # 15
    dim_v = m * p  # 25
    dim_w = p * n  # 15
    
    # Target tensor
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Construct initial rank 71 solution
    U_init = [[0] * 71 for _ in range(dim_u)]
    V_init = [[0] * 71 for _ in range(dim_v)]
    W_init = [[0] * 71 for _ in range(dim_w)]
    
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2),
    ]
    
    col = 0
    for i0, j0, k0 in blocks:
        # M1
        c = col
        U_init[i0*m + j0][c] += 1
        U_init[(i0+1)*m + (j0+1)][c] += 1
        V_init[j0*p + k0][c] += 1
        V_init[(j0+1)*p + (k0+1)][c] += 1
        W_init[k0*n + i0][c] += 1
        W_init[(k0+1)*n + (i0+1)][c] += 1
        col += 1
        # M2
        c = col
        U_init[(i0+1)*m + j0][c] += 1
        U_init[(i0+1)*m + (j0+1)][c] += 1
        V_init[j0*p + k0][c] += 1
        W_init[k0*n + (i0+1)][c] += 1
        W_init[(k0+1)*n + (i0+1)][c] += -1
        col += 1
        # M3
        c = col
        U_init[i0*m + j0][c] += 1
        V_init[j0*p + (k0+1)][c] += 1
        V_init[(j0+1)*p + (k0+1)][c] += -1
        W_init[(k0+1)*n + i0][c] += 1
        W_init[(k0+1)*n + (i0+1)][c] += 1
        col += 1
        # M4
        c = col
        U_init[(i0+1)*m + (j0+1)][c] += 1
        V_init[(j0+1)*p + k0][c] += 1
        V_init[j0*p + k0][c] += -1
        W_init[k0*n + i0][c] += 1
        W_init[k0*n + (i0+1)][c] += 1
        col += 1
        # M5
        c = col
        U_init[i0*m + j0][c] += 1
        U_init[i0*m + (j0+1)][c] += 1
        V_init[(j0+1)*p + (k0+1)][c] += 1
        W_init[k0*n + i0][c] += -1
        W_init[(k0+1)*n + i0][c] += 1
        col += 1
        # M6
        c = col
        U_init[(i0+1)*m + j0][c] += 1
        U_init[i0*m + j0][c] += -1
        V_init[j0*p + k0][c] += 1
        V_init[j0*p + (k0+1)][c] += 1
        W_init[(k0+1)*n + (i0+1)][c] += 1
        col += 1
        # M7
        c = col
        U_init[i0*m + (j0+1)][c] += 1
        U_init[(i0+1)*m + (j0+1)][c] += -1
        V_init[(j0+1)*p + k0][c] += 1
        V_init[(j0+1)*p + (k0+1)][c] += 1
        W_init[k0*n + i0][c] += 1
        col += 1

    covered = [[[False]*p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered[i0+di][j0+dj][k0+dk] = True
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered[i][j][k]:
                    U_init[i*m+j][col] += 1
                    V_init[j*p+k][col] += 1
                    W_init[k*n+i][col] += 1
                    col += 1
    
    U = np.array(U_init, dtype=float)
    V = np.array(V_init, dtype=float)
    W = np.array(W_init, dtype=float)
    
    def compute_error(U, V, W):
        T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
        return np.linalg.norm(T_decomp - T)

    best_U, best_V, best_W = U, V, W
    best_rank = U.shape[1]
    
    def als_step(U, V, W, max_iter=50, tol=1e-8):
        U_curr, V_curr, W_curr = U.copy(), V.copy(), W.copy()
        rank = U_curr.shape[1]
        
        T_U = T.reshape(dim_u, -1)
        T_V = T.transpose(1, 0, 2).reshape(dim_v, -1)
        T_W = T.transpose(2, 0, 1).reshape(dim_w, -1)
        
        prev_err = np.inf
        for it in range(max_iter):
            # Update U
            M_VW = (V_curr.T[:, :, None] * W_curr.T[:, None, :]).reshape(rank, dim_v * dim_w)
            U_curr = np.linalg.lstsq(M_VW.T, T_U.T, rcond=None)[0].T
            
            # Update V
            M_UW = (U_curr.T[:, :, None] * W_curr.T[:, None, :]).reshape(rank, dim_u * dim_w)
            V_curr = np.linalg.lstsq(M_UW.T, T_V.T, rcond=None)[0].T
            
            # Update W
            M_UV = (U_curr.T[:, :, None] * V_curr.T[:, None, :]).reshape(rank, dim_u * dim_v)
            W_curr = np.linalg.lstsq(M_UV.T, T_W.T, rcond=None)[0].T
            
            err = compute_error(U_curr, V_curr, W_curr)
            if abs(prev_err - err) < tol:
                break
            prev_err = err
            
        return U_curr, V_curr, W_curr, err

    current_rank = best_rank
    # Try to reduce rank
    for target_rank in range(current_rank - 1, max(current_rank - 10, 50), -1):
        if time.time() - start > 140:
            break
            
        # Sort by norm
        norms = np.sum(U**2, axis=0) + np.sum(V**2, axis=0) + np.sum(W**2, axis=0)
        indices = np.argsort(norms)[::-1]
        top_indices = indices[:target_rank]
        
        U_start = U[:, top_indices]
        V_start = V[:, top_indices]
        W_start = W[:, top_indices]
        
        U_opt, V_opt, W_opt, err = als_step(U_start, V_start, W_start, max_iter=100)
        
        if err < 1e-6:
            U_rounded = np.round(U_opt * 2) / 2
            V_rounded = np.round(V_opt * 2) / 2
            W_rounded = np.round(W_opt * 2) / 2
            
            err_rounded = compute_error(U_rounded, V_rounded, W_rounded)
            if err_rounded < 1e-9:
                best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                best_rank = target_rank
                U, V, W = best_U, best_V, best_W

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
