import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Shapes
    dim_u = n * m  # 20
    dim_v = m * p  # 25
    dim_w = p * n  # 20
    
    # Construct the target tensor T
    # T is sparse, indices (i*m+j, j*p+k, k*n+i) are 1
    # We will work with the 2D slices for ALS
    # T_uvw: dim_u x dim_v x dim_w
    
    # Precompute the indices for the 1s
    ones_u = []
    ones_v = []
    ones_w = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                ones_u.append(u_idx)
                ones_v.append(v_idx)
                ones_w.append(w_idx)
                
    # T_uvw is a sparse tensor. 
    # For ALS, we need to project T onto the subspace spanned by current V, W.
    # Let's create the full dense tensor? 20x25x20 = 10000 elements. Small enough.
    T = np.zeros((dim_u, dim_v, dim_w))
    for u, v, w in zip(ones_u, ones_v, ones_w):
        T[u, v, w] = 1.0
        
    # Reshape T for ALS steps
    # Step 1: Update U. T is dim_u x (dim_v * dim_w). 
    # T_u = T.reshape(dim_u, dim_v * dim_w)
    # M_vw = np.kron(V.T, W.T) ? No.
    # Term r: u_r (col) * (v_r * w_r)^T (row)
    # v_r is col of V. w_r is col of W.
    # v_r * w_r is outer product? No, v_r \otimes w_r is vector of length dim_v*dim_w.
    # Let M_vw be dim_v*dim_w x R. Column r is v_r \otimes w_r.
    
    def get_kron_matrix(V, W):
        # V is dim_v x R, W is dim_w x R
        # We want column r to be v_r \otimes w_r
        # v_r is length dim_v. w_r is length dim_w.
        # v_r \otimes w_r is length dim_v * dim_w.
        # np.kron(V.T, W.T) gives (R*dim_v) x (R*dim_w)? No.
        # We can use einsum or loop.
        # M[i, r] = V[i // dim_w, r] * W[i % dim_w, r]
        R = V.shape[1]
        M = np.zeros((dim_v * dim_w, R))
        for r in range(R):
            M[:, r] = np.kron(V[:, r], W[:, r])
        return M

    def get_kron_matrix_VU(V, U):
        # For updating V. T is dim_v x (dim_u * dim_w)
        # M_uw column r is u_r \otimes w_r
        R = V.shape[1]
        M = np.zeros((dim_u * dim_w, R))
        for r in range(R):
            M[:, r] = np.kron(U[:, r], W[:, r])
        return M

    def get_kron_matrix_WU(W, U):
        # For updating W. T is dim_w x (dim_u * dim_v)
        # M_uv column r is u_r \otimes v_r
        R = W.shape[1]
        M = np.zeros((dim_u * dim_v, R))
        for r in range(R):
            M[:, r] = np.kron(U[:, r], V[:, r])
        return M

    def verify(U, V, W):
        # Check if U, V, W (snapped) exactly decompose T
        # Compute sum of u_r \otimes v_r \otimes w_r
        R = U.shape[1]
        T_rec = np.zeros((dim_u, dim_v, dim_w))
        for r in range(R):
            T_rec += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        # Check against T
        # T has 1s at specific indices, 0 elsewhere
        # Due to floating point, we check if T_rec is close to 0 or 1
        # And if it matches the pattern
        max_err = np.max(np.abs(T_rec - T))
        return max_err < 1e-6, T_rec

    def snap_half_int(X):
        return np.round(X * 2) / 2.0

    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Try ranks from 90 to 100
    # Schoolbook is 100.
    for R in range(90, 101):
        if time.time() > deadline:
            break
            
        # Try a few restarts for each R
        for restart in range(5):
            if time.time() > deadline:
                break
                
            # Initialize U, V, W
            # Use schoolbook truncated to R, plus noise
            U0 = np.zeros((dim_u, R))
            V0 = np.zeros((dim_v, R))
            W0 = np.zeros((dim_w, R))
            
            # Fill with first R schoolbook terms
            count = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if count >= R:
                            break
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        U0[u_idx, count] = 1.0
                        V0[v_idx, count] = 1.0
                        W0[w_idx, count] = 1.0
                        count += 1
                    if count >= R:
                        break
                if count >= R:
                    break
            
            # Add noise to break symmetry
            U = U0 + 0.1 * np.random.randn(dim_u, R)
            V = V0 + 0.1 * np.random.randn(dim_v, R)
            W = W0 + 0.1 * np.random.randn(dim_w, R)
            
            # ALS Iterations
            converged = False
            for it in range(20):
                if time.time() > deadline:
                    break
                    
                # Update U
                M_vw = get_kron_matrix(V, W)
                T_u = T.reshape(dim_u, dim_v * dim_w)
                # Solve U M_vw^T = T_u
                # U = T_u M_vw (M_vw^T M_vw)^-1
                try:
                    U = np.linalg.lstsq(M_vw, T_u.T, rcond=None)[0].T
                except:
                    U = U0 + 0.01 * np.random.randn(dim_u, R)
                    
                # Update V
                M_uw = get_kron_matrix_VU(V, U) # Note: V is not used in M_uw, only U, W
                # M_uw is dim_u*dim_w x R
                T_v = T.reshape(dim_v, dim_u * dim_w)
                try:
                    V = np.linalg.lstsq(M_uw, T_v.T, rcond=None)[0].T
                except:
                    V = V0 + 0.01 * np.random.randn(dim_v, R)
                    
                # Update W
                M_uv = get_kron_matrix_WU(W, U)
                T_w = T.reshape(dim_w, dim_u * dim_v)
                try:
                    W = np.linalg.lstsq(M_uv, T_w.T, rcond=None)[0].T
                except:
                    W = W0 + 0.01 * np.random.randn(dim_w, R)
                    
            # Snap and Verify
            U_snap = snap_half_int(U)
            V_snap = snap_half_int(V)
            W_snap = snap_half_int(W)
            
            is_valid, _ = verify(U_snap, V_snap, W_snap)
            if is_valid:
                # Convert to lists
                return {
                    "U": U_snap.tolist(),
                    "V": V_snap.tolist(),
                    "W": W_snap.tolist()
                }
                
    # Fallback to Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
