import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    # Build the target tensor T
    T = np.zeros((size_u, size_v, size_w), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def check_decomposition(U, V, W, tol=1e-9):
        R = np.einsum('ar,br,cr->abc', U, V, W)
        if not np.allclose(R, T, atol=tol):
            return False
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        R_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        if np.allclose(R_r, T, atol=1e-9):
            return True
        return False

    def snap_to_half_integers(U, V, W):
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        return U_r, V_r, W_r

    def verify_and_return(U, V, W):
        U_r, V_r, W_r = snap_to_half_integers(U, V, W)
        if check_decomposition(U_r, V_r, W_r):
            return {
                "U": U_r.tolist(),
                "V": V_r.tolist(),
                "W": W_r.tolist()
            }
        return None

    def find_decomposition(rank, max_attempts=20, max_iter=100):
        if time.time() > deadline:
            return None
            
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                return None
                
            # Sparse initialization: many zeros, some +/- 1
            U = np.zeros((size_u, rank))
            V = np.zeros((size_v, rank))
            W = np.zeros((size_w, rank))
            
            # Randomly place non-zeros
            for r in range(rank):
                # Each vector should have a few non-zeros
                num_nz_u = np.random.randint(2, 5)
                num_nz_v = np.random.randint(2, 5)
                num_nz_w = np.random.randint(2, 5)
                
                idx_u = np.random.choice(size_u, num_nz_u, replace=False)
                idx_v = np.random.choice(size_v, num_nz_v, replace=False)
                idx_w = np.random.choice(size_w, num_nz_w, replace=False)
                
                U[idx_u, r] = np.random.choice([-1, 1, -0.5, 0.5], num_nz_u)
                V[idx_v, r] = np.random.choice([-1, 1, -0.5, 0.5], num_nz_v)
                W[idx_w, r] = np.random.choice([-1, 1, -0.5, 0.5], num_nz_w)
            
            # ALS iterations
            prev_err = np.inf
            for it in range(max_iter):
                if time.time() > deadline - 10:
                    return None
                    
                # Update U
                M_vw = np.einsum('br,cr->bcr', V, W).reshape(size_v * size_w, rank)
                T_1 = T.reshape(size_u, size_v * size_w)
                U, _, _, _ = np.linalg.lstsq(M_vw.T, T_1.T, rcond=None)
                U = U.T
                
                # Update V
                M_ua = np.einsum('ar,cr->acr', U, W).reshape(size_u * size_w, rank)
                T_2 = T.transpose(1, 0, 2).reshape(size_v, size_u * size_w)
                V, _, _, _ = np.linalg.lstsq(M_ua.T, T_2.T, rcond=None)
                V = V.T
                
                # Update W
                M_uv = np.einsum('ar,br->abr', U, V).reshape(size_u * size_v, rank)
                T_3 = T.transpose(0, 2, 1).reshape(size_w, size_u * size_v)
                W, _, _, _ = np.linalg.lstsq(M_uv.T, T_3.T, rcond=None)
                W = W.T
                
                # Compute error
                R = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.linalg.norm(R - T)
                
                if abs(prev_err - err) < 1e-10:
                    break
                prev_err = err
            
            # Try to verify and snap
            res = verify_and_return(U, V, W)
            if res is not None:
                return res
                
            # If not exact, try a few local discrete tweaks
            # This is a simple heuristic: snap, then try flipping single values
            U_r, V_r, W_r = snap_to_half_integers(U, V, W)
            
            # Check if close
            R_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
            err_r = np.linalg.norm(R_r - T)
            
            if err_r < 1.0: # Close enough to try discrete repair
                # Try flipping one element in U, V, or W at a time
                for _ in range(50): # Limit repair steps
                    if time.time() > deadline - 5:
                        return None
                    mat_choice = np.random.choice(3)
                    if mat_choice == 0:
                        idx = np.random.choice(size_u * rank)
                        i, r = idx // rank, idx % rank
                        U_r[i, r] = -U_r[i, r] if U_r[i, r] != 0 else 1
                    elif mat_choice == 1:
                        idx = np.random.choice(size_v * rank)
                        i, r = idx // rank, idx % rank
                        V_r[i, r] = -V_r[i, r] if V_r[i, r] != 0 else 1
                    else:
                        idx = np.random.choice(size_w * rank)
                        i, r = idx // rank, idx % rank
                        W_r[i, r] = -W_r[i, r] if W_r[i, r] != 0 else 1
                        
                    R_check = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                    if np.allclose(R_check, T, atol=1e-9):
                        return {
                            "U": U_r.tolist(),
                            "V": V_r.tolist(),
                            "W": W_r.tolist()
                        }
                        
        return None

    best_result = None
    # Try ranks from 23 upward
    for rank in [23, 24, 25, 26, 27]:
        if time.time() > deadline - 20:
            break
        res = find_decomposition(rank, max_attempts=10, max_iter=80)
        if res is not None:
            best_result = res
            break

    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
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
