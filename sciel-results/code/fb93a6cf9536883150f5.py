import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = 1.0

    def verify(U, V, W):
        res = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(res, T, atol=1e-9)

    def get_schoolbook():
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

    def als_step(U, V, W, T, max_iters=50, tol=1e-10, deadline=None):
        I, J, K, R = U.shape[0], V.shape[0], W.shape[0], U.shape[1]
        
        VV = V.T @ V
        WW = W.T @ W
        UU = U.T @ U
        
        for _ in range(max_iters):
            if deadline and time.time() > deadline:
                break
                
            # Normalize columns to improve conditioning
            # Scale each column r such that ||U[:,r]||_2 = 1, ||V[:,r]||_2 = 1, ||W[:,r]||_2 = 1
            # This is a gauge freedom.
            # Let's just use lstsq for robustness.
            
            num_U = np.einsum('ijk,jr,kr->ir', T, V, W)
            den_U = VV * WW
            try:
                U_new = np.linalg.lstsq(den_U, num_U.T, rcond=None)[0].T
            except np.linalg.LinAlgError:
                U_new = num_U / np.maximum(np.diag(den_U), 1e-9)[:, None]
            
            num_V = np.einsum('ijk,ir,kr->jr', T, U, W)
            den_V = UU * WW
            try:
                V_new = np.linalg.lstsq(den_V, num_V.T, rcond=None)[0].T
            except np.linalg.LinAlgError:
                V_new = num_V / np.maximum(np.diag(den_V), 1e-9)[:, None]
                
            num_W = np.einsum('ijk,ir,jr->kr', T, U, V)
            den_W = UU * VV
            try:
                W_new = np.linalg.lstsq(den_W, num_W.T, rcond=None)[0].T
            except np.linalg.LinAlgError:
                W_new = num_W / np.maximum(np.diag(den_W), 1e-9)[:, None]
                
            diff = np.linalg.norm(U - U_new) + np.linalg.norm(V - V_new) + np.linalg.norm(W - W_new)
            U, V, W = U_new, V_new, W_new
            if diff < tol:
                break
                
        return U, V, W

    start_time = time.time()
    deadline = 145.0
    
    best_result = get_schoolbook()
    best_rank = n * m * p

    # Prioritize ranks 35 and 36, then lower
    ranks_to_try = [35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24]
    
    for R in ranks_to_try:
        if time.time() > deadline:
            break
        if R >= best_rank:
            continue
            
        # More attempts for higher ranks (easier to find)
        attempts = 200 if R >= 34 else 100
        
        for attempt in range(attempts):
            if time.time() > deadline:
                break
                
            # Sparse random initialization
            U = np.random.uniform(-1, 1, (n*m, R))
            V = np.random.uniform(-1, 1, (m*p, R))
            W = np.random.uniform(-1, 1, (p*n, R))
            
            # Sparsify: set 20% of entries to 0
            mask_u = np.random.rand(n*m, R) < 0.2
            mask_v = np.random.rand(m*p, R) < 0.2
            mask_w = np.random.rand(p*n, R) < 0.2
            U[mask_u] = 0
            V[mask_v] = 0
            W[mask_w] = 0
            
            U, V, W = als_step(U, V, W, T, max_iters=30, deadline=deadline)
            
            # Try rounding to integer and half-integer
            candidates = []
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            candidates.append((U_int, V_int, W_int))
            
            U_half = np.round(U * 2) / 2
            V_half = np.round(V * 2) / 2
            W_half = np.round(W * 2) / 2
            candidates.append((U_half, V_half, W_half))
            
            for Uc, Vc, Wc in candidates:
                if verify(Uc, Vc, Wc):
                    if R < best_rank:
                        best_rank = R
                        best_result = {
                            "U": Uc.tolist(),
                            "V": Vc.tolist(),
                            "W": Wc.tolist()
                        }
                    break # Found for this rank, break attempts
            
    return best_result
