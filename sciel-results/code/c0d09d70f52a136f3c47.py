import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Build target tensor T
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    T_flat = T.reshape(n*m, m*p * p*n)
    
    def verify(U, V, W):
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        U_check = U_arr * 2
        V_check = V_arr * 2
        W_check = W_arr * 2
        
        if not np.allclose(U_check, np.round(U_check), atol=1e-9):
            return False
        if not np.allclose(V_check, np.round(V_check), atol=1e-9):
            return False
        if not np.allclose(W_check, np.round(W_check), atol=1e-9):
            return False
            
        U_int = np.round(U_check).astype(np.int64)
        V_int = np.round(V_check).astype(np.int64)
        W_int = np.round(W_check).astype(np.int64)
        
        rank = U_int.shape[1]
        
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    val = 0
                    for k in range(rank):
                        val += U_int[a, k] * V_int[b, k] * W_int[c, k]
                    target = int(T[a, b, c] * 8)
                    if val != target:
                        return False
        return True

    def solve_rank(r, time_limit, n_init=5, max_iter=100):
        for init in range(n_init):
            if time.time() > time_limit:
                return None
            
            # Random initialization
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            for it in range(max_iter):
                if time.time() > time_limit:
                    return None
                
                # Update U
                M_U = np.zeros((n*m, m*p * p*n, r))
                for b in range(m*p):
                    for c in range(p*n):
                        vec = V[b, :] * W[c, :]
                        for a in range(n*m):
                            M_U[a, b * p*n + c, :] = vec
                            
                for a in range(n*m):
                    U[a, :], _, _, _ = np.linalg.lstsq(M_U[a], T_flat[a], rcond=None)
                
                # Update V
                M_V = np.zeros((m*p, n*m * p*n, r))
                for a in range(n*m):
                    for c in range(p*n):
                        vec = U[a, :] * W[c, :]
                        for b in range(m*p):
                            M_V[b, a * p*n + c, :] = vec
                            
                for b in range(m*p):
                    t_b = T[:, b, :].flatten()
                    V[b, :], _, _, _ = np.linalg.lstsq(M_V[b], t_b, rcond=None)
                    
                # Update W
                M_W = np.zeros((p*n, n*m * m*p, r))
                for a in range(n*m):
                    for b in range(m*p):
                        vec = U[a, :] * V[b, :]
                        for c in range(p*n):
                            M_W[c, a * m*p + b, :] = vec
                            
                for c in range(p*n):
                    t_c = T[:, :, c].flatten()
                    W[c, :], _, _, _ = np.linalg.lstsq(M_W[c], t_c, rcond=None)
                    
            U_rounded = np.round(U * 2) / 2.0
            V_rounded = np.round(V * 2) / 2.0
            W_rounded = np.round(W * 2) / 2.0
            
            if verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()):
                return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
                
        return None

    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
                
    # Try ranks 62, 61, ...
    for r in range(schoolbook_rank - 1, max(50, schoolbook_rank - 15), -1):
        if time.time() > deadline - 10:
            break
        res = solve_rank(r, deadline - 5, n_init=5, max_iter=50)
        if res is not None:
            U, V, W = res
            return {"U": U, "V": V, "W": W}
            
    return {"U": U_sb, "V": V_sb, "W": W_sb}
