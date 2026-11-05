import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [2, 3, 3]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 20s safety margin
    
    n, m, p = DIMENSIONS
    
    # Fallback: Rank 17 Solution
    def get_rank_17():
        rank = 17
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        def set_U(i, j, r, val): U[i * m + j, r] = val
        def set_V(j, k, r, val): V[j * p + k, r] = val
        def set_W(k, i, r, val): W[k * n + i, r] = val

        # Block 1: Strassen 2x2x2 (j in {0,1}, k in {0,1})
        set_U(0, 0, 0, 1); set_U(1, 1, 0, 1)
        set_V(0, 0, 0, 1); set_V(1, 1, 0, 1)
        set_W(0, 0, 0, 1); set_W(1, 1, 0, 1)
        
        set_U(1, 0, 1, 1); set_U(1, 1, 1, 1)
        set_V(0, 0, 1, 1)
        set_W(0, 1, 1, 1); set_W(1, 1, 1, -1)
        
        set_U(0, 0, 2, 1)
        set_V(0, 1, 2, 1); set_V(1, 1, 2, -1)
        set_W(1, 0, 2, 1); set_W(1, 1, 2, 1)
        
        set_U(1, 1, 3, 1)
        set_V(1, 0, 3, 1); set_V(0, 0, 3, -1)
        set_W(0, 0, 3, 1); set_W(0, 1, 3, 1)
        
        set_U(0, 0, 4, 1); set_U(0, 1, 4, 1)
        set_V(1, 1, 4, 1)
        set_W(0, 0, 4, -1); set_W(1, 0, 4, 1)
        
        set_U(1, 0, 5, 1); set_U(0, 0, 5, -1)
        set_V(0, 0, 5, 1); set_V(0, 1, 5, 1)
        set_W(1, 1, 5, 1)
        
        set_U(0, 1, 6, 1); set_U(1, 1, 6, -1)
        set_V(1, 0, 6, 1); set_V(1, 1, 6, 1)
        set_W(0, 0, 6, 1)

        # Block 2: Direct 2x2x1 (j in {0,1}, k=2)
        set_U(0, 0, 7, 1)
        set_V(0, 2, 7, 1)
        set_W(2, 0, 7, 1)
        
        set_U(0, 1, 8, 1)
        set_V(1, 2, 8, 1)
        set_W(2, 0, 8, 1)
        
        set_U(1, 0, 9, 1)
        set_V(0, 2, 9, 1)
        set_W(2, 1, 9, 1)
        
        set_U(1, 1, 10, 1)
        set_V(1, 2, 10, 1)
        set_W(2, 1, 10, 1)

        # Block 3: Direct 2x1x3 (j=2, k in {0,1,2})
        set_U(0, 2, 11, 1)
        set_V(2, 0, 11, 1)
        set_W(0, 0, 11, 1)
        
        set_U(0, 2, 12, 1)
        set_V(2, 1, 12, 1)
        set_W(1, 0, 12, 1)
        
        set_U(0, 2, 13, 1)
        set_V(2, 2, 13, 1)
        set_W(2, 0, 13, 1)
        
        set_U(1, 2, 14, 1)
        set_V(2, 0, 14, 1)
        set_W(0, 1, 14, 1)
        
        set_U(1, 2, 15, 1)
        set_V(2, 1, 15, 1)
        set_W(1, 1, 15, 1)
        
        set_U(1, 2, 16, 1)
        set_V(2, 2, 16, 1)
        set_W(2, 1, 16, 1)

        return U, V, W

    U17, V17, W17 = get_rank_17()
    
    def verify(U, V, W, rank):
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W.reshape(p, n, rank)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_t[i, j, r] * V_t[j, k, r] * W_t[k, i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def snap_to_half_integers(U, V, W):
        U_s = np.round(U * 2) / 2.0
        V_s = np.round(V * 2) / 2.0
        W_s = np.round(W * 2) / 2.0
        return U_s, V_s, W_s

    def find_decomposition(target_rank, max_seeds=5, max_iter=5000):
        if time.time() > deadline:
            return None
            
        for seed in range(max_seeds):
            if time.time() > deadline:
                break
                
            np.random.seed(seed)
            
            # Initialize with random noise
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            # Or initialize from rank 17 by dropping columns?
            # Let's try random first, then maybe structured.
            
            def objective(params):
                U_flat, V_flat, W_flat = params
                U = U_flat.reshape(n * m, target_rank)
                V = V_flat.reshape(m * p, target_rank)
                W = W_flat.reshape(p * n, target_rank)
                
                U_t = U.reshape(n, m, target_rank)
                V_t = V.reshape(m, p, target_rank)
                W_t = W.reshape(p, n, target_rank)
                
                errors = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(target_rank):
                                s += U_t[i, j, r] * V_t[j, k, r] * W_t[k, i, r]
                            errors.append(s - 1.0)
                return np.array(errors)
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                result = opt.least_squares(objective, x0, method='trf', max_nfev=max_iter, xtol=1e-12, ftol=1e-12, gtol=1e-12)
                
                if result.success or result.cost < 1e-8:
                    U_opt = result.x[:n * m * target_rank].reshape(n * m, target_rank)
                    V_opt = result.x[n * m * target_rank : n * m * target_rank + m * p * target_rank].reshape(m * p, target_rank)
                    W_opt = result.x[n * m * target_rank + m * p * target_rank :].reshape(p * n, target_rank)
                    
                    # Snap to half integers
                    U_snap, V_snap, W_snap = snap_to_half_integers(U_opt, V_opt, W_opt)
                    
                    if verify(U_snap, V_snap, W_snap, target_rank):
                        return U_snap, V_snap, W_snap
            except Exception:
                pass
                
        return None

    # Try to find rank 15
    best_U, best_V, best_W = U17, V17, W17
    best_rank = 17
    
    # Attempt rank 15
    res_15 = find_decomposition(15, max_seeds=3, max_iter=2000)
    if res_15 is not None:
        best_U, best_V, best_W = res_15
        best_rank = 15
    elif time.time() < deadline - 20:
        # Attempt rank 16
        res_16 = find_decomposition(16, max_seeds=3, max_iter=2000)
        if res_16 is not None:
            best_U, best_V, best_W = res_16
            best_rank = 16

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
