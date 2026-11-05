import numpy as np
import time

def construct():
    start_time = time.time()
    max_time = 140  # Safety margin
    
    n, m, p = 2, 3, 4
    
    num_U = n * m  # 6
    num_V = m * p  # 12
    num_W = p * n  # 8
    
    T = np.zeros((num_U, num_V, num_W))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0

    def verify_decomposition(U, V, W, R):
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(T_recon, T, atol=1e-9)

    def snap_and_verify(U, V, W, R):
        """Try to snap to rational values and verify"""
        for mult in [1, 2, 4, 8, 16, 32]:
            U_snap = np.round(U * mult) / mult
            V_snap = np.round(V * mult) / mult
            W_snap = np.round(W * mult) / mult
            if verify_decomposition(U_snap, V_snap, W_snap, R):
                return {
                    "U": U_snap.tolist(),
                    "V": V_snap.tolist(),
                    "W": W_snap.tolist()
                }
        return None

    def try_rank(R, num_restarts=5, max_iters=50):
        for restart in range(num_restarts):
            if time.time() - start_time > max_time - 15:
                return None
                
            seed = hash((R, restart)) % (2**32)
            np.random.seed(seed)
            U = np.random.uniform(-1, 1, (num_U, R))
            V = np.random.uniform(-1, 1, (num_V, R))
            W = np.random.uniform(-1, 1, (num_W, R))
            
            prev_error = np.inf
            for it in range(max_iters):
                if time.time() - start_time > max_time - 10:
                    return None

                # Update U
                for i in range(num_U):
                    M = (V.T[:, :, None] * W.T[:, None, :]).reshape(R, -1)
                    b = T[i].flatten()
                    try:
                        U[i, :], _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                    except:
                        return None

                # Update V
                for j in range(num_V):
                    M = (U.T[:, :, None] * W.T[:, None, :]).reshape(R, -1)
                    b = T[:, j, :].flatten()
                    try:
                        V[j, :], _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                    except:
                        return None

                # Update W
                for k in range(num_W):
                    M = (U.T[:, :, None] * V.T[:, None, :]).reshape(R, -1)
                    b = T[:, :, k].flatten()
                    try:
                        W[k, :], _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                    except:
                        return None

                # Check convergence
                T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
                error = np.max(np.abs(T_recon - T))
                
                if error < 1e-10:
                    result = snap_and_verify(U, V, W, R)
                    if result:
                        return result
                
                if error > prev_error * 1.01 and it > 5:
                    # Diverging, break early
                    break
                prev_error = error
            
            # Final check after all iterations
            T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            error = np.max(np.abs(T_recon - T))
            if error < 1e-8:
                result = snap_and_verify(U, V, W, R)
                if result:
                    return result
        
        return None

    # Fallback to schoolbook rank 24
    def schoolbook_rank24():
        rank = 24
        U = [[0] * rank for _ in range(num_U)]
        V = [[0] * rank for _ in range(num_V)]
        W = [[0] * rank for _ in range(num_W)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    I = i * m + j
                    J = j * p + k
                    K = k * n + i
                    U[I][column] = 1
                    V[J][column] = 1
                    W[K][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # Try ranks from 19 upwards (theoretical lower bound suggests ~19-22)
    best_result = schoolbook_rank24()
    
    for R in range(19, 23):
        if time.time() - start_time > max_time - 20:
            break
        dec = try_rank(R, num_restarts=4, max_iters=40)
        if dec and verify_decomposition(np.array(dec["U"]), np.array(dec["V"]), np.array(dec["W"]), R):
            best_result = dec
            break
    
    return best_result
