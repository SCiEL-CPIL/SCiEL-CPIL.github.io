"""Strategy: Deterministic Constructive Rank-22 Decomposition via Strassen Base + Incremental Extension."""
import time
import numpy as np

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin

    # Fallback: Schoolbook Rank 28
    def schoolbook():
        r = n * m * p
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i * m + k, col] = 1
                    V[k * p + j, col] = 1
                    W[j * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, r

    best_result, best_rank = schoolbook()

    def verify(U, V, W, target_r):
        r = U.shape[1]
        if r != target_r:
            return False
        u_r = U.T.reshape(n, m, r)
        v_r = V.T.reshape(m, p, r)
        w_r = W.T.reshape(p, n, r)
        T = np.einsum('ikr,kjr,jir->ijk', u_r, v_r, w_r)
        return np.allclose(T, 1.0, atol=1e-8)

    # Try Rank 22 first
    target_rank = 22
    
    # Initialize Base Strassen (Rank 7)
    U_base = np.zeros((4, 7))
    V_base = np.zeros((14, 7))
    
    # r=0: M1
    U_base[0, 0] = 1; U_base[3, 0] = 1
    V_base[0, 0] = 1; V_base[8, 0] = 1
    
    # r=1: M2
    U_base[2, 1] = 1; U_base[3, 1] = 1
    V_base[0, 1] = 1
    
    # r=2: M3
    U_base[0, 2] = 1
    V_base[7, 2] = 1; V_base[8, 2] = -1
    
    # r=3: M4
    U_base[3, 3] = 1
    V_base[8, 3] = 1; V_base[0, 3] = -1
    
    # r=4: M5
    U_base[0, 4] = 1; U_base[1, 4] = 1
    V_base[8, 4] = 1
    
    # r=5: M6
    U_base[2, 5] = 1; U_base[0, 5] = -1
    V_base[0, 5] = 1; V_base[8, 5] = 1
    
    # r=6: M7
    U_base[1, 6] = 1; U_base[3, 6] = -1
    V_base[7, 6] = 1; V_base[8, 6] = 1
    
    found = False
    num_restarts = 30
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10: break
        
        np.random.seed(4000 + restart)
        
        U = np.zeros((4, target_rank))
        V = np.zeros((14, target_rank))
        
        U[:, :7] = U_base
        V[:, :7] = V_base
        
        # Randomize extension part
        U[:, 7:] = np.random.choice([0, 1, -1], size=(4, 15)) * 0.5
        V[:, 7:] = np.random.choice([0, 1, -1], size=(14, 15)) * 0.5
        
        W = np.zeros((14, target_rank))
        
        for it in range(100):
            if time.time() > deadline - 5: break
            
            M = np.zeros((n * p * m, p * n * target_rank))
            for i in range(n):
                for j in range(p):
                    for k in range(m):
                        row = (i * p + j) * m + k
                        for r in range(target_rank):
                            col = (j * n + i) * target_rank + r
                            M[row, col] = U[i * m + k, r] * V[k * p + j, r]
            
            b = np.ones(n * p * m)
            
            try:
                W_flat, residuals, rank, s = np.linalg.lstsq(M, b, rcond=None)
                W = W_flat.reshape(p * n, target_rank)
            except Exception:
                break
            
            def get_tensor(U, V, W):
                r = U.shape[1]
                u_r = U.T.reshape(n, m, r)
                v_r = V.T.reshape(m, p, r)
                w_r = W.T.reshape(p, n, r)
                T = np.einsum('ikr,kjr,jir->ijk', u_r, v_r, w_r)
                return T

            T = get_tensor(U, V, W)
            residual = T - 1.0
            
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            
            for i in range(n):
                for k in range(m):
                    for r in range(target_rank):
                        for j in range(p):
                            grad_U[i*m+k, r] += 2 * residual[i, j, k] * V[k*p+j, r] * W[j*n+i, r]
                            
            for k in range(m):
                for j in range(p):
                    for r in range(target_rank):
                        for i in range(n):
                            grad_V[k*p+j, r] += 2 * residual[i, j, k] * U[i*m+k, r] * W[j*n+i, r]
            
            lr = 0.01
            U -= lr * grad_U
            V -= lr * grad_V
            
            if it % 5 == 0:
                U[np.abs(U) < 0.01] = 0.0
                U[np.abs(U - 0.5) < 0.01] = 0.5
                U[np.abs(U + 0.5) < 0.01] = -0.5
                U[np.abs(U - 1.0) < 0.01] = 1.0
                U[np.abs(U + 1.0) < 0.01] = -1.0
                
                V[np.abs(V) < 0.01] = 0.0
                V[np.abs(V - 0.5) < 0.01] = 0.5
                V[np.abs(V + 0.5) < 0.01] = -0.5
                V[np.abs(V - 1.0) < 0.01] = 1.0
                V[np.abs(V + 1.0) < 0.01] = -1.0

            if np.max(np.abs(residual)) < 1e-6:
                U_rounded = np.round(U * 2) / 2.0
                V_rounded = np.round(V * 2) / 2.0
                W_rounded = np.round(W * 2) / 2.0
                
                if verify(U_rounded, V_rounded, W_rounded, target_rank):
                    best_result = {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break

    return best_result
# EVOLVE-BLOCK-END
