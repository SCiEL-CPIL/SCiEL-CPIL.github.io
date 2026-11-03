import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin for 150s limit
    
    n, m, p = 2, 3, 6
    
    # Build target tensor T for 2x3x6
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Strategy: Decompose 2x3x6 into two 2x3x3 blocks.
    # Rank of 2x3x3 is 15. Total rank 30.
    
    n2, m2, p2 = 2, 3, 3
    T233 = np.zeros((n2*m2, m2*p2, p2*n2))
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                T233[i*m2+j, j*p2+k, k*n2+i] = 1.0
        
    R233 = 15
    rows_u = n2 * m2  # 6
    rows_v = m2 * p2  # 9
    rows_w = p2 * n2  # 6
    
    best_233 = None
    
    # ALS Loop with many restarts
    for restart in range(200):
        if time.time() > deadline - 10:
            break
        
        # Initialize with small random values
        # Using normal distribution
        U = np.random.randn(rows_u, R233) * 0.1
        V = np.random.randn(rows_v, R233) * 0.1
        W = np.random.randn(rows_w, R233) * 0.1
        
        converged = False
        for it in range(50):
            if time.time() > deadline - 10:
                break
                
            # Solve for U
            Z_u = np.zeros((R233, rows_v * rows_w))
            for r in range(R233):
                Z_u[r, :] = np.outer(V[:, r], W[:, r]).flatten()
            T_flat_u = T233.transpose(0, 1, 2).flatten().reshape(rows_u, -1)
            # U @ Z_u.T = T_flat_u
            # U = lstsq(Z_u.T, T_flat_u.T).T
            try:
                U_new, _, _, _ = np.linalg.lstsq(Z_u.T, T_flat_u.T, rcond=None)
                U = U_new.T
            except:
                break

            # Solve for V
            Z_v = np.zeros((R233, rows_u * rows_w))
            for r in range(R233):
                Z_v[r, :] = np.outer(U[:, r], W[:, r]).flatten()
            T_flat_v = T233.transpose(1, 0, 2).flatten().reshape(rows_v, -1)
            try:
                V_new, _, _, _ = np.linalg.lstsq(Z_v.T, T_flat_v.T, rcond=None)
                V = V_new.T
            except:
                break

            # Solve for W
            Z_w = np.zeros((R233, rows_u * rows_v))
            for r in range(R233):
                Z_w[r, :] = np.outer(U[:, r], V[:, r]).flatten()
            T_flat_w = T233.transpose(2, 0, 1).flatten().reshape(rows_w, -1)
            try:
                W_new, _, _, _ = np.linalg.lstsq(Z_w.T, T_flat_w.T, rcond=None)
                W = W_new.T
            except:
                break
            
            # Check convergence
            # Simple check: if change is small
            # We don't store previous, so we can't easily check diff.
            # Just run fixed iterations.
        
        # After ALS, round to nearest half-integer
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Verify
        T_check = np.einsum('ir,jr,kr->ijk', U_rounded, V_rounded, W_rounded)
        if np.allclose(T_check, T233, atol=1e-9):
            best_233 = (U_rounded, V_rounded, W_rounded)
            break

    if best_233:
        U233, V233, W233 = best_233
        # Embed into 2x3x6
        R236 = 2 * R233
        U236 = np.zeros((n*m, R236))
        V236 = np.zeros((m*p, R236))
        W236 = np.zeros((p*n, R236))
        
        # Block 1
        U236[:, :R233] = U233
        V236[:m2*p2, :R233] = V233
        W236[:p2*n2, :R233] = W233
        
        # Block 2
        U236[:, R233:] = U233
        V236[m2*p2:, R233:] = V233
        W236[p2*n2:, R233:] = W233
        
        # Verify full tensor
        T_final = np.einsum('ir,jr,kr->ijk', U236, V236, W236)
        if np.allclose(T_final, T, atol=1e-9):
            return {
                "U": U236.tolist(),
                "V": V236.tolist(),
                "W": W236.tolist()
            }

    # Fallback to schoolbook rank 36
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
