import numpy as np
import time
import random

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    # Try to construct rank-47 decomposition directly
    # For 3x4x4, we can use the standard rank-47 decomposition
    # This is based on the known optimal decomposition for this case
    
    # The tensor T[i*m+j, j*p+k, k*n+i] = 1
    # We need U (12 x R), V (16 x R), W (12 x R)
    
    # Known: rank 47 is achievable for 3x4x4
    # Let's construct it using the standard method
    
    # Actually, let's use a different approach:
    # The minimum rank for 3x4x4 is 47 (proven)
    # We'll try to find it via structured search
    
    # Strategy: Use the fact that we can "save" one multiplication
    # compared to schoolbook (48) by finding a linear dependency
    
    # For now, let's try a more efficient ALS with better initialization
    # and tighter time management
    
    def build_tensor():
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        return T
    
    T = build_tensor()
    
    # Try rank 47 with multiple random restarts
    best_result = None
    
    for attempt in range(20):
        if time.time() > deadline - 5:
            break
        
        R = 47
        np.random.seed(attempt * 7 + 3)
        
        # Initialize with small random values
        U = np.random.randn(n*m, R) * 0.1
        V = np.random.randn(m*p, R) * 0.1
        W = np.random.randn(p*n, R) * 0.1
        
        # ALS iterations
        converged = False
        for it in range(30):
            if time.time() > deadline - 5:
                break
            
            # Update U
            for idx in range(n*m):
                # A[row, r] = V[J, r] * W[K, r] for all J, K
                # Vectorized: A_U shape (m*p, p*n, R)
                A_U = np.einsum('jr, kr -> jkr', V, W)
                A_U_flat = A_U.reshape(-1, R)
                b_U = T[idx, :, :].flatten()
                U[idx, :], _, _, _ = np.linalg.lstsq(A_U_flat, b_U, rcond=None)
            
            # Update V
            for idx in range(m*p):
                A_V = np.einsum('ir, kr -> ikr', U, W)
                A_V_flat = A_V.reshape(-1, R)
                b_V = T[:, idx, :].flatten()
                V[idx, :], _, _, _ = np.linalg.lstsq(A_V_flat, b_V, rcond=None)
            
            # Update W
            for idx in range(p*n):
                A_W = np.einsum('ir, jr -> ijr', U, V)
                A_W_flat = A_W.reshape(-1, R)
                b_W = T[:, :, idx].flatten()
                W[idx, :], _, _, _ = np.linalg.lstsq(A_W_flat, b_W, rcond=None)
            
            # Check convergence
            T_approx = np.einsum('ir, jr, kr -> ijk', U, V, W)
            err = np.linalg.norm(T_approx - T)
            if err < 1e-10:
                # Try to round to half-integers
                U2 = U * 2
                V2 = V * 2
                W2 = W * 2
                
                U_r = np.round(U2)
                V_r = np.round(V2)
                W_r = np.round(W2)
                
                U_c = U_r / 2.0
                V_c = V_r / 2.0
                W_c = W_r / 2.0
                
                T_check = np.einsum('ir, jr, kr -> ijk', U_c, V_c, W_c)
                if np.linalg.norm(T_check - T) < 1e-9:
                    best_result = (U_c, V_c, W_c)
                    converged = True
                    break
        
        if converged:
            break
    
    if best_result is not None:
        U, V, W = best_result
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback: schoolbook rank 48
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
