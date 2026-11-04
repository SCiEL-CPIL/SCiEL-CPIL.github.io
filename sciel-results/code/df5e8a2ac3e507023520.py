"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    start_time = time.time()
    deadline = start_time + 100  # 100 seconds for search
    
    # Build the matrix multiplication tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I_idx = i*m + j
                J_idx = j*p + k
                K_idx = k*n + i
                T[I_idx, J_idx, K_idx] = 1.0

    def verify(U, V, W):
        """Verify that U, V, W exactly decompose T with integer/half-integer entries."""
        # Scale by 2 to check for integers
        U2 = U * 2
        V2 = V * 2
        W2 = W * 2
        
        U_round = np.round(U2)
        V_round = np.round(V2)
        W_round = np.round(W2)
        
        # Check if rounding changed the values significantly
        if not (np.allclose(U2, U_round, atol=1e-9) and 
                np.allclose(V2, V_round, atol=1e-9) and 
                np.allclose(W2, W_round, atol=1e-9)):
            return False
            
        U_check = U_round / 2.0
        V_check = V_round / 2.0
        W_check = W_round / 2.0
        
        # Reconstruct T
        T_approx = np.einsum('ir, jr, kr -> ijk', U_check, V_check, W_check)
        if np.allclose(T_approx, T, atol=1e-9):
            return True
        return False

    # Try to find a solution of rank R using ALS with multiple restarts
    def try_find_solution(R, time_limit):
        num_restarts = 10
        for restart in range(num_restarts):
            if time.time() > time_limit:
                return None
            
            # Initialize with random integers in {-1, 0, 1}
            np.random.seed(42 + restart)
            U = np.random.randint(-1, 2, (n*m, R)).astype(float)
            V = np.random.randint(-1, 2, (m*p, R)).astype(float)
            W = np.random.randint(-1, 2, (p*n, R)).astype(float)
            
            # ALS Loop
            for it in range(30):
                if time.time() > time_limit:
                    return None
                
                # Update U
                for i in range(n*m):
                    A_U = np.einsum('jr, kr -> jkr', V, W).reshape(-1, R)
                    b_U = T[i, :, :].flatten()
                    U[i, :], _, _, _ = np.linalg.lstsq(A_U, b_U, rcond=None)
                
                # Update V
                for j in range(m*p):
                    A_V = np.einsum('ir, kr -> ikr', U, W).reshape(-1, R)
                    b_V = T[:, j, :].flatten()
                    V[j, :], _, _, _ = np.linalg.lstsq(A_V, b_V, rcond=None)
                
                # Update W
                for k in range(p*n):
                    A_W = np.einsum('ir, jr -> ijr', U, V).reshape(-1, R)
                    b_W = T[:, :, k].flatten()
                    W[k, :], _, _, _ = np.linalg.lstsq(A_W, b_W, rcond=None)
                
                # Check error
                T_approx = np.einsum('ir, jr, kr -> ijk', U, V, W)
                err = np.linalg.norm(T_approx - T)
                if err < 1e-9:
                    if verify(U, V, W):
                        return U, V, W
        return None

    # Try rank 47 first (known optimal for 3x4x4)
    target_rank = 47
    res = try_find_solution(target_rank, deadline)
    if res is not None:
        U, V, W = res
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

    # Fallback to Schoolbook rank 48
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
