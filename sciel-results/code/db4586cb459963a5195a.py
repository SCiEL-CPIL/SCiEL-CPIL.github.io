"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Attempt to find a rank-33 decomposition for 2x2 * 2x10 matrix multiplication.
    Uses rank-13 decomposition for 2x2 * 2x4 blocks (two of them) and rank-7 for 2x2 * 2x2.
    Falls back to rank-35 (5x Strassen) if the search fails or takes too long.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=10
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # Try to find rank 13 decomposition for 2x2x4
    # If successful, use it for the first 8 columns (two 2x4 blocks)
    # and rank 7 for the last 2 columns.
    # Total rank: 13 + 13 + 7 = 33
    
    def find_rank_13_decomp_2x2x4():
        # Tensor T for 2x2 * 2x4
        # U shape (4, 13), V shape (8, 13), W shape (8, 13)
        # Non-zero entries in T:
        # i in 0..1, j in 0..1, k in 0..3
        # U_idx = i*2 + j
        # V_idx = j*4 + k
        # W_idx = i*4 + k
        T = np.zeros((4, 8, 8))
        for i in range(2):
            for j in range(2):
                for k in range(4):
                    u_idx = i * 2 + j
                    v_idx = j * 4 + k
                    w_idx = i * 4 + k
                    T[u_idx, v_idx, w_idx] = 1.0
        
        # We want to find U, V, W of rank 13 such that sum_r U[:,r] V[:,r] W[:,r] = T
        # This is a nonlinear system.
        # Strategy: Fix U to be random/structured, solve for V and W?
        # Or use gradient descent on the residual.
        
        # Initialize U, V, W
        np.random.seed(42)
        U = np.random.randn(4, 13)
        V = np.random.randn(8, 13)
        W = np.random.randn(8, 13)
        
        # Normalize to prevent explosion
        U = U / np.linalg.norm(U)
        V = V / np.linalg.norm(V)
        W = W / np.linalg.norm(W)
        
        # Simple iterative refinement
        # Residual R = T - sum_r U_r V_r W_r
        # Update one factor at a time.
        
        for iter in range(1000):
            if time.time() > deadline - 10:
                return None
            
            # Update U
            # For each r, U[:,r] = argmin || T - sum_r U_r V_r W_r ||
            # This is equivalent to solving a linear least squares for each r?
            # No, it's coupled.
            # Let's use a simple gradient step.
            
            # Compute current tensor
            T_curr = np.einsum('ir,jr,kr->ijk', U, V, W)
            Res = T - T_curr
            
            # Gradient wrt U
            # dL/dU_ir = sum_jk Res_ijk V_jr W_kr
            grad_U = np.einsum('ijk,jr,kr->ir', Res, V, W)
            grad_V = np.einsum('ijk,ir,kr->jr', Res, U, W)
            grad_W = np.einsum('ijk,ir,jr->kr', Res, U, V)
            
            # Learning rate decay
            lr = 0.1 / (1 + iter * 0.01)
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Check convergence
            if np.linalg.norm(Res) < 1e-6:
                # Try to round to integers/half-integers
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                
                T_int = np.einsum('ir,jr,kr->ijk', U_int, V_int, W_int)
                if np.allclose(T_int, T, atol=1e-5):
                    return U_int, V_int, W_int
                else:
                    # If rounding fails, continue or give up
                    # For now, return None if rounding fails
                    pass
                    
        return None

    U_13, V_13, W_13 = find_rank_13_decomp_2x2x4()
    
    if U_13 is not None:
        rank = 33
        U = np.zeros((4, rank))
        V = np.zeros((20, rank))
        W = np.zeros((20, rank))
        
        # Block 1: 2x2 * 2x4 (cols 0-3 of B, cols 0-3 of C)
        # V indices for B[0:2, 0:4] are 0-3 and 4-7
        # W indices for C[0:2, 0:4] are 0-3 and 4-7 (trace convention: k*n+i)
        # Wait, trace convention W[k*n+i, r].
        # For 2x2x4:
        # B is 2x4. V shape (8, 13).
        # C is 2x4. W shape (8, 13).
        # W index k*n+i. n=2.
        # k=0, i=0 -> 0
        # k=0, i=1 -> 1
        # k=1, i=0 -> 2
        # k=1, i=1 -> 3
        # k=2, i=0 -> 4
        # k=2, i=1 -> 5
        # k=3, i=0 -> 6
        # k=3, i=1 -> 7
        # So W_13 indices 0-7 map directly to global W indices 0-7.
        # V_13 indices 0-7 map to global V indices 0-7.
        
        U[:, 0:13] = U_13
        V[0:8, 0:13] = V_13
        W[0:8, 0:13] = W_13
        
        # Block 2: 2x2 * 2x4 (cols 4-7 of B, cols 4-7 of C)
        # V indices for B[0:2, 4:8] are 8-11 and 12-15
        # W indices for C[0:2, 4:8] are 8-15
        
        U[:, 13:26] = U_13
        V[8:16, 13:26] = V_13
        W[8:16, 13:26] = W_13
        
        # Block 3: 2x2 * 2x2 (cols 8-9 of B, cols 8-9 of C)
        # Use Strassen rank 7
        # V indices for B[0:2, 8:10] are 16-17 and 18-19
        # W indices for C[0:2, 8:10] are 16-19
        
        U_strassen = np.zeros((4, 7))
        V_strassen = np.zeros((4, 7)) # Local 2x2
        W_strassen = np.zeros((4, 7)) # Local 2x2
        
        # Strassen
        # P1 = (a+d)(e+h)
        U_strassen[0, 0] = 1; U_strassen[3, 0] = 1
        V_strassen[0, 0] = 1; V_strassen[3, 0] = 1
        W_strassen[0, 0] = 1; W_strassen[3, 0] = 1
        
        # P2 = (c+d)e
        U_strassen[2, 1] = 1; U_strassen[3, 1] = 1
        V_strassen[0, 1] = 1
        W_strassen[1, 1] = 1; W_strassen[3, 1] = -1
        
        # P3 = a(f-h)
        U_strassen[0, 2] = 1
        V_strassen[1, 2] = 1; V_strassen[3, 2] = -1
        W_strassen[2, 2] = 1; W_strassen[3, 2] = 1
        
        # P4 = d(g-e)
        U_strassen[3, 3] = 1
        V_strassen[2, 3] = 1; V_strassen[0, 3] = -1
        W_strassen[0, 3] = 1; W_strassen[1, 3] = 1
        
        # P5 = (a+b)h
        U_strassen[0, 4] = 1; U_strassen[1, 4] = 1
        V_strassen[3, 4] = 1
        W_strassen[0, 4] = -1; W_strassen[2, 4] = 1
        
        # P6 = (c-a)(e+f)
        U_strassen[2, 5] = 1; U_strassen[0, 5] = -1
        V_strassen[0, 5] = 1; V_strassen[1, 5] = 1
        W_strassen[3, 5] = 1
        
        # P7 = (b-d)(g+h)
        U_strassen[1, 6] = 1; U_strassen[3, 6] = -1
        V_strassen[2, 6] = 1; V_strassen[3, 6] = 1
        W_strassen[0, 6] = 1
        
        U[:, 26:33] = U_strassen
        V[16:20, 26:33] = V_strassen
        W[16:20, 26:33] = W_strassen
        
        # Convert to lists
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to rank 35
        rank = 35
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for t in range(5):
            col_base = t * 7
            
            col = col_base
            U[0][col] = 1; U[3][col] = 1
            V[2*t][col] = 1; V[11+2*t][col] = 1
            W[4*t][col] = 1; W[4*t+3][col] = 1
            
            col = col_base + 1
            U[2][col] = 1; U[3][col] = 1
            V[2*t][col] = 1
            W[4*t+1][col] = 1; W[4*t+3][col] = -1
            
            col = col_base + 2
            U[0][col] = 1
            V[2*t+1][col] = 1; V[11+2*t][col] = -1
            W[4*t+2][col] = 1; W[4*t+3][col] = 1
            
            col = col_base + 3
            U[3][col] = 1
            V[10+2*t][col] = 1; V[2*t][col] = -1
            W[4*t][col] = 1; W[4*t+1][col] = 1
            
            col = col_base + 4
            U[0][col] = 1; U[1][col] = 1
            V[11+2*t][col] = 1
            W[4*t][col] = -1; W[4*t+2][col] = 1
            
            col = col_base + 5
            U[2][col] = 1; U[0][col] = -1
            V[2*t][col] = 1; V[2*t+1][col] = 1
            W[4*t+3][col] = 1
            
            col = col_base + 6
            U[1][col] = 1; U[3][col] = -1
            V[10+2*t][col] = 1; V[11+2*t][col] = 1
            W[4*t][col] = 1
        
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
