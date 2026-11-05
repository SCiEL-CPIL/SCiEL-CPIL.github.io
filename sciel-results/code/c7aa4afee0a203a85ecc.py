"""Matrix multiplication 2,5,6 - rank-minimal tensor decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    def build_rank54():
        rank = 54
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        def add_strassen_block(i_base, j_base, k_base):
            nonlocal col
            U[i_base * m + j_base][col] += 1
            U[(i_base+1) * m + (j_base+1)][col] += 1
            V[j_base * p + k_base][col] += 1
            V[(j_base+1) * p + (k_base+1)][col] += 1
            W[k_base * n + i_base][col] += 1
            W[(k_base+1) * n + (i_base+1)][col] += 1
            col += 1
            
            U[(i_base+1) * m + j_base][col] += 1
            U[(i_base+1) * m + (j_base+1)][col] += 1
            V[j_base * p + k_base][col] += 1
            W[k_base * n + (i_base+1)][col] += 1
            W[(k_base+1) * n + (i_base+1)][col] -= 1
            col += 1
            
            U[i_base * m + j_base][col] += 1
            V[j_base * p + (k_base+1)][col] += 1
            V[(j_base+1) * p + (k_base+1)][col] -= 1
            W[(k_base+1) * n + i_base][col] += 1
            W[(k_base+1) * n + (i_base+1)][col] += 1
            col += 1
            
            U[(i_base+1) * m + (j_base+1)][col] += 1
            V[(j_base+1) * p + k_base][col] += 1
            V[j_base * p + k_base][col] -= 1
            W[k_base * n + i_base][col] += 1
            W[k_base * n + (i_base+1)][col] += 1
            col += 1
            
            U[i_base * m + j_base][col] += 1
            U[i_base * m + (j_base+1)][col] += 1
            V[(j_base+1) * p + (k_base+1)][col] += 1
            W[k_base * n + i_base][col] -= 1
            W[(k_base+1) * n + i_base][col] += 1
            col += 1
            
            U[(i_base+1) * m + j_base][col] += 1
            U[i_base * m + j_base][col] -= 1
            V[j_base * p + k_base][col] += 1
            V[j_base * p + (k_base+1)][col] += 1
            W[(k_base+1) * n + (i_base+1)][col] += 1
            col += 1
            
            U[i_base * m + (j_base+1)][col] += 1
            U[(i_base+1) * m + (j_base+1)][col] -= 1
            V[(j_base+1) * p + k_base][col] += 1
            V[(j_base+1) * p + (k_base+1)][col] += 1
            W[k_base * n + i_base][col] += 1
            col += 1

        def add_naive_2112_block(i_base, j_base, k_base):
            nonlocal col
            U[i_base * m + j_base][col] += 1
            V[j_base * p + k_base][col] += 1
            W[k_base * n + i_base][col] += 1
            col += 1
            U[i_base * m + j_base][col] += 1
            V[j_base * p + (k_base+1)][col] += 1
            W[(k_base+1) * n + i_base][col] += 1
            col += 1
            U[(i_base+1) * m + j_base][col] += 1
            V[j_base * p + k_base][col] += 1
            W[k_base * n + (i_base+1)][col] += 1
            col += 1
            U[(i_base+1) * m + j_base][col] += 1
            V[j_base * p + (k_base+1)][col] += 1
            W[(k_base+1) * n + (i_base+1)][col] += 1
            col += 1

        for j_base in [0, 2]:
            for k_base in [0, 2, 4]:
                add_strassen_block(0, j_base, k_base)
        for k_base in [0, 2, 4]:
            add_naive_2112_block(0, 4, k_base)
        
        assert col == rank
        return {"U": U, "V": V, "W": W}, rank

    def verify_decomposition(U, V, W, n, m, p):
        """Verify by checking that the decomposition correctly computes matrix multiplication."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = U_np.shape[1]
        
        # Test with random matrices
        rng = np.random.RandomState(42)
        for trial in range(10):
            A = rng.randn(n, m)
            B = rng.randn(m, p)
            C_expected = A @ B
            
            # Compute C using the decomposition
            C_computed = np.zeros((n, p))
            for r in range(rank):
                u_r = U_np[:, r].reshape(n, m)
                v_r = V_np[:, r].reshape(m, p)
                w_r = W_np[:, r].reshape(p, n)
                M_r = np.sum(A * u_r) * np.sum(B * v_r)
                C_computed += M_r * w_r.T  # w_r is (p,n), so w_r.T is (n,p)
            
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                return False
        return True

    base_result, base_rank = build_rank54()
    
    if not verify_decomposition(base_result["U"], base_result["V"], base_result["W"], n, m, p):
        # Fallback to naive
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return base_result

# EVOLVE-BLOCK-END
