"""
Optimized 6x6 Matrix Multiplication Tensor Decomposition.

Strategy:
1. Primary: Hybrid Strassen (2x2 outer) + Schoolbook (3x3 inner) decomposition.
   Rank = 7 * 27 = 189.
   This is significantly better than Schoolbook (216).
   We use a list-of-lists implementation for speed and simplicity, matching the
   most efficient inspiration. Verification is done via NumPy for robustness.

2. Fallback: If the primary fails verification (should not happen), use
   Schoolbook 6x6.
   Rank = 216.

The key is the block structure: 6x6 matrices are viewed as 2x2 blocks of 3x3
matrices. Strassen's algorithm reduces the number of block multiplications from
8 to 7. Each block multiplication is then expanded into 27 scalar multiplications
(schoolbook for 3x3). Total rank = 7 * 27 = 189.
"""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 145  # Safety margin before 150s wall-clock kill

def construct():
    start_time = time.time()
    
    n, m, p = 6, 6, 6
    dim = 6
    
    # Fallback: Schoolbook 6x6 (Rank 216)
    def _schoolbook():
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
        
    # Hybrid: 2x2 Strassen (Outer) x 3x3 Schoolbook (Inner) (Rank 189)
    def _hybrid_strassen_schoolbook():
        block_size = 3
        num_blocks = 2
        
        rank_inner = 27  # Schoolbook rank for 3x3
        rank_outer = 7  # Strassen rank for 2x2
        total_rank = rank_inner * rank_outer
        
        U = [[0] * total_rank for _ in range(dim * dim)]
        V = [[0] * total_rank for _ in range(dim * dim)]
        W = [[0] * total_rank for _ in range(dim * dim)]
        
        col = 0
        
        # Strassen steps for 2x2 outer
        # Format: {'A': {(I,J): coeff}, 'B': {(J,K): coeff}, 'C': {(I,K): coeff}}
        strassen_steps = [
            {'A': { (0,0): 1, (1,1): 1 }, 'B': { (0,0): 1, (1,1): 1 }, 'C': { (0,0): 1, (1,1): 1 }}, # M1
            {'A': { (1,0): 1, (1,1): 1 }, 'B': { (0,0): 1 }, 'C': { (1,0): 1, (1,1): -1 }}, # M2
            {'A': { (0,0): 1 }, 'B': { (0,1): 1, (1,1): -1 }, 'C': { (0,1): 1, (1,1): 1 }}, # M3
            {'A': { (1,1): 1 }, 'B': { (1,0): 1, (0,0): -1 }, 'C': { (0,0): 1, (1,0): 1 }}, # M4
            {'A': { (0,0): 1, (0,1): 1 }, 'B': { (1,1): 1 }, 'C': { (0,0): -1, (0,1): 1 }}, # M5
            {'A': { (1,0): 1, (0,0): -1 }, 'B': { (0,0): 1, (0,1): 1 }, 'C': { (1,1): 1 }}, # M6
            {'A': { (0,1): 1, (1,1): -1 }, 'B': { (1,0): 1, (1,1): 1 }, 'C': { (0,0): 1 }} # M7
        ]
        
        for s_step in strassen_steps:
            A_coeffs_outer = s_step['A']
            B_coeffs_outer = s_step['B']
            C_coeffs_outer = s_step['C']
            
            # Iterate over local indices for the 3x3 blocks
            # u: local row of output and X
            # w: local col of output and Y
            # v: local col of X and row of Y (summed index)
            for u in range(block_size):
                for w in range(block_size):
                    for v in range(block_size):
                        # This corresponds to the multiplication X[u,v] * Y[v,w]
                        # which contributes to M[u,w]
                        
                        # Update U (Left factor)
                        # X[u,v] = sum_{I,J} A_coeffs[(I,J)] * A[I*3+u, J*3+v]
                        for (I, J), ca in A_coeffs_outer.items():
                            if ca != 0:
                                i_global = I * block_size + u
                                j_global = J * block_size + v
                                idx_u = i_global * dim + j_global
                                U[idx_u][col] += ca
                        
                        # Update V (Right factor)
                        # Y[v,w] = sum_{J,K} B_coeffs[(J,K)] * B[J*3+v, K*3+w]
                        for (J, K), cb in B_coeffs_outer.items():
                            if cb != 0:
                                j_global = J * block_size + v
                                k_global = K * block_size + w
                                idx_v = j_global * dim + k_global
                                V[idx_v][col] += cb
                        
                        # Update W (Output factor)
                        # M[u,w] contributes to C blocks defined in C_coeffs
                        # C[I,K,u,w] gets contribution with weight C_coeffs[(I,K)]
                        for (I, K), cc in C_coeffs_outer.items():
                            if cc != 0:
                                i_global = I * block_size + u
                                k_global = K * block_size + w
                                # Task convention: W[k*n+i, r]
                                idx_w = k_global * dim + i_global
                                W[idx_w][col] += cc
                        
                        col += 1
        
        return {"U": U, "V": V, "W": W}

    # Try Hybrid Strassen-Schoolbook (189)
    try:
        if time.time() < _DEADLINE:
            res = _hybrid_strassen_schoolbook()
            # Verify using NumPy for speed
            U_np = np.array(res["U"], dtype=float)
            V_np = np.array(res["V"], dtype=float)
            W_np = np.array(res["W"], dtype=float)
            
            # Efficient verification:
            # The tensor T_{i,j,k} = sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}
            # We want T_{i,j,k} = 1 for all i,j,k.
            # Instead of looping 216 times with dot products, we can reshape and use einsum or matrix mult.
            # But 216 * (dot of 189) is small. Let's stick to a robust loop but vectorize the inner dot if possible.
            # Actually, simple loop is fine for 216 cases.
            
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    u_row = U_np[i * m + j, :]
                    for k in range(p):
                        v_row = V_np[j * p + k, :]
                        w_row = W_np[k * n + i, :]
                        s = np.dot(u_row, v_row * w_row)
                        err = abs(s - 1.0)
                        if err > max_err:
                            max_err = err
            
            if max_err < 1e-10:
                return res
    except Exception:
        pass
        
    # Fallback to Schoolbook
    return _schoolbook()

# EVOLVE-BLOCK-END
