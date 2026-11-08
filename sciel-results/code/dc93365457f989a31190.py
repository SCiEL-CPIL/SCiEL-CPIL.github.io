import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # Leave margin before 150s kill
    
    def verify_solution(sol):
        """Verify a solution using the trace convention W[k*n+i,r]"""
        U, V, W = sol['U'], sol['V'], sol['W']
        rank = len(U[0])
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_true = A @ B
        C_calc = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    for r in range(rank):
                        s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                C_calc[i, k] = s
        return np.allclose(C_calc, C_true, atol=1e-10)
    
    def get_tiled_strassen_solution():
        """
        Construct a rank-189 decomposition by tiling 2x2 Strassen (rank 7) 
        over a 3x3 grid of 2x2 blocks.
        
        The 6x6 matrix is divided into 3x3 blocks of 2x2 submatrices.
        For each triple of block positions (I,J,K), we apply Strassen's 7 multiplications.
        Total rank = 3*3*3*7 = 189.
        """
        block_size = 2
        blocks_per_dim = 3
        rank = (blocks_per_dim ** 3) * 7
        
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Strassen's 7 intermediate products for 2x2 matrices:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        #
        # Each entry in U,V,W is indexed by [row, col] where row/col are in [0,3)
        # For U: row = block_row*2 + local_row, col = block_col*2 + local_col
        # For V: row = block_col*2 + local_row, col = block_out*2 + local_col
        # For W: row = block_out*2 + local_row, col = block_row*2 + local_col
        
        # Define the 7 Strassen patterns as (U_entries, V_entries, W_entries)
        # Each entry is a dict mapping (local_row, local_col) -> coefficient
        strassen_patterns = [
            # M1 = (A11+A22)(B11+B22)
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            # M2 = (A21+A22)B11
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            # M3 = A11(B21-B22)
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            # M4 = A22(B12-B11)
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            # M5 = (A11+A12)B22
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            # M6 = (A21-A11)(B11+B12)
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            # M7 = (A12-A22)(B21+B22)
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]
        
        col_idx = 0
        for I in range(blocks_per_dim):  # block row of A / C
            for J in range(blocks_per_dim):  # block col of A / row of B
                for K in range(blocks_per_dim):  # block col of B / C
                    for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                        # Fill U: A block (I,J)
                        for (pi, pj), val in u_coeffs.items():
                            r = I * block_size + pi
                            c = J * block_size + pj
                            U[r * m + c][col_idx] += val
                        
                        # Fill V: B block (J,K)
                        for (pi, pj), val in v_coeffs.items():
                            r = J * block_size + pi
                            c = K * block_size + pj
                            V[r * p + c][col_idx] += val
                        
                        # Fill W: C block (I,K), using trace convention W[k*n+i]
                        for (pi, pj), val in w_coeffs.items():
                            k = K * block_size + pj
                            i = I * block_size + pi
                            W[k * n + i][col_idx] += val
                        
                        col_idx += 1
        
        return {"U": U, "V": V, "W": W}
    
    # Generate the tiled Strassen solution (rank 189)
    sol = get_tiled_strassen_solution()
    
    # Verify before returning
    if verify_solution(sol):
        return sol
    
    # If verification fails, try a simple identity-based decomposition (rank 216)
    # This should always work: C[i,k] = sum_j A[i,j]*B[j,k]
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
