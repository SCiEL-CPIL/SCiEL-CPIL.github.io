"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    n, m, p = DIMENSIONS
    # Using Strassen decomposition for 2x2 blocks.
    # 6x6 matrix is 3x3 blocks of 2x2.
    # 27 block multiplications, each of rank 7.
    # Total rank = 27 * 7 = 189.
    
    block_size = 2
    blocks_per_dim = 3  # 6 / 2
    
    rank = (blocks_per_dim ** 3) * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen patterns for 2x2 matrices
    # Each pattern is a tuple: (u_coeffs, v_coeffs, w_coeffs)
    # coeffs are dicts mapping (row, col) to value
    strassen_patterns = [
        # M0 = (A00+A11)(B00+B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M1 = (A10+A11)B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M2 = A00(B01-B11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M3 = A11(B10-B00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M4 = (A00+A01)B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M5 = (A10-A00)(B00+B01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M6 = (A01-A11)(B10+B11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    col_idx = 0
    for I in range(blocks_per_dim):
        for J in range(blocks_per_dim):
            for K in range(blocks_per_dim):
                for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                    if time.time() > deadline:
                        # If we run out of time, we return what we have.
                        # But the construction is deterministic and fast, so this shouldn't happen.
                        break
                    
                    # Update U: A is n x m, flattened as row-major: index = i*m + j
                    for (pi, pj), val in u_coeffs.items():
                        r = I * block_size + pi
                        c = J * block_size + pj
                        U[r * m + c][col_idx] += val
                    
                    # Update V: B is m x p, flattened as row-major: index = j*p + k
                    for (pi, pj), val in v_coeffs.items():
                        r = J * block_size + pi
                        c = K * block_size + pj
                        V[r * p + c][col_idx] += val
                    
                    # Update W: C is p x n (output), index is k*n + i per trace convention
                    for (pi, pj), val in w_coeffs.items():
                        i = I * block_size + pi
                        k = K * block_size + pj
                        W[k * n + i][col_idx] += val
                    
                    col_idx += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
