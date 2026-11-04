"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Use 3x3 blocks with Laderman's rank-23 algorithm for 3x3 matrix mult.
    # 6x6 = 2x2 blocks of 3x3, so 8 block multiplications, each rank 23.
    # Total rank = 8 * 23 = 184.
    
    block_size = 3
    blocks_per_dim = 2  # 6 / 3
    
    rank = (blocks_per_dim ** 3) * 23
    
    # Laderman's rank-23 algorithm for 3x3 matrix multiplication
    # Each term is (U, V, W) where U, V, W are 3x3 matrices
    # such that sum_t U_t[i,j] * V_t[j,k] * W_t[k,i] = 1 for all i,j,k
    
    # Laderman's 23 products for 3x3 matrix multiplication
    # A and B are 3x3, C = A*B
    # Using the standard Laderman construction
    
    # Let A = [[a00,a01,a02],[a10,a11,a12],[a20,a21,a22]]
    # Let B = [[b00,b01,b02],[b10,b11,b12],[b20,b21,b22]]
    # C = [[c00,c01,c02],[c10,c11,c12],[c20,c21,c22]]
    
    # Laderman's 23 bilinear forms:
    # I'll define them as sparse (row,col)->value dictionaries for U, V, W
    
    laderman_terms = []
    
    # Term 1: (a00+a11+a22)(b00+b11+b22) -> contributes to c00, c11, c22
    # U: (0,0)=1, (1,1)=1, (2,2)=1
    # V: (0,0)=1, (1,1)=1, (2,2)=1  
    # W: (0,0)=1, (1,1)=1, (2,2)=1
    laderman_terms.append((
        {(0,0):1, (1,1):1, (2,2):1},
        {(0,0):1, (1,1):1, (2,2):1},
        {(0,0):1, (1,1):1, (2,2):1}
    ))
    
    # Term 2: (a01+a12+a20)(b01+b12+b20) -> contributes to c01, c12, c20
    laderman_terms.append((
        {(0,1):1, (1,2):1, (2,0):1},
        {(0,1):1, (1,2):1, (2,0):1},
        {(0,1):1, (1,2):1, (2,0):1}
    ))
    
    # Term 3: (a02+a10+a21)(b02+b10+b21) -> contributes to c02, c10, c21
    laderman_terms.append((
        {(0,2):1, (1,0):1, (2,1):1},
        {(0,2):1, (1,0):1, (2,1):1},
        {(0,2):1, (1,0):1, (2,1):1}
    ))
    
    # Term 4: (a00+a01+a11)(b11+b20+b21) -> complex
    # This is getting too complex to write from memory correctly.
    
    # Let me use a verified approach instead. I'll use a known correct set.
    
    # Actually, let me fall back to a safer strategy.
    # Use Strassen 2x2 for rank 189, which is verified to work.
    
    # But first, let me try to verify if I can construct a working Laderman.
    # Given the complexity, I'll use the Strassen 2x2 approach which is known to work.
    
    block_size = 2
    blocks_per_dim = 3
    
    rank = (blocks_per_dim ** 3) * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen patterns for 2x2 matrices
    strassen_patterns = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    col_idx = 0
    for I in range(blocks_per_dim):
        for J in range(blocks_per_dim):
            for K in range(blocks_per_dim):
                for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                    for (pi, pj), val in u_coeffs.items():
                        r = I * block_size + pi
                        c = J * block_size + pj
                        U[r * m + c][col_idx] += val
                    for (pi, pj), val in v_coeffs.items():
                        r = J * block_size + pi
                        c = K * block_size + pj
                        V[r * p + c][col_idx] += val
                    for (pi, pj), val in w_coeffs.items():
                        k = K * block_size + pj
                        i = I * block_size + pi
                        W[k * n + i][col_idx] += val
                    col_idx += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
