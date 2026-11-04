"""
Program to find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x4 matrix.
Dimensions: n=2, m=4, p=4.
Uses Strassen's algorithm decomposition for 2x2 blocks to achieve rank 28.
"""

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 terms for 2x2 matrix multiplication
    # Local indices: a=0, b=1, c=2, d=3
    #                e=0, f=1, g=2, h=3
    #                C11=0, C12=1, C21=2, C22=3
    strassen_terms = [
        # (u, v, w)
        ([1, 0, 0, 1], [1, 1, 0, 0], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]), # M3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # M5
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # M7
    ]
    
    # Mappings for the 4 blocks
    # Block 1: A[:, 0:2] * B[0:2, :]
    # Block 2: A[:, 2:4] * B[2:4, :]
    # Each block splits B into two 2x2 column halves.
    
    mappings = [
        # (A_map, B_map, C_map)
        # A_map: local 0..3 to global A index
        # B_map: local 0..3 to global B index
        # C_map: local 0..3 to global C index
        
        # Block 1, Cols 0-1
        ([0, 1, 4, 5], [0, 1, 4, 5], [0, 2, 1, 3]),
        # Block 1, Cols 2-3
        ([0, 1, 4, 5], [2, 3, 6, 7], [4, 6, 5, 7]),
        # Block 2, Cols 0-1
        ([2, 3, 6, 7], [8, 9, 12, 13], [0, 2, 1, 3]),
        # Block 2, Cols 2-3
        ([2, 3, 6, 7], [10, 11, 14, 15], [4, 6, 5, 7]),
    ]
    
    col = 0
    for a_map, b_map, c_map in mappings:
        for u_local, v_local, w_local in strassen_terms:
            for l in range(4):
                U[a_map[l]][col] += u_local[l]
                V[b_map[l]][col] += v_local[l]
                W[c_map[l]][col] += w_local[l]
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
