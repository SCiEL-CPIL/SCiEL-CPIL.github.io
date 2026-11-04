import time
import sys

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use block decomposition: 6x6 is 2x2 blocks of 3x3 matrices.
    # Outer: 2x2 matrix multiply using Strassen (Rank 7).
    # Inner: 3x3 matrix multiply using Schoolbook (Rank 27).
    # Total Rank = 7 * 27 = 189.
    
    block_size = 3
    num_blocks = 2
    dim = n # 6
    
    rank_inner = block_size ** 3 # 27
    rank_outer = 7 # Strassen rank for 2x2
    total_rank = rank_inner * rank_outer
    
    # Initialize matrices
    # U: (n*m, rank) = (36, 189)
    # V: (m*p, rank) = (36, 189)
    # W: (p*n, rank) = (36, 189)
    U = [[0] * total_rank for _ in range(dim * dim)]
    V = [[0] * total_rank for _ in range(dim * dim)]
    W = [[0] * total_rank for _ in range(dim * dim)]
    
    col = 0
    
    # Strassen steps for 2x2 block matrix multiplication
    # A, B, C are 2x2 blocks of 3x3 matrices.
    # Each step defines linear combinations of blocks.
    steps = [
        # M1 = (A00 + A11) * (B00 + B11)
        {
            'A': { (0,0): 1, (1,1): 1 },
            'B': { (0,0): 1, (1,1): 1 },
            'C': { (0,0): 1, (1,1): 1 }
        },
        # M2 = (A10 + A11) * B00
        {
            'A': { (1,0): 1, (1,1): 1 },
            'B': { (0,0): 1 },
            'C': { (1,0): 1, (1,1): -1 }
        },
        # M3 = A00 * (B01 - B11)
        {
            'A': { (0,0): 1 },
            'B': { (0,1): 1, (1,1): -1 },
            'C': { (0,1): 1, (1,1): 1 }
        },
        # M4 = A11 * (B10 - B00)
        {
            'A': { (1,1): 1 },
            'B': { (1,0): 1, (0,0): -1 },
            'C': { (0,0): 1, (1,0): 1 }
        },
        # M5 = (A00 + A01) * B11
        {
            'A': { (0,0): 1, (0,1): 1 },
            'B': { (1,1): 1 },
            'C': { (0,0): -1, (0,1): 1 }
        },
        # M6 = (A10 - A00) * (B00 + B01)
        {
            'A': { (1,0): 1, (0,0): -1 },
            'B': { (0,0): 1, (0,1): 1 },
            'C': { (1,1): 1 }
        },
        # M7 = (A01 - A11) * (B10 + B11)
        {
            'A': { (0,1): 1, (1,1): -1 },
            'B': { (1,0): 1, (1,1): 1 },
            'C': { (0,0): 1 }
        }
    ]
    
    # Pre-calculate indices for speed
    # For a given step and local indices (u, v, w), we need to update U, V, W.
    # u: row of X, col of M
    # v: col of X, row of Y
    # w: col of Y, col of M
    
    for step in steps:
        A_coeffs = step['A']
        B_coeffs = step['B']
        C_coeffs = step['C']
        
        for u in range(block_size):
            for v in range(block_size):
                for w in range(block_size):
                    # Update U
                    # X[u,v] depends on A blocks
                    for (I, J), ca in A_coeffs.items():
                        if ca != 0:
                            i_global = I * block_size + u
                            j_global = J * block_size + v
                            idx_u = i_global * dim + j_global
                            U[idx_u][col] += ca
                    
                    # Update V
                    # Y[v,w] depends on B blocks
                    for (J, K), cb in B_coeffs.items():
                        if cb != 0:
                            j_global = J * block_size + v
                            k_global = K * block_size + w
                            idx_v = j_global * dim + k_global
                            V[idx_v][col] += cb
                    
                    # Update W
                    # M[u,w] contributes to C blocks
                    for (I, K), cc in C_coeffs.items():
                        if cc != 0:
                            i_global = I * block_size + u
                            k_global = K * block_size + w
                            # Task convention: W[k*n+i, r]
                            idx_w = k_global * dim + i_global
                            W[idx_w][col] += cc
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
