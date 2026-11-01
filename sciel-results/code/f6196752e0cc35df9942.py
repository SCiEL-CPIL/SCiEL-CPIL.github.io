"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS  # 2, 2, 14
    num_blocks = p // 2  # 7 blocks of 2x2
    
    # Strassen's algorithm for 2x2 matrix multiplication
    # The tensor for 2x2 by 2x14 multiplication is the direct sum of 7
    # 2x2 by 2x2 multiplication tensors. The rank of the direct sum is the
    # sum of the ranks. Rank(2x2x2) = 7 (Strassen).
    # Total rank = 7 * 7 = 49.
    
    rank = num_blocks * 7
    
    # U: shape (n*m, rank) = (4, rank)
    # V: shape (m*p, rank) = (28, rank)
    # W: shape (p*n, rank) = (28, rank)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms: (a, b, c, d, e, f, g, h, c11, c12, c21, c22)
    # Coefficients for A, B, and C in the bilinear expansion.
    strassen_terms = [
        (1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1),
        (0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, -1),
        (1, 0, 0, 0, 0, 1, 0, -1, 0, 1, 0, 1),
        (0, 0, 0, 1, -1, 0, 1, 0, 1, 0, 1, 0),
        (1, 1, 0, 0, 0, 0, 0, 1, -1, 1, 0, 0),
        (-1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1),
        (0, 1, 0, -1, 0, 0, 1, 1, 1, 0, 0, 0),
    ]
    
    for block in range(num_blocks):
        if time.time() > deadline:
            # Return best found so far (which is complete if we haven't failed)
            # In this static construction, we just return the current state.
            # Since the loop is small, this is unlikely to trigger.
            break
            
        for t_idx, term in enumerate(strassen_terms):
            col = block * 7 + t_idx
            a_c, b_c, c_c, d_c, e_c, f_c, g_c, h_c, c11_c, c12_c, c21_c, c22_c = term
            
            # A part (shared across blocks in terms of vector definition,
            # but each term in the sum is distinct)
            U[0][col] = a_c
            U[1][col] = b_c
            U[2][col] = c_c
            U[3][col] = d_c
            
            # B part (disjoint supports per block)
            V[0 * p + 2 * block][col] = e_c
            V[0 * p + 2 * block + 1][col] = f_c
            V[1 * p + 2 * block][col] = g_c
            V[1 * p + 2 * block + 1][col] = h_c
            
            # C part (disjoint supports per block)
            W[2 * block * n + 0][col] = c11_c
            W[(2 * block + 1) * n + 0][col] = c12_c
            W[2 * block * n + 1][col] = c21_c
            W[(2 * block + 1) * n + 1][col] = c22_c
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
