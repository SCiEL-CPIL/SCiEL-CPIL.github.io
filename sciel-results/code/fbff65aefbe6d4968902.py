"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Strassen (rank 7) at 2x2 block level x schoolbook 3x3 (rank 27) = 189 total rank.
    Matrix multiplication: A (6x6) * B (6x6) = C (6x6).
    Partition into 2x2 blocks of size 3x3.
    
    Strassen's algorithm for 2x2 block matrices:
    A = [[A11, A12], [A21, A22]]
    B = [[B11, B12], [B21, B22]]
    
    M1 = (A11 + A22) * (B11 + B22)
    M2 = (A21 + A22) * B11
    M3 = A11 * (B12 - B22)
    M4 = A22 * (B21 - B11)
    M5 = (A11 + A12) * B22
    M6 = (A21 - A11) * (B11 + B12)
    M7 = (A12 - A22) * (B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    n, m, p = DIMENSIONS
    block_size = 3
    num_blocks = 2
    
    # Strassen coefficients for 2x2 block level
    # alpha[s, a, b] for A blocks
    # beta[s, c, d] for B blocks  
    # gamma[s, e, f] for C blocks
    
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # alpha[s][a][b] where s=0..6, a,b in {0,1}
    alpha = [
        # M1: A11 + A22
        {(0,0): 1, (1,1): 1},
        # M2: A21 + A22
        {(1,0): 1, (1,1): 1},
        # M3: A11
        {(0,0): 1},
        # M4: A22
        {(1,1): 1},
        # M5: A11 + A12
        {(0,0): 1, (0,1): 1},
        # M6: A21 - A11
        {(1,0): 1, (0,0): -1},
        # M7: A12 - A22
        {(0,1): 1, (1,1): -1},
    ]
    
    # beta[s][c][d] where s=0..6, c,d in {0,1}
    beta = [
        # M1: B11 + B22
        {(0,0): 1, (1,1): 1},
        # M2: B11
        {(0,0): 1},
        # M3: B12 - B22
        {(0,1): 1, (1,1): -1},
        # M4: B21 - B11
        {(1,0): 1, (0,0): -1},
        # M5: B22
        {(1,1): 1},
        # M6: B11 + B12
        {(0,0): 1, (0,1): 1},
        # M7: B21 + B22
        {(1,0): 1, (1,1): 1},
    ]
    
    # gamma[s][e][f] where s=0..6, e,f in {0,1}
    gamma = [
        # M1 -> C11 + C22
        {(0,0): 1, (1,1): 1},
        # M2 -> C21 + C22 (with -1 for C22)
        {(1,0): 1, (1,1): -1},
        # M3 -> C12 + C22
        {(0,1): 1, (1,1): 1},
        # M4 -> C11 + C21
        {(0,0): 1, (1,0): 1},
        # M5 -> C11 (-1) + C12
        {(0,0): -1, (0,1): 1},
        # M6 -> C22
        {(1,1): 1},
        # M7 -> C11
        {(0,0): 1},
    ]
    
    num_s = 7
    inner_rank = block_size ** 3  # 27 for schoolbook 3x3
    total_rank = num_s * inner_rank  # 189
    
    U = [[0] * total_rank for _ in range(n * m)]  # (36, 189)
    V = [[0] * total_rank for _ in range(m * p)]  # (36, 189)
    W = [[0] * total_rank for _ in range(p * n)]  # (36, 189)
    
    for s in range(num_s):
        for r_prime in range(inner_rank):
            r = s * inner_rank + r_prime
            
            # Build U contribution for this (s, r_prime)
            # U index: (i_global, j_global) where i_global = a*3+i, j_global = b*3+j
            # U[i_global * m + j_global][r] += alpha[s,a,b] * U3[i,j][r_prime]
            for a in range(num_blocks):
                for b in range(num_blocks):
                    coeff_a = alpha[s].get((a, b), 0)
                    if coeff_a == 0:
                        continue
                    for i in range(block_size):
                        for j in range(block_size):
                            # Schoolbook U3: U3[i*3+j, r_prime] = 1 if r_prime == i*3+j
                            if r_prime == i * block_size + j:
                                i_global = a * block_size + i
                                j_global = b * block_size + j
                                U[i_global * m + j_global][r] += coeff_a
            
            # Build V contribution
            # V index: (j_global, k_global) where j_global = c*3+j, k_global = d*3+k
            # V[j_global * p + k_global][r] += beta[s,c,d] * V3[j,k][r_prime]
            for c in range(num_blocks):
                for d in range(num_blocks):
                    coeff_b = beta[s].get((c, d), 0)
                    if coeff_b == 0:
                        continue
                    for j in range(block_size):
                        for k in range(block_size):
                            if r_prime == j * block_size + k:
                                j_global = c * block_size + j
                                k_global = d * block_size + k
                                V[j_global * p + k_global][r] += coeff_b
            
            # Build W contribution
            # W index: (k_global, i_global) where k_global = e*3+k, i_global = f*3+i
            # W[k_global * n + i_global][r] += gamma[s,e,f] * W3[k,i][r_prime]
            for e in range(num_blocks):
                for f in range(num_blocks):
                    coeff_g = gamma[s].get((e, f), 0)
                    if coeff_g == 0:
                        continue
                    for k in range(block_size):
                        for i in range(block_size):
                            if r_prime == k * block_size + i:
                                k_global = e * block_size + k
                                i_global = f * block_size + i
                                W[k_global * n + i_global][r] += coeff_g
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
