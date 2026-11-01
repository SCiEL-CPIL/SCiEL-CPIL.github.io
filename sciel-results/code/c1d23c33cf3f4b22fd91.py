"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 44 decomposition
    # Part 1: 2x1x8 schoolbook (rank 16)
    # Part 2: 2x2x8 Strassen (rank 28)
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: j=0
    # A is 2x1, B is 1x8
    # i in 0..1, k in 0..7
    col = 0
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: j in 1,2
    # A is 2x2, B is 2x8
    # Split k into 4 blocks of size 2
    # Strassen 7 terms per block
    
    # Strassen coefficients
    # u, v, w are lists of 4 floats/ints
    # Order: (0,0), (0,1), (1,0), (1,1)
    
    strassen_terms = [
        # T1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # T2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # T3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # T4
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        # T5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # T6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # T7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    for b in range(4):
        k_base = 2 * b
        for t_idx in range(7):
            u_vec, v_vec, w_vec = strassen_terms[t_idx]
            c = col + t_idx
            
            # U: i in 0..1, j in 1..2
            # u_vec index: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # global i, j
            for idx, val in enumerate(u_vec):
                if val != 0:
                    i_prime = idx // 2
                    j_prime = idx % 2
                    i = i_prime
                    j = 1 + j_prime
                    U[i * m + j][c] = val
                    
            # V: j in 1..2, k in k_base..k_base+1
            # v_vec index: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # global j, k
            for idx, val in enumerate(v_vec):
                if val != 0:
                    j_prime = idx // 2
                    k_prime = idx % 2
                    j = 1 + j_prime
                    k = k_base + k_prime
                    V[j * p + k][c] = val
                    
            # W: k in k_base..k_base+1, i in 0..1
            # w_vec index: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # Note: w_vec corresponds to z_{i', k'}
            # idx 0: i'=0, k'=0
            # idx 1: i'=0, k'=1
            # idx 2: i'=1, k'=0
            # idx 3: i'=1, k'=1
            for idx, val in enumerate(w_vec):
                if val != 0:
                    i_prime = idx // 2
                    k_prime = idx % 2
                    i = i_prime
                    k = k_base + k_prime
                    W[k * n + i][c] = val
                    
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
