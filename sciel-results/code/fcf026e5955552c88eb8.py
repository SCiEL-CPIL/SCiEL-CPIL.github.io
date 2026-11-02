"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank: 7 blocks of 2x2x2 (rank 7) + 1 block of 2x2x1 (rank 4) = 49 + 4 = 53
    rank = 53
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    current_rank = 0
    
    # Strassen's algorithm for 2x2x2
    # U[i*m+j], V[j*p+k], W[k*n+i]
    # Local indices i,j in 0..1, k in 0..1
    strassen_U = [
        [[1, 0], [0, 1]], # M1: A00 + A11
        [[0, 0], [1, 1]], # M2: A10 + A11
        [[1, 0], [0, 0]], # M3: A00
        [[0, 0], [0, 1]], # M4: A11
        [[1, 1], [0, 0]], # M5: A00 + A01
        [[-1, 0], [1, 0]],# M6: A10 - A00
        [[0, 1], [0, -1]] # M7: A01 - A11
    ]
    
    strassen_V = [
        [[1, 0], [1, 0]], # M1: B00 + B11 (Note: V is j,k. B is j x k)
                           # Wait, V[j,k] coefficients for B[j,k]
                           # M1 = (A00+A11)(B00+B11).
                           # U part: A00+A11. V part: B00+B11.
                           # V[0,0]=1, V[1,1]=1.
                           # Let's re-verify Strassen V definitions.
                           # B is 2x2. Indices j (row), k (col).
                           # M1: B00 + B11 -> V[0,0]=1, V[1,1]=1.
                           # M2: B00 -> V[0,0]=1.
                           # M3: B01 - B11 -> V[0,1]=1, V[1,1]=-1.
                           # M4: B10 - B00 -> V[1,0]=1, V[0,0]=-1.
                           # M5: B11 -> V[1,1]=1.
                           # M6: B00 + B01 -> V[0,0]=1, V[0,1]=1.
                           # M7: B10 + B11 -> V[1,0]=1, V[1,1]=1.
    ]
    # Let's fix strassen_V list
    strassen_V = [
        [[1, 0], [1, 1]], # M1: B00 + B11
        [[1, 0], [0, 0]], # M2: B00
        [[0, 1], [-1, 0]],# M3: B01 - B11
        [[-1, 0], [1, 0]],# M4: B10 - B00
        [[0, 0], [0, 1]], # M5: B11
        [[1, 1], [0, 0]], # M6: B00 + B01
        [[0, 1], [0, 1]]  # M7: B10 + B11
    ]
    
    strassen_W = [
        [[1, 0], [0, 1]], # M1: C00 + C11 ? No.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # W[k*n+i] -> W[0*n+0]=C00, W[0*n+1]=C10, W[1*n+0]=C01, W[1*n+1]=C11
        # Indices in W matrix: row index is k*n + i.
        # W[0] is C00. W[1] is C10. W[2] is C01. W[3] is C11.
        # So we need W_val for each of these 4 positions.
    ]
    
    # Let's build W carefully.
    # W[r][k*n+i]
    # M1: C00=1, C11=1. Others 0. -> [1, 0, 0, 1]
    # M2: C10=1, C11=-1. Others 0. -> [0, 1, 0, -1]
    # M3: C01=1, C11=1. Others 0. -> [0, 0, 1, 1]
    # M4: C00=1, C10=1. Others 0. -> [1, 1, 0, 0]
    # M5: C00=-1, C01=1. Others 0. -> [-1, 0, 1, 0]
    # M6: C11=1. Others 0. -> [0, 0, 0, 1]
    # M7: C00=1. Others 0. -> [1, 0, 0, 0]
    
    strassen_W = [
        [1, 0, 0, 1], # M1
        [0, 1, 0, -1],# M2
        [0, 0, 1, 1], # M3
        [1, 1, 0, 0], # M4
        [-1, 0, 1, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    # Loop through 7 blocks of size 2 in k
    for g in range(7):
        for r_loc in range(7):
            r_glob = current_rank
            current_rank += 1
            
            # Fill U
            for i_loc in range(2):
                for j_loc in range(2):
                    val = strassen_U[r_loc][i_loc][j_loc]
                    if val != 0:
                        U[i_loc * 2 + j_loc][r_glob] = val
            
            # Fill V
            for j_loc in range(2):
                for k_loc in range(2):
                    val = strassen_V[r_loc][j_loc][k_loc]
                    if val != 0:
                        k_glob = g * 2 + k_loc
                        V[j_loc * 15 + k_glob][r_glob] = val
            
            # Fill W
            # W index is k_glob * n + i_loc = k_glob * 2 + i_loc
            for i_loc in range(2):
                for k_loc in range(2):
                    val = strassen_W[r_loc][k_loc * 2 + i_loc]
                    if val != 0:
                        k_glob = g * 2 + k_loc
                        W[k_glob * 2 + i_loc][r_glob] = val
                        
    # 1 block of size 1 in k (k=14)
    # Schoolbook for 2x2x1
    # Rank 4. Terms: (0,0,0), (0,1,0), (1,0,0), (1,1,0) in local (i,j,k)
    # Global k = 14.
    for i_loc in range(2):
        for j_loc in range(2):
            r_glob = current_rank
            current_rank += 1
            
            U[i_loc * 2 + j_loc][r_glob] = 1
            V[j_loc * 15 + 14][r_glob] = 1
            W[14 * 2 + i_loc][r_glob] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
