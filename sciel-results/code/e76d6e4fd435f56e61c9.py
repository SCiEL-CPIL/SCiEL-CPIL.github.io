"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0

    def add_strassen(A_idxs, B_idxs, C_idxs):
        nonlocal col
        a, b, c, d = A_idxs
        e, f, g, h = B_idxs
        c00, c01, c10, c11 = C_idxs
        
        # M1 = (a+d)(e+h)
        U[a][col] += 1; U[d][col] += 1
        V[e][col] += 1; V[h][col] += 1
        W[c00][col] += 1; W[c11][col] += 1
        col += 1
        
        # M2 = (c+d)e
        U[c][col] += 1; U[d][col] += 1
        V[e][col] += 1
        W[c10][col] += 1; W[c11][col] -= 1
        col += 1
        
        # M3 = a(f-h)
        U[a][col] += 1
        V[f][col] += 1; V[h][col] -= 1
        W[c01][col] += 1; W[c11][col] += 1
        col += 1
        
        # M4 = d(g-e)
        U[d][col] += 1
        V[g][col] += 1; V[e][col] -= 1
        W[c00][col] += 1; W[c10][col] += 1
        col += 1
        
        # M5 = (a+b)h
        U[a][col] += 1; U[b][col] += 1
        V[h][col] += 1
        W[c00][col] -= 1; W[c01][col] += 1
        col += 1
        
        # M6 = (c-a)(e+f)
        U[c][col] += 1; U[a][col] -= 1
        V[e][col] += 1; V[f][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M7 = (b-d)(g+h)
        U[b][col] += 1; U[d][col] -= 1
        V[g][col] += 1; V[h][col] += 1
        W[c00][col] += 1
        col += 1

    def add_trivial_1x2x2(A_idxs, B_idxs, C_idxs):
        nonlocal col
        a, b = A_idxs
        e, f, g, h = B_idxs
        c0, c1 = C_idxs
        
        # ae
        U[a][col] += 1
        V[e][col] += 1
        W[c0][col] += 1
        col += 1
        
        # bg
        U[b][col] += 1
        V[g][col] += 1
        W[c0][col] += 1
        col += 1
        
        # af
        U[a][col] += 1
        V[f][col] += 1
        W[c1][col] += 1
        col += 1
        
        # bh
        U[b][col] += 1
        V[h][col] += 1
        W[c1][col] += 1
        col += 1

    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    # A_T1: [0,0], [0,1], [1,0], [1,1] -> 0, 1, 4, 5
    # A_T2: [0,2], [0,3], [1,2], [1,3] -> 2, 3, 6, 7
    # A_B1: [2,0], [2,1] -> 8, 9
    # A_B2: [2,2], [2,3] -> 10, 11
    
    # B_11: [0,0], [0,1], [1,0], [1,1] -> 0, 1, 4, 5
    # B_12: [0,2], [0,3], [1,2], [1,3] -> 2, 3, 6, 7
    # B_21: [2,0], [2,1], [3,0], [3,1] -> 8, 9, 12, 13
    # B_22: [2,2], [2,3], [3,2], [3,3] -> 10, 11, 14, 15
    
    # C indices for P1/P3 (cols 0,1):
    # C[0,0]=0, C[0,1]=3, C[1,0]=1, C[1,1]=4
    # C indices for P2/P4 (cols 2,3):
    # C[0,2]=6, C[0,3]=9, C[1,2]=7, C[1,3]=10
    
    # C indices for Q1/Q3 (cols 0,1):
    # C[2,0]=2, C[2,1]=5
    # C indices for Q2/Q4 (cols 2,3):
    # C[2,2]=8, C[2,3]=11
    
    # P1: A_T1 * B_11
    add_strassen([0, 1, 4, 5], [0, 1, 4, 5], [0, 3, 1, 4])
    # P2: A_T1 * B_12
    add_strassen([0, 1, 4, 5], [2, 3, 6, 7], [6, 9, 7, 10])
    # P3: A_T2 * B_21
    add_strassen([2, 3, 6, 7], [8, 9, 12, 13], [0, 3, 1, 4])
    # P4: A_T2 * B_22
    add_strassen([2, 3, 6, 7], [10, 11, 14, 15], [6, 9, 7, 10])
    
    # Q1: A_B1 * B_11
    add_trivial_1x2x2([8, 9], [0, 1, 4, 5], [2, 5])
    # Q2: A_B1 * B_12
    add_trivial_1x2x2([8, 9], [2, 3, 6, 7], [8, 11])
    # Q3: A_B2 * B_21
    add_trivial_1x2x2([10, 11], [8, 9, 12, 13], [2, 5])
    # Q4: A_B2 * B_22
    add_trivial_1x2x2([10, 11], [10, 11, 14, 15], [8, 11])
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
