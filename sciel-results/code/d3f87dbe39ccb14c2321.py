"""Program for 2x3x8 matrix multiplication tensor decomposition."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strategy: Split the 2x3x8 problem into blocks to minimize rank.
    # We split p=8 into two p=4 blocks.
    # For each p=4 block, we split m=3 into m=2 and m=1.
    # For the m=2, p=4 part: split p=4 into two p=2 blocks.
    #   Each 2x2x2 block uses Strassen (rank 7). Total: 14.
    # For the m=1, p=4 part: 2x1x4 is schoolbook (rank 8).
    # Total per p=4 block: 14 + 8 = 22.
    # Total for p=8: 22 + 22 = 44.
    
    rank = 44
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]  # (2*3, 44) = (6, 44)
    V = [[0] * rank for _ in range(m * p)]  # (3*8, 44) = (24, 44)
    W = [[0] * rank for _ in range(p * n)]  # (8*2, 44) = (16, 44)
    
    col = 0
    
    # Strassen 2x2x2 decomposition
    # Multiplies A (2x2) by B (2x2) to get C (2x2)
    # A indices in U: (i, j) -> i*m + j where i in 0..1, j in 0..1
    # B indices in V: (j, k) -> j*p + k where j in 0..1, k in 0..1
    # C indices in W: (k, i) -> k*n + i where k in 0..1, i in 0..1
    
    def add_strassen_2x2x2(i_base, j_base, k_base):
        nonlocal col
        
        # m1 = (a + d)(e + h)
        # a = A[0,0], d = A[1,1], e = B[0,0], h = B[1,1]
        # Contributes to C[0,0] and C[1,1]
        U[i_base + 0][col] += 1
        U[i_base + 1][col] += 1
        V[j_base + 0][col] += 1
        V[j_base + p][col] += 1
        W[k_base + 0][col] += 1
        W[k_base + n][col] += 1
        col += 1
        
        # m2 = (c + d)e
        # c = A[1,0], d = A[1,1], e = B[0,0]
        # Contributes to C[1,0] and C[1,1]
        U[i_base + m][col] += 1
        U[i_base + m + 1][col] += 1
        V[j_base + 0][col] += 1
        W[k_base + n][col] += 1
        W[k_base + 2*n][col] += -1
        col += 1
        
        # m3 = a(f - h)
        # a = A[0,0], f = B[0,1], h = B[1,1]
        # Contributes to C[0,1] and C[1,1]
        U[i_base + 0][col] += 1
        V[j_base + 1][col] += 1
        V[j_base + p + 1][col] += -1
        W[k_base + 1][col] += 1
        W[k_base + n + 1][col] += 1
        col += 1
        
        # m4 = d(g - e)
        # d = A[1,1], g = B[1,0], e = B[0,0]
        # Contributes to C[1,0] and C[1,1]
        U[i_base + m + 1][col] += 1
        V[j_base + p][col] += 1
        V[j_base + 0][col] += -1
        W[k_base + n][col] += 1
        W[k_base + 2*n][col] += 1
        col += 1
        
        # m5 = (a + b)h
        # a = A[0,0], b = A[0,1], h = B[1,1]
        # Contributes to C[0,0] and C[0,1]
        U[i_base + 0][col] += 1
        U[i_base + 1][col] += 1
        V[j_base + p + 1][col] += 1
        W[k_base + 0][col] += -1
        W[k_base + 1][col] += 1
        col += 1
        
        # m6 = (c - a)(e + f)
        # c = A[1,0], a = A[0,0], e = B[0,0], f = B[0,1]
        # Contributes to C[1,1]
        U[i_base + m][col] += 1
        U[i_base + 0][col] += -1
        V[j_base + 0][col] += 1
        V[j_base + 1][col] += 1
        W[k_base + n + 1][col] += 1
        col += 1
        
        # m7 = (b - d)(g + h)
        # b = A[0,1], d = A[1,1], g = B[1,0], h = B[1,1]
        # Contributes to C[0,1]
        U[i_base + 1][col] += 1
        U[i_base + m + 1][col] += -1
        V[j_base + p][col] += 1
        V[j_base + p + 1][col] += 1
        W[k_base + 1][col] += 1
        col += 1
    
    # Schoolbook 2x1x4 decomposition
    # Multiplies A (2x1) by B (1x4) to get C (2x4)
    # A indices in U: (i, 0) -> i*m + 0
    # B indices in V: (0, k) -> 0*p + k
    # C indices in W: (k, i) -> k*n + i
    
    def add_schoolbook_2x1x4(i_base, j_base, k_base):
        nonlocal col
        for i in range(2):
            for k in range(4):
                U[i_base + i][col] = 1
                V[j_base + k][col] = 1
                W[k_base + k][col] = 1
                col += 1
    
    # Block 1: k in 0..3
    # Part A: j in 0..1, k in 0..1 -> 2x2x2 Strassen
    add_strassen_2x2x2(0, 0, 0)
    
    # Part B: j in 0..1, k in 2..3 -> 2x2x2 Strassen
    add_strassen_2x2x2(0, 0, 2)
    
    # Part C: j=2, k in 0..3 -> 2x1x4 Schoolbook
    add_schoolbook_2x1x4(0, 2, 0)
    
    # Block 2: k in 4..7
    # Part D: j in 0..1, k in 4..5 -> 2x2x2 Strassen
    add_strassen_2x2x2(0, 0, 4)
    
    # Part E: j in 0..1, k in 6..7 -> 2x2x2 Strassen
    add_strassen_2x2x2(0, 0, 6)
    
    # Part F: j=2, k in 4..7 -> 2x1x4 Schoolbook
    add_schoolbook_2x1x4(0, 2, 4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
