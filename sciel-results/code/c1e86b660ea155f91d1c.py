"""Matrix multiplication 2x3 by 3x9: rank-50 decomposition via Strassen on 2x2 blocks."""
DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = 2, 3, 9
    rank = 50
    U = [[0] * rank for _ in range(n * m)]  # (6, 50)
    V = [[0] * rank for _ in range(m * p)]  # (27, 50)
    W = [[0] * rank for _ in range(p * n)]  # (18, 50)

    col = 0

    # Part 1: (2x1) * (1x9) -> rank 18
    # A is 2x1 (col 0 of A), B is 1x9 (row 0 of B)
    # C[i,k] = A[i,0] * B[0,k] for i in {0,1}, k in 0..8
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    # col is now 18

    # Part 2: (2x2) * (2x9) -> rank 32
    # A is 2x2 (cols 1,2 of A), B is 2x9 (rows 1,2 of B)
    # Split B into B1 (2x4, cols 0-3) and B2 (2x5, cols 4-8)
    # B1 split into two 2x2 blocks: B1a (cols 0,1), B1b (cols 2,3)
    # B2 split into B2a (2x2, cols 4,5) and B2b (2x3, cols 6,7,8)
    # B2b further split into B2ba (2x2, cols 6,7) and B2bc (2x1, col 8)

    # Strassen coefficients for 2x2 * 2x2:
    # M1 = (a+c)(e+g)
    # M2 = (c+d)(g+h)
    # M3 = a(f-g)
    # M4 = d(e-f)
    # M5 = (a+b)(h-g)
    # M6 = (c+a)(e+f)
    # M7 = (b+d)(g+h)
    # C[0,0] = M1+M4-M5+M7
    # C[0,1] = M3+M5
    # C[1,0] = M2+M4
    # C[1,1] = M1-M2+M3+M6

    # For a 2x2 * 2x2 block with A at A-rows {ar0, ar1} and A-cols {ac0, ac1},
    # B at B-rows {br0, br1} and B-cols {bc0, bc1}:
    # A elements: A[ar0,ac0]=a, A[ar0,ac1]=b, A[ar1,ac0]=c, A[ar1,ac1]=d
    # B elements: B[br0,bc0]=e, B[br0,bc1]=f, B[br1,bc0]=g, B[br1,bc1]=h
    # Output C[ar0, bc0] and C[ar0, bc1] are in row ar0 of C (which is row i of full C)
    # C[ar1, bc0] and C[ar1, bc1] are in row ar1 of C

    def add_strassen_2x2_2x2(ar0, ar1, ac0, ac1, br0, br1, bc0, bc1):
        nonlocal col
        # M1 = (a+c)(e+g)
        U[ar0 * m + ac0][col] = 1
        U[ar1 * m + ac0][col] = 1
        V[br0 * p + bc0][col] = 1
        V[br1 * p + bc0][col] = 1
        W[bc0 * n + ar0][col] = 1
        W[bc1 * n + ar0][col] = -1
        W[bc0 * n + ar1][col] = 1
        W[bc1 * n + ar1][col] = 1
        col += 1
        # M2 = (c+d)(g+h)
        U[ar1 * m + ac0][col] = 1
        U[ar1 * m + ac1][col] = 1
        V[br1 * p + bc0][col] = 1
        V[br1 * p + bc1][col] = 1
        W[bc0 * n + ar1][col] = 1
        W[bc1 * n + ar1][col] = -1
        col += 1
        # M3 = a(f-g)
        U[ar0 * m + ac0][col] = 1
        V[br0 * p + bc1][col] = 1
        V[br1 * p + bc0][col] = -1
        W[bc1 * n + ar0][col] = 1
        W[bc1 * n + ar1][col] = 1
        col += 1
        # M4 = d(e-f)
        U[ar1 * m + ac1][col] = 1
        V[br0 * p + bc0][col] = 1
        V[br0 * p + bc1][col] = -1
        W[bc0 * n + ar0][col] = 1
        W[bc0 * n + ar1][col] = 1
        col += 1
        # M5 = (a+b)(h-g)
        U[ar0 * m + ac0][col] = 1
        U[ar0 * m + ac1][col] = 1
        V[br1 * p + bc1][col] = 1
        V[br1 * p + bc0][col] = -1
        W[bc1 * n + ar0][col] = 1
        W[bc0 * n + ar0][col] = -1
        col += 1
        # M6 = (c+a)(e+f)
        U[ar1 * m + ac0][col] = 1
        U[ar0 * m + ac0][col] = 1
        V[br0 * p + bc0][col] = 1
        V[br0 * p + bc1][col] = 1
        W[bc1 * n + ar1][col] = 1
        col += 1
        # M7 = (b+d)(g+h)
        U[ar0 * m + ac1][col] = 1
        U[ar1 * m + ac1][col] = 1
        V[br1 * p + bc0][col] = 1
        V[br1 * p + bc1][col] = 1
        W[bc0 * n + ar0][col] = 1
        W[bc1 * n + ar0][col] = 1
        col += 1

    def add_schoolbook_2x1_1x2(ar0, ar1, ac0, bc0, bc1, br0):
        """For 2x1 * 1x2: C[ar0, bc*] = A[ar0, ac0] * B[br0, bc*], C[ar1, bc*] = A[ar1, ac0] * B[br0, bc*]"""
        nonlocal col
        # This is actually two separate rank-1 terms if we think of it as 2x1 * 1x2
        # C[ar0, bc0] = A[ar0,ac0]*B[br0,bc0]
        # C[ar0, bc1] = A[ar0,ac0]*B[br0,bc1]
        # C[ar1, bc0] = A[ar1,ac0]*B[br0,bc0]
        # C[ar1, bc1] = A[ar1,ac0]*B[br0,bc1]
        # These are 4 independent products
        for ar in [ar0, ar1]:
            for bc in [bc0, bc1]:
                U[ar * m + ac0][col] = 1
                V[br0 * p + bc][col] = 1
                W[bc * n + ar][col] = 1
                col += 1

    # A2: rows 0,1 of A, cols 1,2 (ac0=1, ac1=2)
    # B2: rows 1,2 of B, cols 0..8 (br0=1, br1=2)
    
    # B1a: cols 0,1 (bc0=0, bc1=1)
    add_strassen_2x2_2x2(0, 1, 1, 2, 1, 2, 0, 1)
    
    # B1b: cols 2,3 (bc0=2, bc1=3)
    add_strassen_2x2_2x2(0, 1, 1, 2, 1, 2, 2, 3)
    
    # B2a: cols 4,5 (bc0=4, bc1=5)
    add_strassen_2x2_2x2(0, 1, 1, 2, 1, 2, 4, 5)
    
    # B2ba: cols 6,7 (bc0=6, bc1=7)
    add_strassen_2x2_2x2(0, 1, 1, 2, 1, 2, 6, 7)
    
    # B2bc: col 8 only - 2x1 * 1x1... wait, B2b is 2x3 (cols 6,7,8)
    # I split B2b into 2x2 (cols 6,7) and 2x1 (col 8)
    # For the 2x1 part: A2 (2x2) * B_col8 (2x1)
    # C[ar, 8] = sum_{ac in {1,2}} A[ar, ac] * B[?, 8]
    # But B has rows 1,2. So B_col8 is (B[1,8], B[2,8])
    # C[0,8] = A[0,1]*B[1,8] + A[0,2]*B[2,8]
    # C[1,8] = A[1,1]*B[1,8] + A[1,2]*B[2,8]
    # This is 4 independent products (schoolbook for 2x2 * 2x1)
    # Wait, 2x2 * 2x1 is rank 4 (schoolbook)
    for ac in [1, 2]:
        for br in [1, 2]:
            for ar in [0, 1]:
                U[ar * m + ac][col] = 1
                V[br * p + 8][col] = 1
                W[8 * n + ar][col] = 1
                col += 1
    # That's 4 more, total: 18 + 7*4 + 4 = 18 + 28 + 4 = 50. Good.

    assert col == rank, f"Expected rank {rank}, got {col}"
    return {"U": U, "V": V, "W": W}
