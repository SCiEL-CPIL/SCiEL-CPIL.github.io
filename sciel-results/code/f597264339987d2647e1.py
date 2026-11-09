"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 5, 5, 5
    # We'll use a block decomposition with 2+3 split
    # Total rank will be 124 (using Strassen for 2x2, schoolbook for others)
    
    # Define the blocks:
    # A11: rows 0-1, cols 0-1 (2x2)
    # A12: rows 0-1, cols 2-4 (2x3)
    # A21: rows 2-4, cols 0-1 (3x2)
    # A22: rows 2-4, cols 2-4 (3x3)
    # Similarly for B and C
    
    # Products to compute:
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Ranks:
    # A11*B11: 2x2x2, Strassen, rank 7
    # A12*B21: 2x3x2, schoolbook, rank 12
    # A11*B12: 2x2x3, schoolbook, rank 12
    # A12*B22: 2x3x3, schoolbook, rank 18
    # A21*B11: 3x2x2, schoolbook, rank 12
    # A22*B21: 3x3x2, schoolbook, rank 18
    # A21*B12: 3x2x3, schoolbook, rank 18
    # A22*B22: 3x3x3, schoolbook, rank 27
    # Total: 7 + 12 + 12 + 18 + 12 + 18 + 18 + 27 = 124
    
    total_rank = 124
    U = [[0] * total_rank for _ in range(n * m)]  # 25 x 124
    V = [[0] * total_rank for _ in range(m * p)]  # 25 x 124
    W = [[0] * total_rank for _ in range(p * n)]  # 25 x 124
    
    col = 0
    
    # Helper: add a schoolbook product for A[ar1:ar2, ac1:ac2] * B[br1:br2, bc1:bc2] -> C[cr1:cr2, cc1:cc2]
    def add_schoolbook(ar1, ar2, ac1, ac2, br1, br2, bc1, bc2, cr1, cr2, cc1, cc2):
        nonlocal col
        for ai in range(ar1, ar2):
            for aj in range(ac1, ac2):
                for bk in range(br1, br2):
                    # A[ai, aj] * B[bk, ...] -> C[ai, bk]
                    # But we need B[aj, bk] and C[ai, bk]
                    # Wait, the indices are: C[i,k] = sum_j A[i,j] * B[j,k]
                    # So for A[ai, aj] * B[aj, bk] -> C[ai, bk]
                    if aj >= br1 and aj < br2 and bk >= bc1 and bk < bc2:
                        U[ai * m + aj][col] = 1
                        V[aj * p + bk][col] = 1
                        W[bk * n + ai][col] = 1
                        col += 1
    
    # Helper: add Strassen 2x2x2 product
    def add_strassen_2x2(ar1, ac1, br1, bc1, cr1, cc1):
        nonlocal col
        # A[ar1:ar1+2, ac1:ac1+2] * B[br1:br1+2, bc1:bc1+2] -> C[cr1:cr1+2, cc1:cc1+2]
        # Strassen:
        # M1 = (A11+A22)*(B11+B22)
        # M2 = (A21+A22)*B11
        # M3 = A11*(B12-B22)
        # M4 = A22*(B21-B11)
        # M5 = (A11+A12)*B22
        # M6 = (A21-A11)*(B11+B12)
        # M7 = (A12-A22)*(B21+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # A11 = A[ar1, ac1], A12 = A[ar1, ac1+1], A21 = A[ar1+1, ac1], A22 = A[ar1+1, ac1+1]
        # B11 = B[br1, bc1], B12 = B[br1, bc1+1], B21 = B[br1+1, bc1], B22 = B[br1+1, bc1+1]
        # C11 = C[cr1, cc1], C12 = C[cr1, cc1+1], C21 = C[cr1+1, cc1], C22 = C[cr1+1, cc1+1]
        
        # M1 = (A11+A22)*(B11+B22)
        U[ar1 * m + ac1][col] = 1
        U[(ar1+1) * m + (ac1+1)][col] = 1
        V[br1 * p + bc1][col] = 1
        V[(br1+1) * p + (bc1+1)][col] = 1
        W[cc1 * n + cr1][col] = 1
        W[(cc1+1) * n + (cr1+1)][col] = 1
        col += 1
        
        # M2 = (A21+A22)*B11
        U[(ar1+1) * m + ac1][col] = 1
        U[(ar1+1) * m + (ac1+1)][col] = 1
        V[br1 * p + bc1][col] = 1
        W[cc1 * n + (cr1+1)][col] = 1
        col += 1
        
        # M3 = A11*(B12-B22)
        U[ar1 * m + ac1][col] = 1
        V[br1 * p + (bc1+1)][col] = 1
        V[(br1+1) * p + (bc1+1)][col] = -1
        W[(cc1+1) * n + cr1][col] = 1
        col += 1
        
        # M4 = A22*(B21-B11)
        U[(ar1+1) * m + (ac1+1)][col] = 1
        V[(br1+1) * p + bc1][col] = 1
        V[br1 * p + bc1][col] = -1
        W[cc1 * n + (cr1+1)][col] = 1
        col += 1
        
        # M5 = (A11+A12)*B22
        U[ar1 * m + ac1][col] = 1
        U[ar1 * m + (ac1+1)][col] = 1
        V[(br1+1) * p + (bc1+1)][col] = 1
        W[(cc1+1) * n + cr1][col] = 1
        W[(cc1+1) * n + (cr1+1)][col] = 1
        col += 1
        
        # M6 = (A21-A11)*(B11+B12)
        U[(ar1+1) * m + ac1][col] = 1
        U[ar1 * m + ac1][col] = -1
        V[br1 * p + bc1][col] = 1
        V[br1 * p + (bc1+1)][col] = 1
        W[(cc1+1) * n + (cr1+1)][col] = 1
        col += 1
        
        # M7 = (A12-A22)*(B21+B22)
        U[ar1 * m + (ac1+1)][col] = 1
        U[(ar1+1) * m + (ac1+1)][col] = -1
        V[(br1+1) * p + bc1][col] = 1
        V[(br1+1) * p + (bc1+1)][col] = 1
        W[cc1 * n + cr1][col] = 1
        W[(cc1+1) * n + (cr1+1)][col] = 1
        col += 1
    
    # Now add the products:
    
    # C11 = A11*B11 + A12*B21
    # A11*B11: 2x2x2, Strassen
    add_strassen_2x2(0, 0, 0, 0, 0, 0)  # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # A12*B21: 2x3x2, schoolbook
    # A[0:2, 2:5] * B[2:5, 0:2] -> C[0:2, 0:2]
    # C[i,k] = sum_j A[i,j] * B[j,k] for i=0,1; j=2,3,4; k=0,1
    for ai in range(0, 2):
        for aj in range(2, 5):
            for bk in range(0, 2):
                U[ai * m + aj][col] = 1
                V[aj * p + bk][col] = 1
                W[bk * n + ai][col] = 1
                col += 1
    
    # C12 = A11*B12 + A12*B22
    # A11*B12: 2x2x3, schoolbook
    # A[0:2, 0:2] * B[0:2, 2:5] -> C[0:2, 2:5]
    for ai in range(0, 2):
        for aj in range(0, 2):
            for bk in range(2, 5):
                U[ai * m + aj][col] = 1
                V[aj * p + bk][col] = 1
                W[bk * n + ai][col] = 1
                col += 1
    # A12*B22: 2x3x3, schoolbook
    # A[0:2, 2:5] * B[2:5, 2:5] -> C[0:2, 2:5]
    for ai in range(0, 2):
        for aj in range(2, 5):
            for bk in range(2, 5):
                U[ai * m + aj][col] = 1
                V[aj * p + bk][col] = 1
                W[bk * n + ai][col] = 1
                col += 1
    
    # C21 = A21*B11 + A22*B21
    # A21*B11: 3x2x2, schoolbook
    # A[2:5, 0:2] * B[0:2, 0:2] -> C[2:5, 0:2]
    for ai in range(2, 5):
        for aj in range(0, 2):
            for bk in range(0, 2):
                U[ai * m + aj][col] = 1
                V[aj * p + bk][col] = 1
                W[bk * n + ai][col] = 1
                col += 1
    # A22*B21: 3x3x2, schoolbook
    # A[2:5, 2:5] * B[2:5, 0:2] -> C[2:5, 0:2]
    for ai in range(2, 5):
        for aj in range(2, 5):
            for bk in range(0, 2):
                U[ai * m + aj][col] = 1
                V[aj * p + bk][col] = 1
                W[bk * n + ai][col] = 1
                col += 1
    
    # C22 = A21*B12 + A22*B22
    # A21*B12: 3x2x3, schoolbook
    # A[2:5, 0:2] * B[0:2, 2:5] -> C[2:5, 2:5]
    for ai in range(2, 5):
        for aj in range(0, 2):
            for bk in range(2, 5):
                U[ai * m + aj][col] = 1
                V[aj * p + bk][col] = 1
                W[bk * n + ai][col] = 1
                col += 1
    # A22*B22: 3x3x3, schoolbook
    # A[2:5, 2:5] * B[2:5, 2:5] -> C[2:5, 2:5]
    for ai in range(2, 5):
        for aj in range(2, 5):
            for bk in range(2, 5):
                U[ai * m + aj][col] = 1
                V[aj * p + bk][col] = 1
                W[bk * n + ai][col] = 1
                col += 1
    
    assert col == total_rank, f"Expected {total_rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
