"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    nm, mp, pn = n * m, m * p, p * n  # 16, 28, 28
    
    # Strategy: 
    # Part 1: 4x4 x 4x4 multiplication using Strassen (rank 7) with schoolbook inner (rank 4)
    #   Total rank for this part: 7 * 4 = 28
    # Part 2: 4x4 x 4x3 multiplication using schoolbook
    #   Total rank for this part: 4 * 4 * 3 = 48
    # Grand total rank: 28 + 48 = 76
    
    rank = 76
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    
    # Part 1: Strassen for 4x4 x 4x4 (columns 0-3 of B)
    # Block definitions:
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4]
    # A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4]
    # B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    # C11 = C[0:2, 0:2], C12 = C[0:2, 2:4]
    # C21 = C[2:4, 0:2], C22 = C[2:4, 2:4]
    
    # Strassen formulas:
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
    
    # For each Mi, we compute a 2x2 matrix product using schoolbook (4 scalar mults)
    # Each scalar mult contributes one rank-1 term to U, V, W
    
    def add_2x2_product(A_terms, B_terms, C_row_off, C_col_off, c_coeff):
        """
        A_terms: list of (i, j, coeff) for the 2x2 matrix X
                 X[di,dj] = sum of coeff * A[i,j]
        B_terms: list of (j, k, coeff) for the 2x2 matrix Y
                 Y[dj,dk] = sum of coeff * B[j,k]
        C_row_off, C_col_off: offset in C for the result
        c_coeff: coefficient for W
        
        For schoolbook 2x2 mult:
        C[di,dk] = sum_{dj=0}^{1} X[di,dj] * Y[dj,dk]
        
        Each term X[di,dj]*Y[dj,dk] where:
        X[di,dj] = sum_{(i,j,cx) in A_terms where (i//2, j//2) == (di,dj)} cx * A[i,j]
        Y[dj,dk] = sum_{(j',k,cy) in B_terms where (j'//2, k//2) == (dj,dk)} cy * B[j',k]
        
        Product = sum cx*cy*A[i,j]*B[j',k] where j==j'
        """
        nonlocal col
        for di in range(2):
            for dk in range(2):
                for dj in range(2):
                    # Get terms for X[di,dj]
                    x_terms = [(i, j, cx) for (i, j, cx) in A_terms if i // 2 == di and j // 2 == dj]
                    # Get terms for Y[dj,dk]
                    y_terms = [(j2, k, cy) for (j2, k, cy) in B_terms if j2 // 2 == dj and k // 2 == dk]
                    
                    for (i, j, cx) in x_terms:
                        for (j2, k, cy) in y_terms:
                            if j == j2:
                                U[i * m + j, col] += cx
                                V[j * p + k, col] += cy
                                W[(C_col_off + dk) * n + (C_row_off + di), col] += c_coeff
        col += 1
    
    # M1 = (A11+A22)*(B11+B22)
    A_M1 = [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1),
            (2, 2, 1), (2, 3, 1), (3, 2, 1), (3, 3, 1)]
    B_M1 = [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1),
            (2, 2, 1), (2, 3, 1), (3, 2, 1), (3, 3, 1)]
    add_2x2_product(A_M1, B_M1, 0, 0, 1)   # C11
    add_2x2_product(A_M1, B_M1, 2, 2, 1)   # C22
    
    # M2 = (A21+A22)*B11
    A_M2 = [(2, 0, 1), (2, 1, 1), (3, 0, 1), (3, 1, 1),
            (2, 2, 1), (2, 3, 1), (3, 2, 1), (3, 3, 1)]
    B_M2 = [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1)]
    add_2x2_product(A_M2, B_M2, 2, 0, 1)   # C21
    add_2x2_product(A_M2, B_M2, 2, 2, -1)  # C22
    
    # M3 = A11*(B12-B22)
    A_M3 = [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1)]
    B_M3 = [(0, 2, 1), (0, 3, 1), (1, 2, 1), (1, 3, 1),
            (2, 2, -1), (2, 3, -1), (3, 2, -1), (3, 3, -1)]
    add_2x2_product(A_M3, B_M3, 0, 2, 1)   # C12
    add_2x2_product(A_M3, B_M3, 2, 2, 1)   # C22
    
    # M4 = A22*(B21-B11)
    A_M4 = [(2, 2, 1), (2, 3, 1), (3, 2, 1), (3, 3, 1)]
    B_M4 = [(2, 0, 1), (2, 1, 1), (3, 0, 1), (3, 1, 1),
            (0, 0, -1), (0, 1, -1), (1, 0, -1), (1, 1, -1)]
    add_2x2_product(A_M4, B_M4, 0, 0, 1)   # C11
    add_2x2_product(A_M4, B_M4, 2, 0, 1)   # C21
    
    # M5 = (A11+A12)*B22
    A_M5 = [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1),
            (0, 2, 1), (0, 3, 1), (1, 2, 1), (1, 3, 1)]
    B_M5 = [(2, 2, 1), (2, 3, 1), (3, 2, 1), (3, 3, 1)]
    add_2x2_product(A_M5, B_M5, 0, 0, -1)  # C11
    add_2x2_product(A_M5, B_M5, 0, 2, 1)   # C12
    
    # M6 = (A21-A11)*(B11+B12)
    A_M6 = [(2, 0, 1), (2, 1, 1), (3, 0, 1), (3, 1, 1),
            (0, 0, -1), (0, 1, -1), (1, 0, -1), (1, 1, -1)]
    B_M6 = [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1),
            (0, 2, 1), (0, 3, 1), (1, 2, 1), (1, 3, 1)]
    add_2x2_product(A_M6, B_M6, 2, 2, 1)   # C22
    
    # M7 = (A12-A22)*(B21+B22)
    A_M7 = [(0, 2, 1), (0, 3, 1), (1, 2, 1), (1, 3, 1),
            (2, 2, -1), (2, 3, -1), (3, 2, -1), (3, 3, -1)]
    B_M7 = [(2, 0, 1), (2, 1, 1), (3, 0, 1), (3, 1, 1),
            (2, 2, 1), (2, 3, 1), (3, 2, 1), (3, 3, 1)]
    add_2x2_product(A_M7, B_M7, 0, 0, 1)   # C11
    
    # Part 2: Schoolbook for 4x4 x 4x3 (columns 4-6 of B)
    for i in range(n):
        for j in range(m):
            for k in range(3):  # k = 0,1,2 corresponds to B column 4,5,6
                U[i * m + j, col] = 1
                V[j * p + (k + 4), col] = 1
                W[(k + 4) * n + i, col] = 1
                col += 1
    
    # Verify col == rank
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    # Verify the decomposition with random test
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_decomp = np.zeros((n, p))
    for r in range(rank):
        for i in range(n):
            for k in range(p):
                val = 0
                for j in range(m):
                    val += A_test[i, j] * B_test[j, k] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                C_decomp[i, k] += val
    
    if not np.allclose(C_decomp, C_expected, atol=1e-10):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col_fb] = 1
                    V_fb[j * p + k, col_fb] = 1
                    W_fb[k * n + i, col_fb] = 1
                    col_fb += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
