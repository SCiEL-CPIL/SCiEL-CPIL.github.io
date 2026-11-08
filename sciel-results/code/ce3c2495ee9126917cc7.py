import numpy as np

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 68 solution using Strassen for 2x2 blocks
    # A is 3x3, B is 3x8
    # Split A:
    # A11 (2x2), A12 (2x1), A21 (1x2), A22 (1x1)
    # Split B:
    # B1 (2x8), B2 (1x8)
    # C11 = A11 B1 (2x8) -> 4 * Strassen(2x2) = 28
    # C12 = A12 B2 (2x8) -> 16
    # C21 = A21 B1 (1x8) -> 16
    # C22 = A22 B2 (1x8) -> 8
    # Total 68
    
    rank = 68
    U = [[0] * rank for _ in range(n * m)] # 9 x 68
    V = [[0] * rank for _ in range(m * p)] # 24 x 68
    W = [[0] * rank for _ in range(p * n)] # 24 x 68
    
    col = 0
    
    # Helper to add a Strassen block for 2x2 x 2
    # Inputs: indices for A 2x2, B 2x2
    # A rows: r1, r2; Cols: c1, c2
    # B rows: r1, r2; Cols: c1, c2
    # Output W indices: k*n+i
    def add_strassen(a_r1, a_r2, a_c1, a_c2, b_r1, b_r2, b_c1, b_c2, w_k1, w_k2, w_i1, w_i2):
        nonlocal col
        # Strassen 7 mults
        # M1 = (a11+a22)(b11+b22)
        U[a_r1*m + a_c1][col] = 1
        U[a_r2*m + a_c2][col] = 1
        V[b_r1*p + b_c1][col] = 1
        V[b_r2*p + b_c2][col] = 1
        # W contributions:
        # C11 += M1, C12 -= M1, C21 -= M1, C22 += M1
        W[w_k1*n + w_i1][col] = 1
        W[w_k2*n + w_i1][col] = -1
        W[w_k1*n + w_i2][col] = -1
        W[w_k2*n + w_i2][col] = 1
        col += 1
        
        # M2 = a22(b21+b22)
        U[a_r2*m + a_c2][col] = 1
        V[b_r2*p + b_c1][col] = 1
        V[b_r2*p + b_c2][col] = 1
        W[w_k2*n + w_i1][col] = 1
        W[w_k2*n + w_i2][col] = 1
        col += 1
        
        # M3 = (a11-a22)(b11+b12)
        U[a_r1*m + a_c1][col] = 1
        U[a_r2*m + a_c2][col] = -1
        V[b_r1*p + b_c1][col] = 1
        V[b_r1*p + b_c2][col] = 1
        W[w_k1*n + w_i1][col] = 1
        W[w_k1*n + w_i2][col] = 1
        col += 1
        
        # M4 = (a12+a22)(b11+b22)
        U[a_r1*m + a_c2][col] = 1
        U[a_r2*m + a_c2][col] = 1
        V[b_r1*p + b_c1][col] = 1
        V[b_r2*p + b_c2][col] = 1
        W[w_k1*n + w_i2][col] = 1
        W[w_k2*n + w_i1][col] = 1
        W[w_k2*n + w_i2][col] = -1
        col += 1
        
        # M5 = (a11+a12)(b21-b22)
        U[a_r1*m + a_c1][col] = 1
        U[a_r1*m + a_c2][col] = 1
        V[b_r2*p + b_c1][col] = 1
        V[b_r2*p + b_c2][col] = -1
        W[w_k2*n + w_i1][col] = 1
        W[w_k2*n + w_i2][col] = -1
        col += 1
        
        # M6 = (a21-a11)(b11+b12)
        U[a_r2*m + a_c1][col] = 1
        U[a_r1*m + a_c1][col] = -1
        V[b_r1*p + b_c1][col] = 1
        V[b_r1*p + b_c2][col] = 1
        W[w_k1*n + w_i1][col] = -1
        W[w_k2*n + w_i1][col] = 1
        W[w_k2*n + w_i2][col] = 1
        col += 1
        
        # M7 = (a11-a21-a22)(b21+b22)
        U[a_r1*m + a_c1][col] = 1
        U[a_r2*m + a_c1][col] = -1
        U[a_r2*m + a_c2][col] = -1
        V[b_r2*p + b_c1][col] = 1
        V[b_r2*p + b_c2][col] = 1
        W[w_k1*n + w_i1][col] = -1
        W[w_k1*n + w_i2][col] = -1
        W[w_k2*n + w_i2][col] = 1
        col += 1

    # 1. C11 = A11 B1 (2x8)
    # A11 rows 0,1 cols 0,1
    # B1 rows 0,1 cols 0..7
    # Split B1 into 4 2x2 blocks: B1_00 (cols 0,1), B1_01 (cols 2,3), B1_02 (cols 4,5), B1_03 (cols 6,7)
    for k in range(4):
        b_c1 = k*2
        b_c2 = k*2 + 1
        # W indices for C11:
        # C11 is rows 0,1 of C. Cols 0..7.
        # W index is k*n + i.
        # For C11 block k, the output columns are b_c1, b_c2.
        # But W is indexed by k (output row) and i (input row)?
        # W[k*n + i, r] corresponds to C[i, k]?
        # The convention is W[k*n + i, r].
        # C is 3x8.
        # C[i, k] = sum U[i*m+j] V[j*p+l] W[l*n+i]?
        # Standard: C[i, k] = sum_{j,l} A[i,j] B[j,l] ...
        # Here W is (p*n, rank). p=3, n=8? No.
        # Task: U (n*m, rank), V (m*p, rank), W (p*n, rank).
        # n=3, m=3, p=8.
        # U (9, r), V (24, r), W (24, r).
        # W index is p*n = 24.
        # Index in W is k*n + i?
        # If W is (p*n), then index is likely `k*n + i` where k is in 0..p-1 (8) and i is in 0..n-1 (3).
        # So W[k*n + i] corresponds to C[i, k].
        
        # For C11 (rows 0,1), the i indices are 0,1.
        # The k indices are b_c1, b_c2 (which are 0..7).
        
        add_strassen(
            a_r1=0, a_r2=1, a_c1=0, a_c2=1,
            b_r1=0, b_r2=1, b_c1=b_c1, b_c2=b_c2,
            w_k1=b_c1, w_k2=b_c2, w_i1=0, w_i2=1
        )

    # 2. C12 = A12 B2 (2x8)
    # A12 is rows 0,1 col 2. (2x1)
    # B2 is row 2 cols 0..7. (1x8)
    # C12[i, k] = A12[i, 0] * B2[0, k]
    # This is 2 * 8 = 16 terms.
    for i in range(2):
        for k in range(8):
            U[i*m + 2][col] = 1
            V[2*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1

    # 3. C21 = A21 B1 (1x8)
    # A21 is row 2 cols 0,1. (1x2)
    # B1 is rows 0,1 cols 0..7. (2x8)
    # C21[0, k] = A21[0,0]*B1[0,k] + A21[0,1]*B1[1,k]
    # This is 1 * 8 * 2 = 16 terms.
    for k in range(8):
        # Term 1: A21[0,0] * B1[0,k]
        U[2*m + 0][col] = 1
        V[0*p + k][col] = 1
        W[k*n + 2][col] = 1
        col += 1
        
        # Term 2: A21[0,1] * B1[1,k]
        U[2*m + 1][col] = 1
        V[1*p + k][col] = 1
        W[k*n + 2][col] = 1
        col += 1

    # 4. C22 = A22 B2 (1x8)
    # A22 is row 2 col 2. (1x1)
    # B2 is row 2 cols 0..7. (1x8)
    # C22[0, k] = A22[0,0] * B2[0,k]
    # This is 1 * 8 = 8 terms.
    for k in range(8):
        U[2*m + 2][col] = 1
        V[2*p + k][col] = 1
        W[k*n + 2][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    print(construct())
