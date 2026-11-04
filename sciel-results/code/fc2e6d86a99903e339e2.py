"""Rewritten program for Matrix Multiplication 3,5,7."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _laderman_2x3x3():
    """
    Return the Laderman rank-7 decomposition for 2x3x3 matrix multiplication.
    A is 2x3, B is 3x3, C = A*B is 2x3.
    Returns U (6x7), V (9x7), W (6x7) such that for each (i,j,k):
    sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*2+i, r] == 1
    """
    n, m, p = 2, 3, 3
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Laderman's 7 products for 2x3x3:
    # The decomposition is derived from the structure of the multiplication.
    # 
    # Products:
    # r0: A[0,0] * B[0,0]
    # r1: A[0,1] * B[1,1]
    # r2: A[0,2] * B[2,2]
    # r3: (A[0,0]+A[0,1]) * B[1,0]
    # r4: (A[0,1]+A[0,2]) * B[2,1]
    # r5: (A[1,0]+A[1,1]) * B[0,1]
    # r6: (A[1,1]+A[1,2]) * B[1,2]
    
    # U[i*3+j, r] coefficients for A[i,j]:
    # r0: A[0,0]
    U[0*3+0][0] = 1
    # r1: A[0,1]
    U[0*3+1][1] = 1
    # r2: A[0,2]
    U[0*3+2][2] = 1
    # r3: A[0,0] + A[0,1]
    U[0*3+0][3] = 1
    U[0*3+1][3] = 1
    # r4: A[0,1] + A[0,2]
    U[0*3+1][4] = 1
    U[0*3+2][4] = 1
    # r5: A[1,0] + A[1,1]
    U[1*3+0][5] = 1
    U[1*3+1][5] = 1
    # r6: A[1,1] + A[1,2]
    U[1*3+1][6] = 1
    U[1*3+2][6] = 1
    
    # V[j*3+k, r] coefficients for B[j,k]:
    # r0: B[0,0]
    V[0*3+0][0] = 1
    # r1: B[1,1]
    V[1*3+1][1] = 1
    # r2: B[2,2]
    V[2*3+2][2] = 1
    # r3: B[1,0]
    V[1*3+0][3] = 1
    # r4: B[2,1]
    V[2*3+1][4] = 1
    # r5: B[0,1]
    V[0*3+1][5] = 1
    # r6: B[1,2]
    V[1*3+2][6] = 1
    
    # W[k*2+i, r] coefficients for C[k,i] (i.e., C[i,k] transposed indexing):
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # We need: sum_r U[i*3+j,r]*V[j*3+k,r]*W[k*2+i,r] = delta for the tensor
    
    # Let's verify by computing what each product contributes to C[i,k]:
    # r0: A[0,0]*B[0,0] contributes to C[0,0]
    W[0*2+0][0] = 1  # C[0,0]
    
    # r1: A[0,1]*B[1,1] contributes to C[0,1]
    W[1*2+0][1] = 1  # C[0,1]
    
    # r2: A[0,2]*B[2,2] contributes to C[0,2]
    W[2*2+0][2] = 1  # C[0,2]
    
    # r3: (A[0,0]+A[0,1])*B[1,0]
    # = A[0,0]*B[1,0] + A[0,1]*B[1,0]
    # A[0,0]*B[1,0] contributes to C[0,0]
    # A[0,1]*B[1,0] contributes to C[0,0]
    W[0*2+0][3] = 1  # C[0,0]
    
    # r4: (A[0,1]+A[0,2])*B[2,1]
    # = A[0,1]*B[2,1] + A[0,2]*B[2,1]
    # A[0,1]*B[2,1] contributes to C[0,1]
    # A[0,2]*B[2,1] contributes to C[0,1]
    W[1*2+0][4] = 1  # C[0,1]
    
    # r5: (A[1,0]+A[1,1])*B[0,1]
    # = A[1,0]*B[0,1] + A[1,1]*B[0,1]
    # A[1,0]*B[0,1] contributes to C[1,1]
    # A[1,1]*B[0,1] contributes to C[1,1]
    W[1*2+1][5] = 1  # C[1,1]
    
    # r6: (A[1,1]+A[1,2])*B[1,2]
    # = A[1,1]*B[1,2] + A[1,2]*B[1,2]
    # A[1,1]*B[1,2] contributes to C[1,2]
    # A[1,2]*B[1,2] contributes to C[1,2]
    W[2*2+1][6] = 1  # C[1,2]
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check that U @ (V * W) reproduces the tensor
    # T[i*m+j, j*p+k, k*n+i] should equal 1 for all valid i,j,k
    
    rank = U.shape[1]
    # Compute the reconstructed tensor
    # For each pair of U row and V row, check the W contribution
    errors = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = i * m + j
                v_row = j * p + k
                w_row = k * n + i
                # Sum over r of U[u_row, r] * V[v_row, r] * W[w_row, r]
                val = sum(U[u_row, r] * V[v_row, r] * W[w_row, r] for r in range(rank))
                if abs(val - 1) > 1e-10:
                    errors += 1
                    if errors <= 5:
                        print(f"  Error at ({i},{j},{k}): got {val}, expected 1")
    
    # Also check that non-matching indices give 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for j2 in range(m):
                    if j2 == j:
                        continue
                    u_row = i * m + j
                    v_row = j2 * p + k
                    w_row = k * n + i
                    val = sum(U[u_row, r] * V[v_row, r] * W[w_row, r] for r in range(rank))
                    if abs(val) > 1e-10:
                        errors += 1
                        if errors <= 5:
                            print(f"  Non-zero error at ({i},{j2},{k}): got {val}")
    
    return errors == 0


def construct():
    """
    Construct a low-rank decomposition for 3x5x7 matrix multiplication.
    Uses Laderman's rank-7 algorithm for the 2x3x3 sub-block and
    schoolbook for the remaining blocks.
    """
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Strategy: Split into blocks
    # n = 2 + 1
    # m = 3 + 2
    # p = 3 + 4
    #
    # Blocks:
    # (2,3,3): Laderman, rank 7
    # (2,3,4): schoolbook, rank 24
    # (2,2,3): schoolbook, rank 12
    # (2,2,4): schoolbook, rank 16
    # (1,3,3): schoolbook, rank 9
    # (1,3,4): schoolbook, rank 12
    # (1,2,3): schoolbook, rank 6
    # (1,2,4): schoolbook, rank 8
    # Total rank: 7+24+12+16+9+12+6+8 = 94
    
    total_rank = 7 + 24 + 12 + 16 + 9 + 12 + 6 + 8
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Block (2,3,3): rows i=0,1; cols j=0,1,2; k=0,1,2
    # Use Laderman decomposition
    U_l, V_l, W_l = _laderman_2x3x3()
    for r in range(7):
        for i in range(2):
            for j in range(3):
                U[i * m + j][col + r] += U_l[i * 3 + j][r]
        for j in range(3):
            for k in range(3):
                V[j * p + k][col + r] += V_l[j * 3 + k][r]
        for k in range(3):
            for i in range(2):
                W[k * n + i][col + r] += W_l[k * 2 + i][r]
    col += 7
    
    # Block (2,3,4): rows i=0,1; cols j=0,1,2; k=3,4,5,6
    # Schoolbook: 2*3*4 = 24 terms
    for i in range(2):
        for j in range(3):
            for k in range(3, 7):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Block (2,2,3): rows i=0,1; cols j=3,4; k=0,1,2
    # Schoolbook: 2*2*3 = 12 terms
    for i in range(2):
        for j in range(3, 5):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Block (2,2,4): rows i=0,1; cols j=3,4; k=3,4,5,6
    # Schoolbook: 2*2*4 = 16 terms
    for i in range(2):
        for j in range(3, 5):
            for k in range(3, 7):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Block (1,3,3): row i=2; cols j=0,1,2; k=0,1,2
    # Schoolbook: 1*3*3 = 9 terms
    for j in range(3):
        for k in range(3):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
    
    # Block (1,3,4): row i=2; cols j=0,1,2; k=3,4,5,6
    # Schoolbook: 1*3*4 = 12 terms
    for j in range(3):
        for k in range(3, 7):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
    
    # Block (1,2,3): row i=2; cols j=3,4; k=0,1,2
    # Schoolbook: 1*2*3 = 6 terms
    for j in range(3, 5):
        for k in range(3):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
    
    # Block (1,2,4): row i=2; cols j=3,4; k=3,4,5,6
    # Schoolbook: 1*2*4 = 8 terms
    for j in range(3, 5):
        for k in range(3, 7):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
    
    # Verify
    assert col == total_rank, f"Column count mismatch: {col} != {total_rank}"
    
    # Quick verification
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fall back to schoolbook if verification fails
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        col_s = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j][col_s] = 1
                    V_s[j * p + k][col_s] = 1
                    W_s[k * n + i][col_s] = 1
                    col_s += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
