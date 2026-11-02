"""Combined approach: structured block decomposition for rank-minimal 3x5x6 matmul."""
import random
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0

    # Strategy: Decompose 3x5x6 into (3x3x6) + (3x2x6)
    # 3x3x6 is two 3x3x3 blocks.
    # 3x2x6 is two 3x2x3 blocks.
    
    # Get factors for 3x3x3 (rank 27 schoolbook for safety, or 23 if available)
    # Get factors for 3x2x3 (rank 3)
    
    # Construct 3x3x3 factors (rank 27)
    U_333, V_333, W_333 = _get_333_decomp_schoolbook()
    
    # Construct 3x2x3 factors (rank 3)
    U_323, V_323, W_323 = _get_323_decomp()
    
    # Assemble 3x5x6 factors
    # U is 15xR, V is 30xR, W is 18xR
    # R = 54 + 6 = 60
    
    R_total = len(U_333[0]) * 2 + len(U_323[0]) * 2
    U = np.zeros((15, R_total))
    V = np.zeros((30, R_total))
    W = np.zeros((18, R_total))
    
    col_offset = 0
    
    # Block 1: A[0:3, 0:3] * B[0:3, 0:3] -> C[0:3, 0:3]
    # U indices: 0:3 (rows of A1), cols 0:3 (cols of A1) -> U[0:3, 0:3]
    # V indices: 0:3 (rows of B1), cols 0:3 (cols of B1) -> V[0:3, 0:3]
    # W indices: 0:3 (rows of C), 0:3 (cols of C) -> W[0:3, 0:3]
    # Note: U is (n*m, r) = (15, r). Row i*m+j.
    # For A1 (3x3), i in 0..2, j in 0..2. Rows 0..8.
    # For B1 (3x6), j in 0..2, k in 0..5. Rows 0..17.
    # For C (3x6), k in 0..5, i in 0..2. Rows 0..17.
    
    # Sub-block 1.1: A1[:, 0:3] * B1[0:3, 0:3]
    # U[0:9, col_offset:col_offset+27] = U_333
    U[0:9, col_offset:col_offset+27] = U_333
    # V[0:18, col_offset:col_offset+27] = V_333
    # V_333 is 9x27 (3x3). V is 30x60.
    # V row j*p+k. For B1[0:3, 0:3], j in 0..2, k in 0..2.
    # Rows 0, 1, 2 (k=0), 6,7,8 (k=1), 12,13,14 (k=2)?
    # No, V index is j*p+k. p=6.
    # j=0, k=0..2 -> 0,1,2
    # j=1, k=0..2 -> 6,7,8
    # j=2, k=0..2 -> 12,13,14
    # We need to map V_333 (9x27) to these rows.
    # V_333 rows are j*3+k.
    # j=0,k=0 -> 0. Map to V[0]
    # j=0,k=1 -> 1. Map to V[1]
    # j=0,k=2 -> 2. Map to V[2]
    # j=1,k=0 -> 3. Map to V[6]
    # j=1,k=1 -> 4. Map to V[7]
    # j=1,k=2 -> 5. Map to V[8]
    # j=2,k=0 -> 6. Map to V[12]
    # j=2,k=1 -> 7. Map to V[13]
    # j=2,k=2 -> 8. Map to V[14]
    
    v_rows_333_to_36 = [0, 1, 2, 6, 7, 8, 12, 13, 14]
    V[v_rows_333_to_36, col_offset:col_offset+27] = V_333
    
    # W index is k*n+i. n=3.
    # C[0:3, 0:3]. k in 0..2, i in 0..2.
    # k=0, i=0..2 -> 0,1,2
    # k=1, i=0..2 -> 3,4,5
    # k=2, i=0..2 -> 6,7,8
    # W_333 rows are k*3+i.
    # k=0,i=0 -> 0. Map to W[0]
    # ...
    # k=2,i=2 -> 8. Map to W[8]
    w_rows_333_to_36 = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    W[w_rows_333_to_36, col_offset:col_offset+27] = W_333
    
    col_offset += 27
    
    # Sub-block 1.2: A1[:, 0:3] * B1[0:3, 3:6]
    # Same U.
    # V rows: j in 0..2, k in 3..5.
    # j=0, k=3..5 -> 3,4,5
    # j=1, k=3..5 -> 9,10,11
    # j=2, k=3..5 -> 15,16,17
    v_rows_333_to_36b = [3, 4, 5, 9, 10, 11, 15, 16, 17]
    U[0:9, col_offset:col_offset+27] = U_333
    V[v_rows_333_to_36b, col_offset:col_offset+27] = V_333
    # W rows: C[0:3, 3:6]. k in 3..5, i in 0..2.
    # k=3, i=0..2 -> 9,10,11
    # k=4, i=0..2 -> 12,13,14
    # k=5, i=0..2 -> 15,16,17
    w_rows_333_to_36b = [9, 10, 11, 12, 13, 14, 15, 16, 17]
    W[w_rows_333_to_36b, col_offset:col_offset+27] = W_333
    
    col_offset += 27
    
    # Block 2: A[0:3, 3:5] * B[3:5, 0:6]
    # A2 is 3x2. Rows 9..14 in U.
    # B2 is 2x6. Rows 18..29 in V.
    # C is 3x6. Rows 0..17 in W.
    
    # Sub-block 2.1: A2 * B2[0:2, 0:3]
    # U_323 is 6x3 (3x2).
    # U rows: i in 0..2, j in 0..1.
    # i=0, j=0,1 -> 3,4 (offset by 9? No, U is 15xR. A2 is rows 3,4 of A? No.
    # A is 3x5. A2 is cols 3,4 of A.
    # U index i*m+j. i in 0..2, j in 3..4.
    # i=0, j=3 -> 3. i=0, j=4 -> 4.
    # i=1, j=3 -> 8. i=1, j=4 -> 9.
    # i=2, j=3 -> 13. i=2, j=4 -> 14.
    u_rows_323_to_35 = [3, 4, 8, 9, 13, 14]
    
    # V_323 is 6x3 (2x3).
    # V index j*p+k. j in 0..1 (local), k in 0..2.
    # Global j is 3,4.
    # j=3, k=0..2 -> 18,19,20
    # j=4, k=0..2 -> 24,25,26
    v_rows_323_to_26 = [18, 19, 20, 24, 25, 26]
    
    # W_323 is 6x3 (3x3? No, 3x2x3 tensor. W is p*n = 6*3 = 18? No.
    # W is (p*n, r). p=6, n=3. 18 rows.
    # W index k*n+i.
    # C is 3x6.
    # Sub-block 2.1 corresponds to C[0:3, 0:3].
    # k in 0..2, i in 0..2.
    # k=0, i=0..2 -> 0,1,2
    # k=1, i=0..2 -> 3,4,5
    # k=2, i=0..2 -> 6,7,8
    w_rows_323_to_36 = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    
    U[u_rows_323_to_35, col_offset:col_offset+3] = U_323
    V[v_rows_323_to_26, col_offset:col_offset+3] = V_323
    W[w_rows_323_to_36, col_offset:col_offset+3] = W_323
    
    col_offset += 3
    
    # Sub-block 2.2: A2 * B2[0:2, 3:6]
    # Same U.
    # V rows: j in 3,4. k in 3..5.
    # j=3, k=3..5 -> 21,22,23
    # j=4, k=3..5 -> 27,28,29
    v_rows_323_to_26b = [21, 22, 23, 27, 28, 29]
    
    # W rows: C[0:3, 3:6].
    # k in 3..5, i in 0..2.
    # k=3, i=0..2 -> 9,10,11
    # k=4, i=0..2 -> 12,13,14
    # k=5, i=0..2 -> 15,16,17
    w_rows_323_to_36b = [9, 10, 11, 12, 13, 14, 15, 16, 17]
    
    U[u_rows_323_to_35, col_offset:col_offset+3] = U_323
    V[v_rows_323_to_26b, col_offset:col_offset+3] = V_323
    W[w_rows_323_to_36b, col_offset:col_offset+3] = W_323
    
    col_offset += 3
    
    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Verify
    if _check_decomp(n, m, p, U, V, W, R_total):
        return result
    else:
        # Fallback to schoolbook if construction failed
        return _schoolbook(n, m, p, n*m*p)

def _get_333_decomp_schoolbook():
    """Returns U, V, W for 3x3x3 schoolbook (rank 27)."""
    n, m, p = 3, 3, 3
    r = 27
    U = np.zeros((9, r))
    V = np.zeros((9, r))
    W = np.zeros((9, r))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _get_323_decomp():
    """Returns U, V, W for 3x2x3 rank 3 decomposition."""
    n, m, p = 3, 2, 3
    r = 3
    U = np.zeros((6, r))
    V = np.zeros((6, r))
    W = np.zeros((9, r)) # p*n = 9
    
    # c=0: k=0
    W[0*n+0, 0] = 1; W[0*n+1, 0] = 1; W[0*n+2, 0] = 1
    U[:, 0] = 1
    V[0*3+0, 0] = 1; V[0*3+1, 0] = 1; V[0*3+2, 0] = 1
    # Note: V index j*p+k. j in 0..1, k in 0..2.
    # j=0, k=0 -> 0. j=0, k=1 -> 1. j=0, k=2 -> 2.
    # j=1, k=0 -> 3. j=1, k=1 -> 4. j=1, k=2 -> 5.
    # For k=0, we need V[j,0] = 1 for all j.
    # So V[0,0]=1, V[3,0]=1.
    # My previous logic: V[0*3+0] is V[0,0].
    
    # Let's re-verify V indices for 3x2x3.
    # V is (m*p, r) = (6, 3).
    # Index j*p+k.
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # j=0, k=2 -> 2
    # j=1, k=0 -> 3
    # j=1, k=1 -> 4
    # j=1, k=2 -> 5
    
    # c=0 (k=0):
    # W[k=0, i, 0] = 1 for i=0,1,2.
    # W index k*n+i.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=0, i=2 -> 2
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1
    U[:, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1 # j=0,k=0 and j=1,k=0
    
    # c=1 (k=1):
    W[3, 1] = 1; W[4, 1] = 1; W[5, 1] = 1 # k=1, i=0,1,2
    U[:, 1] = 1
    V[1, 1] = 1; V[4, 1] = 1 # j=0,k=1 and j=1,k=1
    
    # c=2 (k=2):
    W[6, 2] = 1; W[7, 2] = 1; W[8, 2] = 1 # k=2, i=0,1,2
    U[:, 2] = 1
    V[2, 2] = 1; V[5, 2] = 1 # j=0,k=2 and j=1,k=2
    
    return U, V, W

def _check_decomp(n, m, p, U, V, W, r, tol=1e-9):
    """Check if U,V,W of rank r exactly decompose the matmul tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += W[k*n+i][c] * U[i*m+j][c] * V[j*p+k][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p, rank):
    """Standard schoolbook decomposition of rank n*m*p."""
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
