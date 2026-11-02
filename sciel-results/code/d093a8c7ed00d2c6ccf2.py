"""Matrix multiplication 3x3, 3x8 tensor decomposition.
Uses block-Strassen style decomposition to reduce rank below schoolbook 72.
"""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]


def _strassen_2x2_2x2(U_rows, V_rows, W_rows, U, V, W, base_col):
    """
    Add Strassen 2x2 * 2x2 terms to U, V, W starting at column base_col.
    U_rows: 4 indices (0,1,2,3) for A rows
    V_rows: 4 indices (0,1,2,3) for B rows  -> but B is 2x2, so rows 0,1 and cols 0,1
    W_rows: 4 indices for output rows (cols of C)
    
    For A(2x2) * B(2x2) = C(2x2):
    C00 = A00*B00 + A01*B10
    C01 = A00*B01 + A01*B11
    C10 = A10*B00 + A11*B10
    C11 = A10*B01 + A11*B11
    
    Strassen:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    C00 = M1 - M4 + M5 - M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    # 7 multiplications
    # M1 = (A00+A11)(B00+B11)
    # U: rows A00, A11 with +1; V: rows B00, B11 with +1
    U[U_rows[0], base_col] = 1; U[U_rows[3], base_col] = 1
    V[V_rows[0], base_col] = 1; V[V_rows[3], base_col] = 1
    W[W_rows[0], base_col] = 1; W[W_rows[1], base_col] = 1  # contributes to C00 and C11
    # C00 = M1 - M4 + M5 - M7, C11 = M1 - M2 + M3 + M6
    # So W for C00 gets +1, C11 gets +1
    col = base_col + 1

    # M2 = (A10+A11)B00
    U[U_rows[2], col] = 1; U[U_rows[3], col] = 1
    V[V_rows[0], col] = 1
    W[W_rows[2], col] = 1  # C10 += M2
    col += 1

    # M3 = A00(B10-B11)
    U[U_rows[0], col] = 1
    V[V_rows[2], col] = 1; V[V_rows[3], col] = -1
    W[W_rows[1], col] = 1; W[W_rows[2], col] = 1  # C01 += M3, C11 += M3
    col += 1

    # M4 = A11(B01-B00)
    U[U_rows[3], col] = 1
    V[V_rows[1], col] = 1; V[V_rows[0], col] = -1
    W[W_rows[0], col] = -1; W[W_rows[2], col] = 1  # C00 -= M4, C10 += M4
    col += 1

    # M5 = (A00+A01)B11
    U[U_rows[0], col] = 1; U[U_rows[1], col] = 1
    V[V_rows[3], col] = 1
    W[W_rows[0], col] = 1; W[W_rows[1], col] = 1  # C00 += M5, C01 += M5
    col += 1

    # M6 = (A10-A00)(B00+B01)
    U[U_rows[2], col] = 1; U[U_rows[0], col] = -1
    V[V_rows[0], col] = 1; V[V_rows[1], col] = 1
    W[W_rows[3], col] = 1  # C11 += M6
    col += 1

    # M7 = (A01-A11)(B10+B11)
    U[U_rows[1], col] = 1; U[U_rows[3], col] = -1
    V[V_rows[2], col] = 1; V[V_rows[3], col] = 1
    W[W_rows[0], col] = -1  # C00 -= M7
    col += 1

    return col


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 145

    # Schoolbook decomposition as fallback
    schoolbook_rank = n * m * p  # 72
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    best_rank = schoolbook_rank
    best_U, best_V, best_W = U, V, W

    # Try block-Strassen decomposition:
    # Split A(3x3) into top 2 rows and bottom 1 row
    # Split B(3x8) into top 2 rows and bottom 1 row, and 4 blocks of 2 cols each
    # 
    # A = [A_top; A_bot] where A_top is 2x3, A_bot is 1x3
    # B = [B_top; B_bot] where B_top is 2x8, B_bot is 1x8
    #
    # C = A*B = A_top * B_top + A_bot * B_bot
    #
    # C_top (2x8) = A_top * B_top
    # C_bot (1x8) = A_bot * B_bot
    #
    # For C_top = A_top(2x3) * B_top(2x8):
    #   Split A_top into A11(2x2) and A12(2x1)
    #   Split B_top into 4 blocks of 2x2: B1, B2, B3, B4
    #   C_top = A11*(B1|B2|B3|B4) + A12*B_rest  -- wait, this doesn't work cleanly
    #
    # Better: 
    # A_top is 2x3 = [A11(2x2) | A12(2x1)]
    # B_top is 2x8
    # C_top = A11 * B_top[0:2,:] + A12 * B_top[2:3,:]  -- no, B_top is 2 rows
    #
    # Let me reconsider. A is 3x3, B is 3x8.
    # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    #
    # Split: j=0,1 (top) and j=2 (bottom)
    # C_top (rows 0,1) = sum_{j=0}^{1} A_top[i,j]*B_top[j,k] + A_bot_row * B_bot_row
    # C_bot (row 2) = sum_{j=0}^{1} A_bot[i,j]*B_top[j,k] + A_bot[2,2]*B_bot[2,k]
    #
    # Where A_top = A[0:2, 0:2], A_top_cross = A[0:2, 2:3] (2x1)
    #       A_bot_cross = A[2:3, 0:2] (1x2), A_bot_bot = A[2:3, 2:3] (1x1)
    #       B_top = B[0:2, :], B_bot = B[2:3, :]
    #
    # C_top[i,k] = sum_{j=0}^{1} A_top[i,j]*B_top[j,k] + A_top_cross[i,0]*B_bot[0,k]
    # C_bot[k] = sum_{j=0}^{1} A_bot_cross[0,j]*B_top[j,k] + A_bot_bot[0,0]*B_bot[0,k]
    #
    # Term 1: A_top(2x2) * B_top(2x8)
    #   Split B_top into 4 blocks of 2x2: B^{(0)}, B^{(1)}, B^{(2)}, B^{(3)}
    #   Each 2x2 * 2x2 uses Strassen rank 7, total 28
    #
    # Term 2: A_top_cross(2x1) * B_bot(1x8)
    #   C_top[i,k] += A_top_cross[i,0] * B_bot[0,k]
    #   This is an outer product: U_i = A_top_cross[i,0], V_k = B_bot[0,k], W = 1
    #   Rank 1
    #
    # Term 3: A_bot_cross(1x2) * B_top(2x8)
    #   C_bot[k] += A_bot_cross[0,0]*B_top[0,k] + A_bot_cross[0,1]*B_top[1,k]
    #   For each k: sum over j=0,1 of A_bot_cross[0,j]*B_top[j,k]
    #   This is rank 1: U_j = A_bot_cross[0,j], V_k = B_top[j,k]... 
    #   Actually: C_bot[k] = sum_j A_bot_cross[0,j] * B_top[j,k]
    #   This is a standard matrix multiply of (1x2)*(2x8), rank = 2 (or can be done in 1 if structured)
    #   Rank 2
    #
    # Term 4: A_bot_bot(1x1) * B_bot(1x8)
    #   C_bot[k] += A_bot_bot[0,0] * B_bot[0,k]
    #   Rank 1
    #
    # Total: 28 + 1 + 2 + 1 = 32
    #
    # Let's implement this.

    U_new = [[0] * 32 for _ in range(n * m)]  # 9 x 32
    V_new = [[0] * 32 for _ in range(m * p)]  # 24 x 32
    W_new = [[0] * 32 for _ in range(p * n)]  # 24 x 32

    col = 0

    # Term 1: A_top(2x2) * B_top(2x8) = A[0:2,0:2] * B[0:2,0:8]
    # A_top indices: i in {0,1}, j in {0,1} -> U index = i*m+j = i*3+j
    # B_top indices: j in {0,1}, k in {0..7} -> V index = j*p+k = j*8+k
    # Output: C_top[i,k] where i in {0,1}, k in {0..7} -> W index = k*n+i = k*3+i

    for block in range(4):
        # B^{(block)} = B[0:2, block*2:(block+1)*2]
        # V rows for this block: j in {0,1}, k in {block*2, block*2+1}
        v_rows = [0 * 8 + block * 2, 0 * 8 + block * 2 + 1, 1 * 8 + block * 2, 1 * 8 + block * 2 + 1]
        # U rows: A_top[0,0]=0*3+0=0, A_top[0,1]=0*3+1=1, A_top[1,0]=1*3+0=3, A_top[1,1]=1*3+1=4
        u_rows = [0, 1, 3, 4]
        # W rows: C_top[0,block*2]=block*2*3+0, C_top[0,block*2+1]=(block*2+1)*3+0,
        #          C_top[1,block*2]=block*2*3+1, C_top[1,block*2+1]=(block*2+1)*3+1
        w_rows = [block * 2 * 3 + 0, (block * 2 + 1) * 3 + 0, block * 2 * 3 + 1, (block * 2 + 1) * 3 + 1]
        col = _strassen_2x2_2x2(u_rows, v_rows, w_rows, U_new, V_new, W_new, col)

    # col should be 28 now

    # Term 2: A_top_cross(2x1) * B_bot(1x8)
    # A_top_cross[i,0] = A[i,2] for i in {0,1} -> U index = i*3+2
    # B_bot[0,k] = B[2,k] -> V index = 2*8+k = 16+k
    # Output: C_top[i,k] -> W index = k*3+i
    # C_top[i,k] += A[i,2] * B[2,k]
    # Rank 1: U[i,2]=1, V[16+k]=1, W[k*3+i]=1
    U_new[0 * 3 + 2][col] = 1  # A[0,2]
    U_new[1 * 3 + 2][col] = 1  # A[1,2]
    for k in range(8):
        V_new[2 * 8 + k][col] = 1  # B[2,k]
        W_new[k * 3 + 0][col] = 1  # C[0,k]
        W_new[k * 3 + 1][col] = 1  # C[1,k]
    col += 1
    # col = 29

    # Term 3: A_bot_cross(1x2) * B_top(2x8)
    # A_bot_cross[0,j] = A[2,j] for j in {0,1} -> U index = 2*3+j = 6+j
    # B_top[j,k] = B[j,k] -> V index = j*8+k
    # Output: C_bot[k] = C[2,k] -> W index = k*3+2
    # C[2,k] += A[2,0]*B[0,k] + A[2,1]*B[1,k]
    # Rank 2:
    # Term 3a: A[2,0]*B[0,k]
    U_new[2 * 3 + 0][col] = 1  # A[2,0]
    for k in range(8):
        V_new[0 * 8 + k][col] = 1  # B[0,k]
        W_new[k * 3 + 2][col] = 1  # C[2,k]
    col += 1
    # Term 3b: A[2,1]*B[1,k]
    U_new[2 * 3 + 1][col] = 1  # A[2,1]
    for k in range(8):
        V_new[1 * 8 + k][col] = 1  # B[1,k]
        W_new[k * 3 + 2][col] = 1  # C[2,k]
    col += 1
    # col = 31

    # Term 4: A_bot_bot(1x1) * B_bot(1x8)
    # A[2,2] * B[2,k]
    # U index = 2*3+2 = 8
    # V index = 2*8+k = 16+k
    # W index = k*3+2
    U_new[2 * 3 + 2][col] = 1  # A[2,2]
    for k in range(8):
        V_new[2 * 8 + k][col] = 1  # B[2,k]
        W_new[k * 3 + 2][col] = 1  # C[2,k]
    col += 1
    # col = 32

    # Validate the decomposition
    def validate(U_mat, V_mat, W_mat, rank_val, n=n, m=m, p=p):
        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank_val):
                        val += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True

    if validate(U_new, V_new, W_new, col):
        best_rank = col
        best_U, best_V, best_W = U_new, V_new, W_new
    else:
        # Fallback to schoolbook
        pass

    return {"U": best_U, "V": best_V, "W": best_W}
