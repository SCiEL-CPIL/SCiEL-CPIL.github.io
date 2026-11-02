import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal tensor decomposition for 4x4 * 4x7 matrix multiplication.
    
    Strategy: Exploit the 2x2 block structure with Strassen for 2x2*2x2 blocks,
    then use a Kronecker-product-like construction for the extra 3 columns.
    
    The key insight is that for 4x4 * 4x7, we can view B as [B_left(4x4), B_right(4x3)].
    For the 4x4 * 4x4 part, we can use Strassen's 7-multiplication algorithm on 2x2 blocks,
    giving rank 7^2 = 49 for that part (instead of 16 schoolbook).
    For the 4x4 * 4x3 part, we use schoolbook (12 multiplications per 2x2*2x3 block, 
    4 blocks = 48, but we can do better).
    
    Actually, let me think more carefully. The Kronecker product approach:
    If we have a rank-r decomposition for 2x2*2x2 (Strassen, r=7), then for 4x4*4x4
    we can get rank 7*7=49 by treating 4x4 as 2x2 blocks of 2x2.
    
    For 4x4 * 4x7: Split B into B1 (4x4) and B2 (4x3).
    C = A*B1 + A*B2
    
    For A*B1: Use the 49-rank decomposition from Kronecker product of Strassen.
    For A*B2: Split into 2x2 * 2x3 blocks. There are 4 such blocks.
    Each 2x2*2x3 needs at least 6 multiplications (schoolbook).
    So 4*6 = 24 for the A*B2 part.
    
    Total: 49 + 24 = 73.
    
    But wait - the Kronecker product of Strassen gives rank 49 for 4x4*4x4.
    Let me verify this is correct and implement it properly.
    
    Actually, the Kronecker product of two rank-7 decompositions gives a rank-49
    decomposition for 4x4*4x4. This is a standard result.
    
    For the 4x4 * 4x3 part, we can use schoolbook: each of the 4 (2x2 * 2x3) blocks
    needs 6 multiplications, so 4*6 = 24.
    
    Total rank: 49 + 24 = 73.
    
    This is much better than the current 52... wait, 52 < 73. Let me re-examine.
    
    Actually, the current program achieves rank 52 by using Strassen for 2x2*2x2 blocks
    (4 blocks * 7 = 28) and schoolbook for 2x2*2x3 blocks (2 blocks * 12 = 24).
    Total = 52.
    
    The Kronecker product approach gives 49 + 24 = 73, which is worse.
    
    Let me think of a different approach. What if we use the "subspace method" or
    find a lower-rank decomposition for the 2x2 * 2x3 case?
    
    Actually, the minimum rank for 2x2 * 2x3 is 6 (it's a bilinear map from 4+6=10
    variables to 6 outputs, and the tensor is non-degenerate). Schoolbook achieves 6.
    
    Wait, no. Schoolbook for 2x2 * 2x3 uses 12 multiplications (each output entry
    requires 2 multiplications, 6 outputs = 12). But can we do better?
    
    The tensor for 2x2 * 2x3 has rank at least 6 (since there are 6 output entries
    and the tensor is non-degenerate). But 6 might not be achievable with the
    constraint that we need to compute all 6 outputs simultaneously.
    
    Actually, for matrix multiplication m*n*p, the minimum number of multiplications
    is the tensor rank. For 2x2*2x3, the tensor rank is known to be 6 (it's a
    "slice rank" type problem). But achieving rank 6 requires a specific decomposition.
    
    Let me just use the current approach (rank 52) which is already better than
    schoolbook (112), and see if I can improve it further.
    
    Actually, let me reconsider. The current program has a bug - it's using
    schoolbook_2x2x3 which returns rank 12, not 6. Let me fix this and also
    try to find a rank-6 decomposition for 2x2*2x3.
    
    For 2x2 * 2x3, the tensor is:
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    This is actually two independent 2x2*2x1 problems (one for A[0,:] and one for A[1,:]).
    Each 2x2*2x1 has rank 2 (schoolbook). So total rank is 2+2 = 4? No, that's not right.
    
    Actually, 2x2 * 2x3 can be seen as: A (2x2) times B (2x3). The result is 2x3.
    The tensor rank of 2x2*2x3 is 6 (this is a known result).
    
    Let me just implement the current approach correctly and verify it works.
    """
    n, m, p = 4, 4, 7
    deadline = time.time() + 110  # Leave 10s safety margin
    
    def strassen_2x2x2():
        """Return U, V, W matrices for 2x2 * 2x2 using Strassen (rank 7)."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Strassen formulas for [[a,b],[c,d]] * [[e,f],[g,h]]:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        #
        # C00 = M1 - M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 - M6
        
        # A indices: a=0, b=1, c=2, d=3
        # B indices: e=0, f=1, g=2, h=3
        # C indices: C00=0, C01=1, C10=2, C11=3
        
        # M1 = (a+d)(e+h)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (c+d)e
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3 = a(f-h)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = d(g-e)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = -1; W[2, 3] = 1
        
        # M5 = (a+b)h
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (c-a)(e+f)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = -1
        
        # M7 = (b-d)(g+h)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W
    
    def schoolbook_2x2x3():
        """Return U, V, W matrices for 2x2 * 2x3 using schoolbook (rank 12)."""
        U = np.zeros((4, 12))
        V = np.zeros((6, 12))
        W = np.zeros((6, 12))
        
        col = 0
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        U[0, col] = 1; V[0, col] = 1; W[0, col] = 1; col += 1
        U[1, col] = 1; V[3, col] = 1; W[0, col] = 1; col += 1
        
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        U[0, col] = 1; V[1, col] = 1; W[1, col] = 1; col += 1
        U[1, col] = 1; V[4, col] = 1; W[1, col] = 1; col += 1
        
        # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
        U[0, col] = 1; V[2, col] = 1; W[2, col] = 1; col += 1
        U[1, col] = 1; V[5, col] = 1; W[2, col] = 1; col += 1
        
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        U[2, col] = 1; V[0, col] = 1; W[3, col] = 1; col += 1
        U[3, col] = 1; V[3, col] = 1; W[3, col] = 1; col += 1
        
        # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
        U[2, col] = 1; V[1, col] = 1; W[4, col] = 1; col += 1
        U[3, col] = 1; V[4, col] = 1; W[4, col] = 1; col += 1
        
        # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
        U[2, col] = 1; V[2, col] = 1; W[5, col] = 1; col += 1
        U[3, col] = 1; V[5, col] = 1; W[5, col] = 1; col += 1
        
        return U, V, W
    
    # Build the full decomposition using 2x2 blocks
    total_rank = 4 * 7 + 2 * 12  # 28 + 24 = 52
    U_full = np.zeros((n * m, total_rank))  # 16 x 52
    V_full = np.zeros((m * p, total_rank))  # 28 x 52
    W_full = np.zeros((p * n, total_rank))  # 28 x 52
    
    col_offset = 0
    
    # Block (0,0): A00 * B00 + A01 * B10 -> C00 (2x2)
    U_str, V_str, W_str = strassen_2x2x2()
    a_map = [0, 1, 4, 5]
    b_map = [0, 1, 7, 8]
    c_map = [0, 4, 1, 5]
    
    for c in range(7):
        for i in range(4):
            U_full[a_map[i], col_offset + c] += U_str[i, c]
        for j in range(4):
            V_full[b_map[j], col_offset + c] += V_str[j, c]
        for k in range(4):
            W_full[c_map[k], col_offset + c] += W_str[k, c]
    col_offset += 7
    
    # Block (0,1): A00 * B01 + A01 * B11 -> C01 (2x2)
    U_str, V_str, W_str = strassen_2x2x2()
    a_map = [0, 1, 4, 5]
    b_map = [2, 3, 9, 10]
    c_map = [8, 12, 9, 13]
    
    for c in range(7):
        for i in range(4):
            U_full[a_map[i], col_offset + c] += U_str[i, c]
        for j in range(4):
            V_full[b_map[j], col_offset + c] += V_str[j, c]
        for k in range(4):
            W_full[c_map[k], col_offset + c] += W_str[k, c]
    col_offset += 7
    
    # Block (1,0): A10 * B00 + A11 * B10 -> C10 (2x2)
    U_str, V_str, W_str = strassen_2x2x2()
    a_map = [8, 9, 12, 13]
    b_map = [0, 1, 7, 8]
    c_map = [2, 6, 3, 7]
    
    for c in range(7):
        for i in range(4):
            U_full[a_map[i], col_offset + c] += U_str[i, c]
        for j in range(4):
            V_full[b_map[j], col_offset + c] += V_str[j, c]
        for k in range(4):
            W_full[c_map[k], col_offset + c] += W_str[k, c]
    col_offset += 7
    
    # Block (1,1): A10 * B01 + A11 * B11 -> C11 (2x2)
    U_str, V_str, W_str = strassen_2x2x2()
    a_map = [8, 9, 12, 13]
    b_map = [2, 3, 9, 10]
    c_map = [10, 14, 11, 15]
    
    for c in range(7):
        for i in range(4):
            U_full[a_map[i], col_offset + c] += U_str[i, c]
        for j in range(4):
            V_full[b_map[j], col_offset + c] += V_str[j, c]
        for k in range(4):
            W_full[c_map[k], col_offset + c] += W_str[k, c]
    col_offset += 7
    
    # Block (0,2): A00 * B02 + A01 * B12 -> C02 (2x3)
    U_sb, V_sb, W_sb = schoolbook_2x2x3()
    a_map = [0, 1, 4, 5]
    b_map = [4, 5, 6, 11, 12, 13]
    c_map = [16, 20, 24, 17, 21, 25]
    
    for c in range(12):
        for i in range(4):
            U_full[a_map[i], col_offset + c] += U_sb[i, c]
        for j in range(6):
            V_full[b_map[j], col_offset + c] += V_sb[j, c]
        for k in range(6):
            W_full[c_map[k], col_offset + c] += W_sb[k, c]
    col_offset += 12
    
    # Block (1,2): A10 * B02 + A11 * B12 -> C12 (2x3)
    U_sb, V_sb, W_sb = schoolbook_2x2x3()
    a_map = [8, 9, 12, 13]
    b_map = [4, 5, 6, 11, 12, 13]
    c_map = [18, 22, 26, 19, 23, 27]
    
    for c in range(12):
        for i in range(4):
            U_full[a_map[i], col_offset + c] += U_sb[i, c]
        for j in range(6):
            V_full[b_map[j], col_offset + c] += V_sb[j, c]
        for k in range(6):
            W_full[c_map[k], col_offset + c] += W_sb[k, c]
    col_offset += 12
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    ui = U[i * m + j]
                    vj = V[j * p + k]
                    wk = W[k * n + i]
                    for c in range(rank):
                        val += ui[c] * vj[c] * wk[c]
                    if abs(val - 1) > 1e-10:
                        return False
        return True
    
    if not verify(U_full, V_full, W_full, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
# EVOLVE-BLOCK-END
