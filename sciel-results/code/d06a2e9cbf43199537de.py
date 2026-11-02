"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """
    Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W each of shape (4, 7) with entries in {-1, 0, 1}.
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # The 7 Strassen products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U rows: (i,j) -> i*2+j
    # V rows: (j,k) -> j*2+k
    # W rows: (k,i) -> k*2+i
    
    # For each of 7 terms, define which (i,j), (j,k), (k,i) indices are involved
    # and the sign.
    
    # M1 = (A00+A11)(B00+B11)
    # A part: A00 (i=0,j=0) and A11 (i=1,j=1)
    # B part: B00 (j=0,k=0) and B11 (j=1,k=1)
    # C part: C00 (k=0,i=0) + C11 (k=1,i=1)
    # This is a product of sums, so it contributes to 4 terms in C, each being a product
    # But in tensor decomposition, M1 is a single rank-1 term:
    # M1 = (A00+A11) * (B00+B11)
    # This contributes to C00 with +1 and C11 with +1... wait no.
    # Actually Strassen's trick: C = M1 * (E00+E11) + ... 
    # Let me be more careful.
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # For rank-1 decomposition, each M_c is a product of a linear form in A and a linear form in B.
    # U[i*2+j, c] = coefficient of A[i,j] in the A-part of M_c
    # V[j*2+k, c] = coefficient of B[j,k] in the B-part of M_c
    # W[k*2+i, c] = coefficient of C[k,i]... wait, W[k*n+i, c] is the coefficient in C.
    
    # Actually the decomposition is:
    # C[k,i] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, that's not right either. The tensor is T[i,j,k] = delta which is 1 always for matmul.
    # C = A * B means C[k,i] = sum_j A[i,j] * B[j,k]
    
    # So the tensor T[i,j,k] = 1 for all i,j,k (it's the matmul tensor).
    # A decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # For Strassen on 2x2x2:
    # M1: A-part = A00+A11, B-part = B00+B11
    #   U[0*2+0, 0] = 1, U[1*2+1, 0] = 1
    #   V[0*2+0, 0] = 1, V[1*2+1, 0] = 1
    #   C-part: contributes to C00 and C11
    #   W[0*2+0, 0] = 1 (C00), W[1*2+1, 0] = 1 (C11)
    #   But wait, W has only one value per column. So W[k*2+i, 0] must be set.
    #   C00 gets +1 from M1, C11 gets +1 from M1.
    #   So W[0, 0] = 1 (k=0,i=0 -> C00), W[3, 0] = 1 (k=1,i=1 -> C11)
    
    # M2: A-part = A10+A11, B-part = B00
    #   U[1*2+0, 1] = 1, U[1*2+1, 1] = 1
    #   V[0*2+0, 1] = 1
    #   C-part: C10 gets +1, C11 gets -1
    #   W[0*2+1, 1] = 1 (C10), W[1*2+1, 1] = -1 (C11)
    
    # M3: A-part = A00, B-part = B01-B11
    #   U[0*2+0, 2] = 1
    #   V[0*2+1, 2] = 1, V[1*2+1, 2] = -1
    #   C-part: C01 gets +1, C11 gets +1
    #   W[1*2+0, 2] = 1 (C01), W[1*2+1, 2] = 1 (C11)
    
    # M4: A-part = A11, B-part = B10-B00
    #   U[1*2+1, 3] = 1
    #   V[1*2+0, 3] = 1, V[0*2+0, 3] = -1
    #   C-part: C00 gets +1, C10 gets +1
    #   W[0*2+0, 3] = 1 (C00), W[0*2+1, 3] = 1 (C10)
    
    # M5: A-part = A00+A01, B-part = B11
    #   U[0*2+0, 4] = 1, U[0*2+1, 4] = 1
    #   V[1*2+1, 4] = 1
    #   C-part: C00 gets -1, C01 gets +1
    #   W[0*2+0, 4] = -1 (C00), W[1*2+0, 4] = 1 (C01)
    
    # M6: A-part = A10-A00, B-part = B00+B01
    #   U[1*2+0, 5] = 1, U[0*2+0, 5] = -1
    #   V[0*2+0, 5] = 1, V[0*2+1, 5] = 1
    #   C-part: C11 gets +1
    #   W[1*2+1, 5] = 1 (C11)
    
    # M7: A-part = A01-A11, B-part = B10+B11
    #   U[0*2+1, 6] = 1, U[1*2+1, 6] = -1
    #   V[1*2+0, 6] = 1, V[1*2+1, 6] = 1
    #   C-part: C00 gets +1
    #   W[0*2+0, 6] = 1 (C00)
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_rank44():
    """
    Build a rank-44 decomposition using Strassen for 2x2x2 blocks.
    
    For 2x3x8: we can split into blocks.
    A is 2x3, B is 3x8.
    
    Split B into four 3x2 blocks: B[:, 0:2], B[:, 2:4], B[:, 4:6], B[:, 6:8]
    Each C[:, 2k:2k+2] = A * B[:, 2k:2k+2]
    
    For each block, A is 2x3, B_block is 3x2.
    We can split A into A[:, 0:2] (2x2) and A[:, 2:3] (2x1).
    
    C_block = A[:, 0:2] * B_block + A[:, 2:3] * B_block
    
    For A[:, 0:2] * B_block (2x2 * 3x2):
    Split B_block into B_block[0:1, :] and B_block[1:3, :]... hmm this is getting complex.
    
    Alternative: use Strassen for the 2x2 part of A and handle the 3rd column separately.
    
    Actually, let me think differently. The knowledge says:
    "Strassen rank-7 decomposition of each 2x2x2 sub-tensor plus 4 rank-1 terms for the third A-column reduces per-block rank from 12 to 11"
    
    So for each 2x3x2 block (A is 2x3, B is 3x2, C is 2x2):
    - Use Strassen (rank 7) for the 2x2x2 part (first 2 columns of A)
    - Add 4 rank-1 terms for the 3rd column of A
    - Total per block: 7 + 4 = 11
    
    With 4 such blocks (since B is 3x8 = 3x(4*2)): total rank = 4 * 11 = 44
    
    Let me implement this.
    """
    n, m, p = 2, 3, 8
    num_blocks = p // 2  # 4 blocks of width 2
    
    U = []
    V = []
    W = []
    
    for b in range(num_blocks):
        k_offset = b * 2  # start column in B
        
        # Get Strassen decomposition for 2x2x2
        sU, sV, sW = _strassen_2x2x2()
        
        # sU is 4x7, sV is 4x7, sW is 4x7
        # sU[i*2+j, c] for i,j in 0..1 (A indices)
        # sV[j*2+k, c] for j in 0..1, k in 0..1 (B indices, local)
        # sW[k*2+i, c] for k in 0..1, i in 0..1 (C indices, local)
        
        # Extend to 2x3x2:
        # U has 2*3=6 rows, V has 3*2=6 rows, W has 2*2=4 rows
        
        # For the Strassen part (first 2 columns of A, 2 columns of B):
        # U rows: i*3+j for i in 0..1, j in 0..1 (only first 2 cols of A)
        # V rows: j*2+k for j in 0..1, k in 0..1 (only first 2 rows of B_block)
        # W rows: k*2+i for k in 0..1, i in 0..1
        
        for c in range(7):
            u_row = [0] * 6
            v_row = [0] * 6
            w_row = [0] * 4
            
            # U: A has 3 columns, Strassen uses first 2
            for i in range(2):
                for j in range(2):
                    u_row[i * 3 + j] = sU[i * 2 + j][c]
            
            # V: B_block has 3 rows, Strassen uses first 2
            for j in range(2):
                for k in range(2):
                    v_row[j * 2 + k] = sV[j * 2 + k][c]
            
            # W: C is 2x2, all used
            for k in range(2):
                for i in range(2):
                    w_row[k * 2 + i] = sW[k * 2 + i][c]
            
            U.append(u_row)
            V.append(v_row)
            W.append(w_row)
        
        # Now add 4 rank-1 terms for the 3rd column of A (j=2)
        # A[i,2] * B[j, 2k:2k+2] contributes to C[i, :]
        # For each j in 0..2 and k in 0..1: A[i,2] * B[j, 2k+k_local]
        # That's 2 * 3 * 2 = 12 terms, but we can do better.
        
        # Actually, for the 3rd column of A:
        # C[i, 2k+k_local] += A[i, 2] * sum_j B[j, 2k+k_local]
        # Wait no: C[i, col] = sum_j A[i,j] * B[j, col]
        # The contribution from j=2: C[i, col] += A[i,2] * B[2, col]
        
        # So for each i in 0..1 and col in 0..1 (local):
        # C[i, 2k+col] += A[i,2] * B[2, 2k+col]
        
        # That's 2 * 2 = 4 rank-1 terms!
        for i in range(2):
            for k_local in range(2):
                u_row = [0] * 6
                v_row = [0] * 6
                w_row = [0] * 4
                
                u_row[i * 3 + 2] = 1  # A[i, 2]
                v_row[2 * 2 + k_local] = 1  # B[2, k_local] (row 2 of B, local col k_local)
                w_row[k_local * 2 + i] = 1  # C[i, k_local]
                
                U.append(u_row)
                V.append(v_row)
                W.append(w_row)
    
    return U, V, W

def _build_schoolbook():
    """Schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
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
    return U, V, W

def _verify_decomp(U, V, W):
    """Verify that the decomposition exactly represents matrix multiplication."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    
    # Build the tensor T[i,j,k] from the decomposition
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This should equal 1 for all i,j,k.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if total != 1:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Try rank-44 first
    try:
        U, V, W = _build_rank44()
        if _verify_decomp(U, V, W):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to schoolbook
    U, V, W = _build_schoolbook()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
