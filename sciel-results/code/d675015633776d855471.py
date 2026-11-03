import time
import sys
from itertools import product

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Use exhaustive search over block splits to find optimal decomposition
    # with Strassen for 2x2 blocks.
    #
    # The knowledge says: "Exhaustive search over all block split compositions
    # finds the optimal partitioning that maximizes 2x2x2 Strassen blocks,
    # achieving rank 48 vs schoolbook 75"
    #
    # Let me implement this by trying different splits of 5 into parts
    # and computing the rank for each, using Strassen for 2x2 blocks.
    
    def strassen_rank_2x2():
        return 7  # Strassen rank for 2x2
    
    def schoolbook_rank(n1, m1, p1):
        return n1 * m1 * p1
    
    # Try all ways to split 5 into 2 or 3 parts
    # For each split, compute the total rank
    
    best_rank = 75
    best_split = None
    
    # Split inner dimension m=5 into parts
    # And outer dimensions n=3, p=5 into corresponding parts
    
    # For simplicity, try splits of 5 into 2 parts: 2+3, 3+2
    # And into 3 parts: 2+2+1, 2+1+2, 1+2+2, etc.
    
    # For a split of m into m1+m2, and n into n1+n2, p into p1+p2:
    # Blocks: A11(n1,m1)*B11(m1,p1), A12(n1,m2)*B21(m2,p1),
    #         A21(n2,m1)*B12(m1,p2), A22(n2,m2)*B22(m2,p2)
    #
    # For 2x2 blocks, use Strassen; for others, use schoolbook.
    
    # Try split m=5 as 2+3, n=3 as 2+1, p=5 as 2+3
    # A11(2,2)*B11(2,2): Strassen -> 7
    # A12(2,3)*B21(3,2): schoolbook -> 2*3*2 = 12
    # A21(1,2)*B12(2,3): schoolbook -> 1*2*3 = 6
    # A22(1,3)*B22(3,3): schoolbook -> 1*3*3 = 9
    # Total: 7 + 12 + 6 + 9 = 34
    
    # That's much better! Let me verify this decomposition works.
    
    # Use split: n=3 -> 2+1, m=5 -> 2+3, p=5 -> 2+3
    n1, n2 = 2, 1
    m1, m2 = 2, 3
    p1, p2 = 2, 3
    
    # Block 1: A11(n1,m1) * B11(m1,p1) = (2x2)*(2x2) -> Strassen rank 7
    # Block 2: A12(n1,m2) * B21(m2,p1) = (2x3)*(3x2) -> schoolbook rank 12
    # Block 3: A21(n2,m1) * B12(m1,p2) = (1x2)*(2x3) -> schoolbook rank 6
    # Block 4: A22(n2,m2) * B22(m2,p2) = (1x3)*(3x3) -> schoolbook rank 9
    # Total: 7 + 12 + 6 + 9 = 34
    
    rank = 34
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a decomposition
    def add_decomp(n_sub, m_sub, p_sub, U_sub, V_sub, W_sub, i_offset, j_offset, k_offset, sub_rank):
        nonlocal col
        for i in range(n_sub):
            for j in range(m_sub):
                for c in range(sub_rank):
                    if U_sub[i*m_sub+j][c] != 0:
                        U[(i + i_offset) * m + (j + j_offset)][col + c] += U_sub[i*m_sub+j][c]
        for j in range(m_sub):
            for k in range(p_sub):
                for c in range(sub_rank):
                    if V_sub[j*p_sub+k][c] != 0:
                        V[(j + j_offset) * p + (k + k_offset)][col + c] += V_sub[j*p_sub+k][c]
        for k in range(p_sub):
            for i in range(n_sub):
                for c in range(sub_rank):
                    if W_sub[k*n_sub+i][c] != 0:
                        W[(k + k_offset) * n + (i + i_offset)][col + c] += W_sub[k*n_sub+i][c]
        col += sub_rank
    
    # Block 1: (2x2)*(2x2) -> Strassen rank 7
    U1 = [[0]*7 for _ in range(2*2)]
    V1 = [[0]*7 for _ in range(2*2)]
    W1 = [[0]*7 for _ in range(2*2)]
    
    # Strassen decomposition for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # M1 = (A00+A11)(B00+B11)
    U1[0][0] = 1  # A00
    U1[3][0] = 1  # A11
    V1[0][0] = 1  # B00
    V1[1][1] = 1  # B11
    W1[0][0] = 1  # C00
    W1[0][1] = 1  # C01? No, let me redo this.
    
    # Let me use a cleaner Strassen implementation.
    # For 2x2:
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[1,0]-B[1,1])
    # M4 = A[1,1] * (B[0,1]-B[0,0])
    # M5 = (A[0,0]+A[0,1]) * B[1,1]
    # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[1,0])
    # M7 = (A[0,1]-A[1,1]) * (B[0,1]+B[1,1])
    #
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # M1
    U1[0][0] = 1  # A[0,0]
    U1[3][0] = 1  # A[1,1]
    V1[0][0] = 1  # B[0,0]
    V1[3][0] = 1  # B[1,1]
    W1[0][0] = 1  # C[0,0]
    W1[3][0] = 1  # C[1,1]
    
    # M2
    U1[2][1] = 1  # A[1,0]
    U1[3][1] = 1  # A[1,1]
    V1[0][1] = 1  # B[0,0]
    W1[2][1] = 1  # C[1,0]
    W1[3][1] = -1  # C[1,1]
    
    # M3
    U1[0][2] = 1  # A[0,0]
    V1[2][2] = 1  # B[1,0]
    V1[3][2] = -1  # B[1,1]
    W1[1][2] = 1  # C[0,1]
    W1[3][2] = 1  # C[1,1]
    
    # M4
    U1[3][3] = 1  # A[1,1]
    V1[1][3] = 1  # B[0,1]
    V1[0][3] = -1  # B[0,0]
    W1[0][3] = 1  # C[0,0]
    W1[2][3] = 1  # C[1,0]
    
    # M5
    U1[0][4] = 1  # A[0,0]
    U1[1][4] = 1  # A[0,1]
    V1[3][4] = 1  # B[1,1]
    W1[0][4] = -1  # C[0,0]
    W1[1][4] = 1  # C[0,1]
    
    # M6
    U1[2][5] = 1  # A[1,0]
    U1[0][5] = -1  # A[0,0]
    V1[0][5] = 1  # B[0,0]
    V1[2][5] = 1  # B[1,0]
    W1[3][5] = 1  # C[1,1]
    
    # M7
    U1[1][6] = 1  # A[0,1]
    U1[3][6] = -1  # A[1,1]
    V1[1][6] = 1  # B[0,1]
    V1[3][6] = 1  # B[1,1]
    W1[0][6] = 1  # C[0,0]
    
    add_decomp(2, 2, 2, U1, V1, W1, 0, 0, 0, 7)
    
    # Block 2: (2x3)*(3x2) -> schoolbook rank 12
    U2 = [[0]*12 for _ in range(2*3)]
    V2 = [[0]*12 for _ in range(3*2)]
    W2 = [[0]*12 for _ in range(2*2)]
    c = 0
    for i in range(2):
        for j in range(3):
            for k in range(2):
                U2[i*3+j][c] = 1
                V2[j*2+k][c] = 1
                W2[k*2+i][c] = 1
                c += 1
    add_decomp(2, 3, 2, U2, V2, W2, 0, 2, 0, 12)
    
    # Block 3: (1x2)*(2x3) -> schoolbook rank 6
    U3 = [[0]*6 for _ in range(1*2)]
    V3 = [[0]*6 for _ in range(2*3)]
    W3 = [[0]*6 for _ in range(3*1)]
    c = 0
    for i in range(1):
        for j in range(2):
            for k in range(3):
                U3[i*2+j][c] = 1
                V3[j*3+k][c] = 1
                W3[k*1+i][c] = 1
                c += 1
    add_decomp(1, 2, 3, U3, V3, W3, 2, 0, 2, 6)
    
    # Block 4: (1x3)*(3x3) -> schoolbook rank 9
    U4 = [[0]*9 for _ in range(1*3)]
    V4 = [[0]*9 for _ in range(3*3)]
    W4 = [[0]*9 for _ in range(3*1)]
    c = 0
    for i in range(1):
        for j in range(3):
            for k in range(3):
                U4[i*3+j][c] = 1
                V4[j*3+k][c] = 1
                W4[k*1+i][c] = 1
                c += 1
    add_decomp(1, 3, 3, U4, V4, W4, 2, 2, 2, 9)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

