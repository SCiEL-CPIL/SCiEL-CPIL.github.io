"""Matrix multiplication tensor decomposition for 2x4 * 4x5."""
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """Rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2.
    Returns (U, V, W) each as list of 7-element lists.
    U has 4 rows (i*m+j), V has 4 rows (j*p+k), W has 4 rows (k*n+i).
    """
    # Strassen's 7 products:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # A layout: A11=(0,0), A12=(0,1), A21=(1,0), A22=(1,1) -> U rows 0,1,2,3
    # B layout: B11=(0,0), B12=(0,1), B21=(1,0), B22=(1,1) -> V rows 0,1,2,3
    # C layout: C11=(0,0), C12=(0,1), C21=(1,0), C22=(1,1) -> W rows 0,1,2,3
    # W convention: W[k*n+i, c] where n=2, so W row = k*2+i
    # C11 -> k=0,i=0 -> row 0
    # C12 -> k=0,i=1 -> row 1
    # C21 -> k=1,i=0 -> row 2
    # C22 -> k=1,i=1 -> row 3

    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]

    # M1: (A11+A22)*(B11+B22)
    U[0][0] = 1; U[3][0] = 1  # A11, A22
    V[0][0] = 1; V[3][0] = 1  # B11, B22
    W[0][0] = 1; W[3][0] = 1  # C11, C22

    # M2: (A21+A22)*B11
    U[2][1] = 1; U[3][1] = 1  # A21, A22
    V[0][1] = 1              # B11
    W[2][1] = 1; W[3][1] = -1  # C21, -C22

    # M3: A11*(B12-B22)
    U[0][2] = 1              # A11
    V[1][2] = 1; V[3][2] = -1  # B12, -B22
    W[1][2] = 1; W[3][2] = 1  # C12, C22

    # M4: A22*(B21-B11)
    U[3][3] = 1              # A22
    V[2][3] = 1; V[0][3] = -1  # B21, -B11
    W[0][3] = 1; W[2][3] = 1  # C11, C21

    # M5: (A11+A12)*B22
    U[0][4] = 1; U[1][4] = 1  # A11, A12
    V[3][4] = 1              # B22
    W[0][4] = -1; W[1][4] = 1  # -C11, C12

    # M6: (A21-A11)*(B11+B12)
    U[2][5] = 1; U[0][5] = -1  # A21, -A11
    V[0][5] = 1; V[1][5] = 1  # B11, B12
    W[3][5] = 1              # C22

    # M7: (A12-A22)*(B21+B22)
    U[1][6] = 1; U[3][6] = -1  # A12, -A22
    V[2][6] = 1; V[3][6] = 1  # B21, B22
    W[0][6] = 1              # C11

    return U, V, W


def _rank4_2x2_2x1():
    """Rank-4 decomposition for 2x2 * 2x1 -> 2x1.
    C = A * b where A is 2x2, b is 2x1.
    C1 = A11*b1 + A12*b2
    C2 = A21*b1 + A22*b2
    Trivial rank 4: each product separately.
    Returns (U, V, W): U 4 rows, V 2 rows, W 2 rows.
    """
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(2)]
    W = [[0]*4 for _ in range(2)]

    # C1 = A11*b1: U[0], V[0], W[0]
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    # C1 += A12*b2: U[1], V[1], W[0]
    U[1][1] = 1; V[1][1] = 1; W[0][1] = 1
    # C2 = A21*b1: U[2], V[0], W[1]
    U[2][2] = 1; V[0][2] = 1; W[1][2] = 1
    # C2 += A22*b2: U[3], V[1], W[1]
    U[3][3] = 1; V[1][3] = 1; W[1][3] = 1

    return U, V, W


def _rank4_2x1_2x2():
    """Rank-4 decomposition for 2x1 * 1x2 -> 2x2.
    C = a * b^T where a is 2x1, b is 1x2.
    C11 = a1*b1, C12 = a1*b2, C21 = a2*b1, C22 = a2*b2.
    Trivial rank 4.
    Returns (U, V, W): U 2 rows, V 2 rows, W 4 rows.
    W layout for 2x2 output with n=2: C11->k=0,i=0->row 0; C12->k=1,i=0->row 1; C21->k=0,i=1->row 2; C22->k=1,i=1->row 3.
    Wait, the output is 2x2 but in our full problem the output dimension is p=5 and n=2.
    Actually for the sub-block the output is 2x2, and W index is k*n+i.
    For a 2x2 output: C[k][i] with k in {0,1}, i in {0,1}, row = k*2+i.
    C[0][0]=row0, C[0][1]=row1, C[1][0]=row2, C[1][1]=row3.
    """
    U = [[0]*4 for _ in range(2)]
    V = [[0]*4 for _ in range(2)]
    W = [[0]*4 for _ in range(4)]

    # C[0][0] = a1*b1
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    # C[0][1] = a1*b2
    U[0][1] = 1; V[1][1] = 1; W[1][1] = 1
    # C[1][0] = a2*b1
    U[1][2] = 1; V[0][2] = 1; W[2][2] = 1
    # C[1][1] = a2*b2
    U[1][3] = 1; V[1][3] = 1; W[3][3] = 1

    return U, V, W


def _place_block(U_full, V_full, W_full, U_block, V_block, W_block,
                 u_map, v_map, w_map, start_col):
    """Place a block decomposition into full matrices at given row/column mappings."""
    rank_block = len(U_block[0])
    for c in range(rank_block):
        col = start_col + c
        for u_idx, u_val in enumerate(U_block):
            if u_val[c] != 0:
                U_full[u_map[u_idx]][col] += u_val[c]
        for v_idx, v_val in enumerate(V_block):
            if v_val[c] != 0:
                V_full[v_map[v_idx]][col] += v_val[c]
        for w_idx, w_val in enumerate(W_block):
            if w_val[c] != 0:
                W_full[w_map[w_idx]][col] += w_val[c]
    return start_col + rank_block


def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    # U has n*m = 8 rows, V has m*p = 20 rows, W has p*n = 10 rows
    total_rank = 7 + 4 + 4 + 7 + 7 + 4 + 4 + 7 + 7 + 4  # Let me recalculate

    # Strategy: Split A (2x4) into two 2x2 blocks: A = [A1 | A2]
    # Split B (4x5) into two 2x5 blocks: B = [B1; B2]
    # C = A1*B1 + A2*B2  (2x5)
    #
    # Further split:
    # A1 (2x2) * B1 (2x5): split B1 into [B1a(2x2) | B1b(2x2) | B1c(2x1)]
    #   -> A1*B1a (2x2, rank 7) + A1*B1b (2x2, rank 7) + A1*B1c (2x1, rank 4)
    # A2 (2x2) * B2 (2x5): split B2 into [B2a(2x2) | B2b(2x2) | B2c(2x1)]
    #   -> A2*B2a (2x2, rank 7) + A2*B2b (2x2, rank 7) + A2*B2c (2x1, rank 4)
    #
    # Total rank = 7+7+4 + 7+7+4 = 36
    # That's better than 40 (schoolbook) but not as good as the 16 mentioned in directive.

    # Actually, let me try a different split to get rank 16 as suggested.
    # The directive says: 2x2x2 block (rank 7) + 2x2x3 block (rank 9) = 16.
    # But 2x2x3 with rank 9? The trivial is 12. Let me think...
    #
    # Actually, let me just do the straightforward partition and get rank 36.
    # Or better, let me try to be smarter.

    # Let me use the partition:
    # A = [A1(2x2) | A2(2x2)], B = [B1(2x5); B2(2x5)]
    # C = A1*B1 + A2*B2
    #
    # For A1*B1 (2x2 * 2x5): 
    #   We can view this as multiplying by each column of B1.
    #   B1 has columns b1, b2, b3, b4, b5 (each 2x1).
    #   A1*bj is 2x1. Rank of 2x2*2x1 is 4 (trivial).
    #   So A1*B1 has rank 4*5 = 20 if done column by column.
    #   But we can do better by grouping.
    #
    # Alternative: Use the fact that for fixed A (2x2), multiplying by B (2x5)
    # can be seen as: for each of the 5 columns, do a 2x2*2x1 multiply.
    # Each such multiply has rank 4. Total = 20 per A block, so 40 total.
    #
    # Let me just go with the Strassen-based approach for the 2x2*2x2 parts.
    # Partition B into 2x2 + 2x2 + 2x1:
    # A1*B1a: 2x2*2x2 -> rank 7 (Strassen)
    # A1*B1b: 2x2*2x2 -> rank 7 (Strassen)
    # A1*B1c: 2x2*2x1 -> rank 4
    # A2*B2a: 2x2*2x2 -> rank 7 (Strassen)
    # A2*B2b: 2x2*2x2 -> rank 7 (Strassen)
    # A2*B2c: 2x2*2x1 -> rank 4
    # Total: 7+7+4+7+7+4 = 36

    total_rank = 36
    U_full = [[0]*total_rank for _ in range(n*m)]   # 8 rows
    V_full = [[0]*total_rank for _ in range(m*p)]   # 20 rows
    W_full = [[0]*total_rank for _ in range(p*n)]   # 10 rows

    col = 0

    # Row mappings for U (n*m = 8 rows, indexed by i*m+j):
    # A1: i=0,1; j=0,1 -> rows 0,1,2,3
    # A2: i=0,1; j=2,3 -> rows 4,5,6,7
    # Wait: U row index = i*m + j where m=4
    # A1: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
    # A2: (0,2)->2, (0,3)->3, (1,2)->6, (1,3)->7

    u_map_A1 = [0, 1, 4, 5]  # (i,j) = (0,0),(0,1),(1,0),(1,1)
    u_map_A2 = [2, 3, 6, 7]  # (i,j) = (0,2),(0,3),(1,0+2),(1,1+2) -> wait
    # i=0,j=2 -> 0*4+2=2
    # i=0,j=3 -> 0*4+3=3
    # i=1,j=2 -> 1*4+2=6
    # i=1,j=3 -> 1*4+3=7
    u_map_A2 = [2, 3, 6, 7]

    # V row index = j*p + k where p=5
    # B1: j=0,1; B2: j=2,3
    # B1a: j=0,1; k=0,1 -> rows 0,1,5,6
    # B1b: j=0,1; k=2,3 -> rows 2,3,7,8
    # B1c: j=0,1; k=4   -> rows 4,9
    # B2a: j=2,3; k=0,1 -> rows 10,11,15,16
    # B2b: j=2,3; k=2,3 -> rows 12,13,17,18
    # B2c: j=2,3; k=4   -> rows 14,19

    v_map_B1a = [0, 1, 5, 6]    # (j,k) = (0,0),(0,1),(1,0),(1,1)
    v_map_B1b = [2, 3, 7, 8]    # (j,k) = (0,2),(0,3),(1,2),(1,3)
    v_map_B1c = [4, 9]          # (j,k) = (0,4),(1,4)
    v_map_B2a = [10, 11, 15, 16]  # (j,k) = (2,0),(2,1),(3,0),(3,1)
    v_map_B2b = [12, 13, 17, 18]  # (j,k) = (2,2),(2,3),(3,2),(3,3)
    v_map_B2c = [14, 19]        # (j,k) = (2,4),(3,4)

    # W row index = k*n + i where n=2
    # C output is 5x2 (p x n)
    # C1 (from A1*B1): rows for k=0..4, i=0..1
    # C2 (from A2*B2): same rows, different values
    # W rows:
    # k=0: i=0->0, i=1->1
    # k=1: i=0->2, i=1->3
    # k=2: i=0->4, i=1->5
    # k=3: i=0->6, i=1->7
    # k=4: i=0->8, i=1->9

    # For Strassen 2x2 blocks, W is 4 rows corresponding to C11,C12,C21,C22
    # C11 -> k=0,i=0 -> row 0
    # C12 -> k=0,i=1 -> row 1
    # C21 -> k=1,i=0 -> row 2
    # C22 -> k=1,i=1 -> row 3

    # For B1a (k=0,1): w_map = [0,1,2,3]
    # For B1b (k=2,3): w_map = [4,5,6,7]
    # For B1c (k=4): w_map = [8,9]
    # For B2a (k=0,1): w_map = [0,1,2,3]
    # For B2b (k=2,3): w_map = [4,5,6,7]
    # For B2c (k=4): w_map = [8,9]

    w_map_k01 = [0, 1, 2, 3]
    w_map_k23 = [4, 5, 6, 7]
    w_map_k4 = [8, 9]

    # Block 1: A1 * B1a (2x2 * 2x2) - Strassen
    U1, V1, W1 = _strassen_2x2()
    col = _place_block(U_full, V_full, W_full, U1, V1, W1,
                       u_map_A1, v_map_B1a, w_map_k01, col)

    # Block 2: A1 * B1b (2x2 * 2x2) - Strassen
    U2, V2, W2 = _strassen_2x2()
    col = _place_block(U_full, V_full, W_full, U2, V2, W2,
                       u_map_A1, v_map_B1b, w_map_k23, col)

    # Block 3: A1 * B1c (2x2 * 2x1) - rank 4
    U3, V3, W3 = _rank4_2x2_2x1()
    col = _place_block(U_full, V_full, W_full, U3, V3, W3,
                       u_map_A1, v_map_B1c, w_map_k4, col)

    # Block 4: A2 * B2a (2x2 * 2x2) - Strassen
    U4, V4, W4 = _strassen_2x2()
    col = _place_block(U_full, V_full, W_full, U4, V4, W4,
                       u_map_A2, v_map_B2a, w_map_k01, col)

    # Block 5: A2 * B2b (2x2 * 2x2) - Strassen
    U5, V5, W5 = _strassen_2x2()
    col = _place_block(U_full, V_full, W_full, U5, V5, W5,
                       u_map_A2, v_map_B2b, w_map_k23, col)

    # Block 6: A2 * B2c (2x2 * 2x1) - rank 4
    U6, V6, W6 = _rank4_2x2_2x1()
    col = _place_block(U_full, V_full, W_full, U6, V6, W6,
                       u_map_A2, v_map_B2c, w_map_k4, col)

    return {"U": U_full, "V": V_full, "W": W_full}

# EVOLVE-BLOCK-END
