import time

_DIMENSIONS = [2, 4, 6]
_DEADLINE = 120.0

def construct():
    _t0 = time.time()
    n, m, p = 2, 4, 6
    
    # Strassen 2x2 @ 2x2 rank-7 decomposition
    # Local indices: 0=a11/b11/c11, 1=a12/b12/c12, 2=a21/b21/c21, 3=a22/b22/c22
    strassen_U = [
        {0: 1, 3: 1},   # P1: (a11+a22)
        {2: 1, 3: 1},   # P2: (a21+a22)
        {0: 1},         # P3: a11
        {3: 1},         # P4: a22
        {0: 1, 1: 1},   # P5: (a11+a12)
        {0: -1, 2: 1},  # P6: (a21-a11)
        {1: 1, 3: -1},  # P7: (a12-a22)
    ]
    strassen_V = [
        {0: 1, 3: 1},   # (b11+b22)
        {0: 1},         # b11
        {1: 1, 3: -1},  # (b12-b22)
        {2: 1, 0: -1},  # (b21-b11)
        {3: 1},         # b22
        {0: 1, 1: 1},   # (b11+b12)
        {2: 1, 3: 1},   # (b21+b22)
    ]
    strassen_W = [
        {0: 1, 3: 1},   # P1 -> c11, c22
        {2: 1, 3: -1},  # P2 -> c21, -c22
        {1: 1, 3: 1},   # P3 -> c12, c22
        {0: 1, 2: 1},   # P4 -> c11, c21
        {0: -1, 1: 1},  # P5 -> -c11, c12
        {3: 1},         # P6 -> c22
        {0: 1},         # P7 -> c11
    ]

    # We need 2x4 @ 4x6. Partition A into two 2x2 blocks (left/right), 
    # B into two 2x6 blocks (top/bottom). C = A_left @ B_top + A_right @ B_bottom
    # Each 2x2 @ 2x6 can be done with 3 Strassen 2x2@2x2 blocks = 7*3 = 21 rank terms
    # Total: 2 * 21 = 42 rank terms
    
    rank_total = 42
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]

    col_offset = 0
    # For each of the 2 A-blocks (left, right) and 2 B-blocks (top, bottom)
    # A_left multiplies B_top, A_right multiplies B_bottom
    for a_block in range(2):  # 0=left, 1=right
        for b_block in range(2):  # 0=top, 1=bottom
            # A_block is 2x2, B_block is 2x6
            # Split B_block into 3 2x2 sub-blocks
            for b_sub in range(3):
                for prod_idx in range(7):
                    col = col_offset + prod_idx
                    
                    # U: A_block indices: local 0=(0,a_block*2), 1=(0,a_block*2+1), 2=(1,a_block*2), 3=(1,a_block*2+1)
                    for local_idx, sign in strassen_U[prod_idx].items():
                        i_loc = local_idx // 2
                        j_loc = local_idx % 2
                        U[i_loc * m + a_block * 2 + j_loc][col] += sign
                    
                    # V: B_block indices: local 0=(b_block*2, b_sub*2), 1=(b_block*2, b_sub*2+1), 
                    #    2=(b_block*2+1, b_sub*2), 3=(b_block*2+1, b_sub*2+1)
                    for local_idx, sign in strassen_V[prod_idx].items():
                        i_loc = local_idx // 2
                        j_loc = local_idx % 2
                        V[(b_block * 2 + i_loc) * p + b_sub * 2 + j_loc][col] += sign
                    
                    # W: C indices: local 0=(b_sub*2, 0), 1=(b_sub*2+1, 0), 2=(b_sub*2, 1), 3=(b_sub*2+1, 1)
                    for local_idx, sign in strassen_W[prod_idx].items():
                        i_loc = local_idx // 2
                        j_loc = local_idx % 2
                        W[(b_sub * 2 + j_loc) * n + i_loc][col] += sign
                
                col_offset += 7
    
    if time.time() - _t0 > _DEADLINE - 5.0:
        raise RuntimeError("Deadline exceeded during construction")

    return {"U": U, "V": V, "W": W}
