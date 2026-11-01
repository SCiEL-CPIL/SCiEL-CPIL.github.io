"""Matrix multiplication 2x3x8 rank-minimal decomposition.

Strategy: Split B (3x8) into 4 blocks of (3x2). Each block gives a 2x3x2 sub-tensor.
The 2x3x2 tensor = (2x2x2 part) + (2x1x2 part).
  - 2x2x2 uses Strassen rank 7.
  - 2x1x2 uses 2 rank-1 terms.
Total per block: 9, total rank: 36.
"""
DIMENSIONS = [2, 3, 8]

# Strassen rank-7 decomposition for 2x2x2.
# Indices: u in {0,1}, v in {0,1}, w in {0,1}
# C = A*B where A, B are 2x2, C is 2x2.
# w convention: W[k*n+i, c] for n=p=2, so w index = k*2 + i.
STRASSEN_U = [
    [1, 1, 0, 0, 0, 0, 0],  # i=0
    [0, 0, 1, 1, 0, 0, 0],  # i=1
]
STRASSEN_V = [
    [1, 0, 1, 0, 1, 0, 0],  # j=0
    [0, 1, 0, 1, 0, 1, 1],  # j=1
]
# w: k*2 + i. For C[i,k] = sum_c U[i,c]*V[j,c]*W[k*2+i,c]
# Standard Strassen:
# m1 = (A00)(B00+B11) -> contributes to C00, C11
# m2 = (A10+A11)(B00) -> contributes to C10, C11
# m3 = (A00)(B01-B11) -> contributes to C00, C01
# m4 = (A11)(B10-B00) -> contributes to C10, C11
# m5 = (A00+A11)(B10+B11) -> contributes to C01, C10
# m6 = (A01-A10)(B10) -> contributes to C01, C10
# m7 = (A01)(B01-B10) -> contributes to C00, C01
#
# C00 = m1 - m3 + m7
# C01 = m3 + m5
# C10 = m2 - m4 - m6
# C11 = m1 + m2 - m4 + m5
#
# Let's build W[k*2+i, c] for c=0..6:
# c=0 (m1): C00 += 1, C11 += 1 -> W[0,0]=1 (i=0,k=0), W[3,0]=1 (i=1,k=1)
# c=1 (m2): C10 += 1, C11 += 1 -> W[2,1]=1 (i=1,k=0), W[3,1]=1 (i=1,k=1)
# c=2 (m3): C00 -= 1, C01 += 1 -> W[0,2]=-1 (i=0,k=0), W[1,2]=1 (i=0,k=1)
# c=3 (m4): C10 -= 1, C11 -= 1 -> W[2,3]=-1 (i=1,k=0), W[3,3]=-1 (i=1,k=1)
# c=4 (m5): C01 += 1, C10 += 1 -> W[1,4]=1 (i=0,k=1), W[2,4]=1 (i=1,k=0)
# c=5 (m6): C01 -= 1, C10 -= 1 -> W[1,5]=-1 (i=0,k=1), W[2,5]=-1 (i=1,k=0)
# c=6 (m7): C00 += 1, C01 -= 1 -> W[0,6]=1 (i=0,k=0), W[1,6]=-1 (i=0,k=1)

STRASSEN_W = [
    [1, 0, -1, 0, 0, 0, 1],   # k=0, i=0 -> C00
    [0, 0, 1, 0, 1, -1, -1],  # k=1, i=0 -> C01
    [0, 1, 0, -1, 1, -1, 0],  # k=0, i=1 -> C10
    [1, 1, 0, -1, 0, 0, 0],   # k=1, i=1 -> C11
]


def construct():
    n, m, p = 2, 3, 8
    # 4 blocks, each 2 columns of B
    num_blocks = p // 2  # 4
    rank_per_block = 7 + 2  # Strassen + 2 rank-1 for 2x1x2
    total_rank = num_blocks * rank_per_block

    U = [[0] * total_rank for _ in range(n * m)]  # 6 x 36
    V = [[0] * total_rank for _ in range(m * p)]  # 24 x 36
    W = [[0] * total_rank for _ in range(p * n)]  # 16 x 36

    for b in range(num_blocks):
        base = b * rank_per_block
        col_start = b * 2  # columns of B for this block

        # Strassen part: uses rows j=0,1 of A and columns col_start, col_start+1 of B
        # U: n*m = 6 rows, indexed by i*m + j = i*3 + j
        # For i in {0,1}, j in {0,1}:
        for i in range(2):
            for j in range(2):
                u_idx = i * 3 + j
                for c in range(7):
                    U[u_idx][base + c] = STRASSEN_U[i][c]

        # V: m*p = 24 rows, indexed by j*p + k
        # For j in {0,1}, k in {col_start, col_start+1}:
        for j in range(2):
            for dk in range(2):
                k = col_start + dk
                v_idx = j * 8 + k
                for c in range(7):
                    V[v_idx][base + c] = STRASSEN_V[j][dk]

        # W: p*n = 16 rows, indexed by k*n + i = k*2 + i
        # For i in {0,1}, k in {col_start, col_start+1}:
        for i in range(2):
            for dk in range(2):
                k = col_start + dk
                w_idx = k * 2 + i
                for c in range(7):
                    W[w_idx][base + c] = STRASSEN_W[i][dk]  # STRASSEN_W[i][dk] = W[k*2+i, c]

        # 2x1x2 part: uses row j=2 of A and columns col_start, col_start+1 of B
        # Tensor: sum over i,k of e_i ⊗ e_2 ⊗ e_k
        # This can be done with 2 rank-1 terms:
        # Term 1: u=(1,0), v=(1,1), w=(1,0) -> C00 += A02*B2,col_start
        # Term 2: u=(0,1), v=(1,1), w=(0,1) -> C11 += A12*B2,col_start+1
        # Wait, that's not right. Let me think again.
        # We need: C[i,k] += A[i,2]*B[2,k] for i in {0,1}, k in {col_start, col_start+1}
        # This is a 2x1x2 tensor. Rank of 2x1x2 tensor:
        # T_{i,2,k} = delta_{i,k} if we think of it as identity... no.
        # Actually T_{i,2,k} = 1 for all i, k in the block.
        # So T = (1,1) ⊗ (1) ⊗ (1,1) where the (1,1) is for the 2 columns.
        # That's rank 1!
        # u = (1,1) for i in {0,1}
        # v = (1) for j=2 (but v has 2 entries for j in {0,1} of the sub-block... 
        #   actually for the 2x1x2, v is just 1 entry since j is fixed at 2)
        # w = (1,1) for k in {col_start, col_start+1}
        # 
        # So rank 1 suffices:
        c = base + 7
        for i in range(2):
            U[i * 3 + 2][c] = 1  # i*3 + j=2
        V[2 * 8 + col_start][c] = 1
        V[2 * 8 + col_start + 1][c] = 1
        for i in range(2):
            W[(col_start) * 2 + i][c] = 1
            W[(col_start + 1) * 2 + i][c] = 1

    return {"U": U, "V": V, "W": W}
