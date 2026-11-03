"""Matrix multiplication 6x6x6: Strassen-style block decomposition (7 blocks of 3x3)."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def _build_3x3_block():
    """Schoolbook rank-27 decomposition for 3x3 matrix multiplication.
    Returns U, V, W each of shape (9, 27) with entries in {0,1}."""
    n = m = p = 3
    rank = n * m * p  # 27
    U = [[0] * rank for _ in range(n * m)]   # 9 x 27
    V = [[0] * rank for _ in range(m * p)]   # 9 x 27
    W = [[0] * rank for _ in range(p * n)]   # 9 x 27
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def construct():
    """Construct a rank-189 exact tensor decomposition for 6x6x6 matrix multiplication
    using Strassen's 2x2 block scheme with 7 sub-blocks of size 3x3x3 (rank 27 each).
    
    The 6x6 matrices are partitioned into 2x2 blocks of 3x3:
        A = [[A00, A01],
             [A10, A11]]
        B = [[B00, B01],
             [B10, B11]]
    
    Strassen computes:
        M1 = (A00 + A11) * (B00 + B11)
        M2 = (A10 + A11) * B00
        M3 = A00 * (B01 - B11)
        M4 = A11 * (B10 - B00)
        M5 = (A00 + A01) * B11
        M6 = (A10 - A00) * (B00 + B01)
        M7 = (A11 - A10) * (B10 + B11)
    
    Result:
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M6
    
    Total rank = 7 * 27 = 189
    """
    deadline = time.time() + 110.0  # safety margin before 120s limit
    
    # Build the 3x3 block decomposition once
    Ub, Vb, Wb = _build_3x3_block()
    sub_rank = 27  # rank of each 3x3 block
    
    # Total rank
    rank = 7 * sub_rank  # 189
    
    # U: shape (36, 189)  -- rows indexed by (block_row, local_row) where block_row in {0,1}, local_row in 0..8
    # V: shape (36, 189)  -- rows indexed by (block_col, local_col) where block_col in {0,1}, local_col in 0..8
    # W: shape (36, 189)  -- rows indexed by (result_col, result_row) i.e. (p_block, n_block) then local
    
    # U rows: 6x6 matrix A partitioned as 2x2 blocks of 3x3
    # Row index in U: block_i * 9 + local_i, where block_i in {0,1}, local_i in 0..8
    # V rows: block_j * 9 + local_j
    # W rows: block_k * 9 + local_k  (output C partitioned as 2x2 blocks of 3x3)
    #   Output C: C[global_row, global_col] where global_row = block_i*3 + local_i, global_col = block_j*3 + local_j
    #   W index: k*n + i in the 6x6 sense -> (block_k, local_k) for col, (block_i, local_i) for row
    #   So W row index = block_k * 9 + local_k? No wait.
    #   
    #   The W matrix has rows indexed by (k, i) in the full 6x6 multiplication: C[i,k] = sum_j A[i,j] B[j,k]
    #   W index is k*n + i = k*6 + i.
    #   With blocks: k = block_k * 3 + local_k, i = block_i * 3 + local_i
    #   So W row = block_k * 18 + block_i * 9 + local_k * 3 + local_i? No:
    #   k*6 + i = (block_k*3 + local_k)*6 + (block_i*3 + local_i) = block_k*18 + local_k*6 + block_i*3 + local_i
    #   
    #   Hmm, this is getting complex. Let me think differently.
    #
    #   Actually, the convention is W[k*n + i, r]. For the full 6x6:
    #   k ranges 0..5, i ranges 0..5.
    #   k = block_k * 3 + local_k, i = block_i * 3 + local_i
    #   W row index = k*6 + i = (block_k*3 + local_k)*6 + (block_i*3 + local_i)
    #              = 18*block_k + 6*local_k + 3*block_i + local_i
    #
    #   U row index = i*m + j = i*6 + j = (block_i*3 + local_i)*6 + (block_j*3 + local_j)
    #              = 18*block_i + 6*local_i + 3*block_j + local_j
    #
    #   V row index = j*p + k = j*6 + k = (block_j*3 + local_j)*6 + (block_k*3 + local_k)
    #              = 18*block_j + 6*local_j + 3*block_k + local_k
    
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    # Strassen factors:
    # Each M_s (s=1..7) is computed as (P_s) * (Q_s) where P_s and Q_s are 3x3 blocks.
    # P_s is a linear combination of A-blocks, Q_s is a linear combination of B-blocks.
    # For each term in P_s with coefficient c on A[bi, bj], and each term in Q_s with coefficient d on B[bj2, bk],
    # the sub-decomposition contributes to U, V, W.
    
    # Define the 7 Strassen products:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A11-A10)*(B10+B11)
    
    # For each product s, define:
    #   A_terms: list of (coeff, block_i, block_j) -- linear combo of A blocks
    #   B_terms: list of (coeff, block_j, block_k) -- linear combo of B blocks
    #   C_terms: list of (coeff, block_i, block_k) -- where the result goes (with sign)
    
    strassen_products = [
        # M1
        {
            'A': [(1, 0, 0), (1, 1, 1)],
            'B': [(1, 0, 0), (1, 1, 1)],
            'C': [(1, 0, 0), (1, 1, 1)],
        },
        # M2
        {
            'A': [(1, 1, 0), (1, 1, 1)],
            'B': [(1, 0, 0)],
            'C': [(-1, 1, 1), (1, 1, 0)],
        },
        # M3
        {
            'A': [(1, 0, 0)],
            'B': [(1, 0, 1), (-1, 1, 1)],
            'C': [(1, 0, 1), (1, 1, 1)],
        },
        # M4
        {
            'A': [(1, 1, 1)],
            'B': [(1, 1, 0), (-1, 0, 0)],
            'C': [(1, 0, 0), (1, 1, 0)],
        },
        # M5
        {
            'A': [(1, 0, 0), (1, 0, 1)],
            'B': [(1, 1, 1)],
            'C': [(-1, 0, 0)],
        },
        # M6
        {
            'A': [(1, 1, 0), (-1, 0, 0)],
            'B': [(1, 0, 0), (1, 0, 1)],
            'C': [(1, 1, 1)],
        },
        # M7
        {
            'A': [(1, 1, 1), (-1, 1, 0)],
            'B': [(1, 1, 0), (1, 1, 1)],
            'C': [(1, 0, 0)],
        },
    ]
    
    # For each Strassen product s, we have sub_rank=27 columns.
    # Within each product, the A-side is a linear combination of A-blocks, B-side of B-blocks.
    # The decomposition of a single 3x3 product (A_block * B_block) has 27 terms.
    # For a linear combination: sum_{a} ca * A[ai, aj] * sum_{b} cb * B[bj, bk]
    # = sum_{a,b} ca*cb * A[ai,aj] * B[bj,bk]
    # Each A[ai,aj] * B[bj,bk] is a 3x3x3 tensor that decomposes into 27 rank-1 terms.
    # But we can't just sum 27 * |A_terms| * |B_terms| because that would blow up the rank.
    #
    # Key insight: For each of the 27 basis terms (local_i, local_j, local_k) in the 3x3 schoolbook,
    # the contribution to U at row (ai, aj) is ca, to V at row (bj, bk) is cb, and to W at row (ci, ck) is cc.
    # But we need to be careful: the 27 terms in the 3x3 decomposition are indexed by (li, lj, lk) where
    # U[li*3+lj] = 1, V[lj*3+lk] = 1, W[lk*3+li] = 1.
    #
    # For the combined product:
    #   U contribution at row (ai, aj) for basis term (li, lj, lk): sum over a-terms where (ai,aj) matches
    #   Actually no. Let me re-derive.
    #
    # The 3x3 multiplication tensor: T[i,j,k] = 1 if the term is A[i,j]*B[j,k] contributing to C[k,i].
    # Schoolbook: T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    #
    # For our block product: the "effective" A is P = sum_a ca * A[ai, aj], effective B is Q = sum_b cb * B[bj, bk].
    # The product P*Q contributes to C blocks via the C_terms.
    #
    # The tensor for this block product is:
    #   sum_{li, lj, lk} (sum_a ca * e_{ai,aj,li,lj}) ⊗ (sum_b cb * e_{bj,bk,lj,lk}) ⊗ (sum_c cc * e_{ci,ck,lk,li})
    #
    # Wait, I need to be more careful. The 3x3 schoolbook decomposition gives:
    #   For each (li, lj, lk): U[li*3+lj] = 1, V[lj*3+lk] = 1, W[lk*3+li] = 1
    #   This means: sum_{li,lj,lk} U[...] * V[...] * W[...] = sum_{li,lj,lk} A[li,lj] * B[lj,lk] * C[lk,li]
    #
    # Now for the block version:
    #   A[li,lj] in block (ai,aj) -> global A[(ai*3+li), (aj*3+lj)]
    #   B[lj,lk] in block (bj,bk) -> global B[(bj*3+lj), (bk*3+lk)]
    #   C[lk,li] in block (ci,ck) -> global C[(ci*3+li), (ck*3+lk)]  [C row=ci*3+li, col=ck*3+lk]
    #
    # The Strassen formula says:
    #   C[ci,ck] += cc * (sum_a ca * A[ai,aj]) * (sum_b cb * B[bj,bk])
    #   = sum_{li,lj,lk} sum_a sum_b cc * ca * cb * A[(ai*3+li),(aj*3+lj)] * B[(bj*3+lj),(bk*3+lk)] * C[(ci*3+li),(ck*3+lk)]
    #
    # For this to match the tensor decomposition, for each (li,lj,lk) and each column c in this product's 27 columns:
    #   U[row_A, c] = sum_a ca (for the a-term that has block (ai,aj)) -- but we need U to have a single value per row per column.
    #   
    #   The issue: U[row, c] must be a scalar. If multiple a-terms contribute to the same global row, they add.
    #   Global A row = ai*3 + li, col = aj*3 + lj.
    #   For a given (li, lj) and column c (which corresponds to a specific (li,lj,lk) basis term),
    #   U[(ai*3+li)*6 + (aj*3+lj), c] = ca for the a-term with blocks (ai,aj).
    #   But if two different a-terms have the same (ai,aj), they'd both contribute -- but in our Strassen formulas,
    #   each (ai,aj) appears at most once in the A list. Let me verify... yes, in each product, the A blocks are distinct.
    #
    #   Similarly for V and W.
    #
    #   So the construction is:
    #   For product s (s=0..6), for each basis term (li, lj, lk) (27 total):
    #     column = s * 27 + (li*3 + lj)*3 + lk  ... wait, let me index properly.
    #     Actually the 3x3 basis index: let's say idx = li*9 + lj*3 + lk (0..26)
    #     column = s * 27 + idx
    #     
    #     For each a-term (ca, ai, aj) in product s's A list:
    #       U_row = (ai*3 + li) * 6 + (aj*3 + lj)
    #       U[U_row][column] += ca
    #     
    #     For each b-term (cb, bj, bk) in product s's B list:
    #       V_row = (bj*3 + lj) * 6 + (bk*3 + lk)
    #       V[V_row][column] += cb
    #     
    #     For each c-term (cc, ci, ck) in product s's C list:
    #       W_row = (ck*3 + lk) * 6 + (ci*3 + li)  # W[k*n+i] where k=ck*3+lk, i=ci*3+li
    #       W[W_row][column] += cc
    
    for s, prod in enumerate(strassen_products):
        for li in range(3):
            for lj in range(3):
                for lk in range(3):
                    col = s * 27 + li * 9 + lj * 3 + lk
                    # U contributions
                    for (ca, ai, aj) in prod['A']:
                        u_row = (ai * 3 + li) * 6 + (aj * 3 + lj)
                        U[u_row][col] += ca
                    # V contributions
                    for (cb, bj, bk) in prod['B']:
                        v_row = (bj * 3 + lj) * 6 + (bk * 3 + lk)
                        V[v_row][col] += cb
                    # W contributions
                    for (cc, ci, ck) in prod['C']:
                        w_row = (ck * 3 + lk) * 6 + (ci * 3 + li)
                        W[w_row][col] += cc
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
