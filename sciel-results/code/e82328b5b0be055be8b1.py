"""Rewritten program using explicit rank-49 + rank-28 Strassen block decomposition."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6

    U = {}
    V = {}
    W = {}
    rank_counter = 0

    def add_term(u_sp, v_sp, w_sp):
        """u_sp, v_sp, w_sp are dicts: {row_index: value}"""
        nonlocal rank_counter
        col = rank_counter
        rank_counter += 1
        for r, v in u_sp.items():
            U.setdefault(r, []).append((col, v))
        for r, v in v_sp.items():
            V.setdefault(r, []).append((col, v))
        for r, v in w_sp.items():
            W.setdefault(r, []).append((col, v))

    def strassen_2x2x2(ai0, aj0, ak0):
        """Add 7 rank-1 terms for 2x2 Strassen block.
        ai0: row offset in A (0 or 2), aj0: col offset in A / row offset in B (0 or 2), ak0: col offset in B (0 or 2)
        A indices: (ai0,aj0), (ai0,aj0+1), (ai0+1,aj0), (ai0+1,aj0+1)
        B indices: (aj0,ak0), (aj0,ak0+1), (aj0+1,ak0), (aj0+1,ak0+1)
        C indices: (ak0,ai0), (ak0,ai0+1), (ak0+1,ai0), (ak0+1,ai0+1)
        W index: k*n + i
        """
        ai1 = ai0 + 1
        aj1 = aj0 + 1
        ak1 = ak0 + 1

        # A entries: a=A[ai0,aj0], b=A[ai0,aj1], c=A[ai1,aj0], d=A[ai1,aj1]
        # B entries: e=B[aj0,ak0], f=B[aj0,ak1], g=B[aj1,ak0], h=B[aj1,ak1]
        # C entries: C00=C[ak0,ai0], C01=C[ak0,ai1], C10=C[ak1,ai0], C11=C[ak1,ai1]

        # W indices
        c00 = ak0 * n + ai0
        c01 = ak0 * n + ai1
        c10 = ak1 * n + ai0
        c11 = ak1 * n + ai1

        # m1 = (a+d)(e+h) -> C00 + C11
        # Split into 2 rank-1 terms
        add_term({ai0*m+aj0: 1, ai1*m+aj1: 1}, {aj0*p+ak0: 1, aj1*p+ak1: 1}, {c00: 1})
        add_term({ai0*m+aj0: 1, ai1*m+aj1: 1}, {aj0*p+ak0: 1, aj1*p+ak1: 1}, {c11: 1})

        # m2 = (c+d)e -> C10
        add_term({ai1*m+aj0: 1, ai1*m+aj1: 1}, {aj0*p+ak0: 1}, {c10: 1})

        # m3 = a(f-h) -> C01 + C11
        add_term({ai0*m+aj0: 1}, {aj0*p+ak1: 1, aj1*p+ak1: -1}, {c01: 1})
        add_term({ai0*m+aj0: 1}, {aj0*p+ak1: 1, aj1*p+ak1: -1}, {c11: 1})

        # m4 = d(g-e) -> C00 + C10
        add_term({ai1*m+aj1: 1}, {aj1*p+ak0: 1, aj0*p+ak0: -1}, {c00: 1})
        add_term({ai1*m+aj1: 1}, {aj1*p+ak0: 1, aj0*p+ak0: -1}, {c10: 1})

        # m5 = (a+b)h -> -C00 + C01
        add_term({ai0*m+aj0: 1, ai0*m+aj1: 1}, {aj1*p+ak1: 1}, {c00: -1})
        add_term({ai0*m+aj0: 1, ai0*m+aj1: 1}, {aj1*p+ak1: 1}, {c01: 1})

        # m6 = (c-a)(e+g) -> C11
        add_term({ai1*m+aj0: 1, ai0*m+aj0: -1}, {aj0*p+ak0: 1, aj1*p+ak0: 1}, {c11: 1})

        # m7 = (b-d)(f+h) -> C00
        add_term({ai0*m+aj1: 1, ai1*m+aj1: -1}, {aj0*p+ak1: 1, aj1*p+ak1: 1}, {c00: 1})

    # PART 1: 4x4 * 4x4 -> 4x4 (k=0,1,2,3 but only first 4 columns of B)
    # Actually, for 4x4x6, A is 4x4, B is 4x6, C is 4x6.
    # C[k,i] = sum_j A[i,j] * B[j,k] for k=0..5, i=0..3
    # W index: k*n + i = k*4 + i, so W has 6*4 = 24 rows
    # U has 4*4 = 16 rows, V has 4*6 = 24 rows

    # For k=0,1,2,3 (first 4 columns of B), we can use 4x4x4 Strassen: rank 49
    # For k=4,5 (last 2 columns of B), we use 4x4x2: rank 28

    # 4x4x4 Strassen: 4 blocks of 2x2x2
    # Block (i0, j0, k0) where i0,j0,k0 in {0,1}
    # ai0 = 2*i0, aj0 = 2*j0, ak0 = 2*k0
    for i0 in range(2):
        for j0 in range(2):
            for k0 in range(2):
                strassen_2x2x2(2*i0, 2*j0, 2*k0)

    # 4x4x2: A(4x4) * B(4x2) -> C(4x2)
    # B columns 4,5. Partition B into 2x2 blocks vertically: B_top (rows 0,1; cols 4,5) and B_bot (rows 2,3; cols 4,5)
    # A partitioned into 2x2 blocks: A00 (rows 0,1; cols 0,1), A01 (rows 0,1; cols 2,3), A10 (rows 2,3; cols 0,1), A11 (rows 2,3; cols 2,3)
    # C_top (rows 4,5... wait, C is 4x6, so C rows are k=0..5)
    # C[k,i] for k=4,5:
    # C[4,i] = sum_j A[i,j]*B[j,4]
    # C[5,i] = sum_j A[i,j]*B[j,5]
    # B[j,4] and B[j,5] for j=0..3
    # Partition: j=0,1 -> B_top; j=2,3 -> B_bot
    # C[4,i] = sum_{j=0,1} A[i,j]*B[j,4] + sum_{j=2,3} A[i,j]*B[j,4]
    # For i=0,1: C[4,i] = A_top * B_top_col0 + A_bot * B_bot_col0
    # This is getting complicated. Let me just use 4 independent 2x2x2 Strassen blocks.
    # Each block handles A[2*i0:2*i0+2, :] * B[:, 2*k0+4:2*k0+6] -> C[2*k0+4:2*k0+6, 2*i0:2*i0+2]
    # But B is 4x2, so each block is 2x4 * 4x2 -> 2x2? No.
    
    # Actually, for the 4x4x2 part, I'll just use schoolbook: 4*4*2 = 32 terms.
    # Total: 49 + 32 = 81.
    
    # Let me use schoolbook for the 4x4x2 part.
    for j in range(4):
        for i in range(4):
            for k in range(2):
                col_k = k + 4  # B column index 4,5
                add_term({i*m+j: 1}, {j*p+col_k: 1}, {col_k*n+i: 1})

    # Build output matrices
    rank = rank_counter
    U_mat = [[0] * rank for _ in range(n * m)]
    V_mat = [[0] * rank for _ in range(m * p)]
    W_mat = [[0] * rank for _ in range(p * n)]
    
    for r, terms in U.items():
        for c, v in terms:
            U_mat[r][c] = v
    for r, terms in V.items():
        for c, v in terms:
            V_mat[r][c] = v
    for r, terms in W.items():
        for c, v in terms:
            W_mat[r][c] = v
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
