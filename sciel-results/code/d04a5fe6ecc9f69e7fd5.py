"""Strassen-based tensor decomposition for 6x6 matrix multiplication."""
import itertools

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    assert n == m == p == 6

    # Strassen's 7 multiplication terms for 2x2 block matrices
    # Each term is defined by which block sub-matrices it multiplies.
    # Block indices: 0=upper-left, 1=upper-right, 2=lower-left, 3=lower-right
    # Within each 3x3 block, we use schoolbook (rank 9)
    
    # Strassen 7 terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We'll build the U, V, W matrices column by column.
    # Each Strassen term contributes a schoolbook decomposition of 3x3 * 3x3,
    # which has rank 9 (3*3*3).
    # Total rank = 7 * 9 = 63.
    
    rank = 7 * 9  # 63
    U = [[0] * rank for _ in range(n * m)]  # 36 x 63
    V = [[0] * rank for _ in range(m * p)]  # 36 x 63
    W = [[0] * rank for _ in range(p * n)]  # 36 x 63

    # Block size is 3 (since 6 = 2*3)
    bs = 3

    # Define the 7 Strassen terms with their coefficient patterns
    # Each term: list of (block_row, block_col, coeff) for U side, 
    #              (block_row, block_col, coeff) for V side
    # A blocks: A00=[0,0], A01=[0,1], A10=[1,0], A11=[1,1]
    # B blocks: B00=[0,0], B01=[0,1], B10=[1,0], B11=[1,1]
    
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11)
        ([((0,0,1), (1,1,1))], [((0,0,1), (1,1,1))]),
        # M2 = (A10+A11)B00
        ([((1,0,1), (1,1,1))], [((0,0,1))]),
        # M3 = A00(B10-B11)
        ([((0,0,1))], [((1,0,1), (1,1,-1))]),
        # M4 = A11(B01-B00)
        ([((1,1,1))], [((0,1,1), (0,0,-1))]),
        # M5 = (A00+A01)B11
        ([((0,0,1), (0,1,1))], [((1,1,1))]),
        # M6 = (A10-A00)(B00+B01)
        ([((1,0,1), (0,0,-1))], [((0,0,1), (0,1,1))]),
        # M7 = (A01-A11)(B10+B11)
        ([((0,1,1), (1,1,-1))], [((1,0,1), (1,1,1))]),
    ]

    col = 0
    for u_blocks, v_blocks in strassen_terms:
        # For each Strassen term, we decompose the product of the summed block matrices
        # using schoolbook on the 3x3 blocks.
        # We iterate over all local (i,j,k) within the 3x3 blocks
        for li in range(bs):
            for lj in range(bs):
                for lk in range(bs):
                    # For this local position, determine which global (i,j,k) pairs
                    # this column contributes to.
                    # 
                    # The U side: for each (br, bc, coeff) in u_blocks,
                    #   global row = br*bs + li, global col = bc*bs + lj
                    # The V side: for each (br, bc, coeff) in v_blocks,
                    #   global row = br*bs + lj, global col = bc*bs + lk
                    # The W side: for each global (k, i) where i comes from U row block
                    #   and k comes from V col block
                    #
                    # Actually, let me think more carefully.
                    #
                    # In schoolbook for a single 3x3*3x3 block multiplication:
                    # C[i,k] = sum_j A[i,j] B[j,k]
                    # The tensor term for (i,j,k) has U[i,j]=1, V[j,k]=1, W[k,i]=1
                    #
                    # For Strassen, each term is a product of linear combinations of blocks.
                    # U side: sum over (br, bc, coeff_u) of coeff_u * A[br*bs + :, bc*bs + :]
                    # V side: sum over (br, bc, coeff_v) of coeff_v * B[br*bs + :, bc*bs + :]
                    #
                    # For a specific local (li, lj, lk):
                    # U column entry at (global_i, global_j) where global_i = br_u*bs + li, global_j = bc_u*bs + lj
                    #   gets coeff_u for each (br_u, bc_u, coeff_u) in u_blocks
                    # V column entry at (global_j, global_k) where global_j = br_v*bs + lj, global_k = bc_v*bs + lk
                    #   gets coeff_v for each (br_v, bc_v, coeff_v) in v_blocks
                    # W column entry at (global_k, global_i) where global_k = bc_v*bs + lk, global_i = br_u*bs + li
                    #   gets coeff_u * coeff_v for each pair (from u_blocks and v_blocks)
                    
                    # Accumulate U, V, W values for this column
                    u_val = {}  # (global_row, global_col) -> value
                    v_val = {}
                    w_val = {}
                    
                    for (br_u, bc_u, coeff_u) in u_blocks[0]:
                        gr_u = br_u * bs + li
                        gc_u = bc_u * bs + lj
                        u_val[(gr_u, gc_u)] = u_val.get((gr_u, gc_u), 0) + coeff_u
                    
                    for (br_v, bc_v, coeff_v) in v_blocks[0]:
                        gr_v = br_v * bs + lj
                        gc_v = bc_v * bs + lk
                        v_val[(gr_v, gc_v)] = v_val.get((gr_v, gc_v), 0) + coeff_v
                    
                    for (br_u, bc_u, coeff_u) in u_blocks[0]:
                        for (br_v, bc_v, coeff_v) in v_blocks[0]:
                            gi = br_u * bs + li
                            gk = bc_v * bs + lk
                            w_key = (gk, gi)
                            w_val[w_key] = w_val.get(w_key, 0) + coeff_u * coeff_v
                    
                    # Set U, V, W for this column
                    for (gi, gj), val in u_val.items():
                        if val != 0:
                            U[gi * m + gj][col] = val
                    for (gj, gk), val in v_val.items():
                        if val != 0:
                            V[gj * p + gk][col] = val
                    for (gk, gi), val in w_val.items():
                        if val != 0:
                            W[gk * n + gi][col] = val
                    
                    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
