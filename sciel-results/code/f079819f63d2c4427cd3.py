"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    # Block decomposition: 3 blocks of 2x3x2 multiplication.
    # Each 2x3x2 block has rank 11 (7 from Strassen on 2x2x2 part + 4 from 2x1x2 part).
    # Total rank 33.
    
    n, m, p = DIMENSIONS
    # We split p=6 into three chunks of size 2.
    # Each chunk requires multiplying A (2x3) by B_chunk (3x2).
    
    rank_per_block = 11
    total_rank = 3 * rank_per_block
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Helper to add a rank-11 decomposition for a 2x3x2 block
    # Block b corresponds to B columns [2*b, 2*b+1]
    for b in range(3):
        offset = b * rank_per_block
        
        # Indices for this block
        # A: rows 0..5 (2x3)
        # B: rows 0..8 (3x6), but only cols 2*b, 2*b+1 are used in this block's V part?
        # Wait, V is (m*p) x rank. The rows of V correspond to B entries.
        # B is 3x6. Rows are (j, k).
        # For block b, we use B[j, 2*b] and B[j, 2*b+1].
        # C: rows 0..11 (6x2), but only rows corresponding to this block's output?
        # W is (p*n) x rank. Rows are (k, i).
        # For block b, we produce C[i, 2*b] and C[i, 2*b+1].
        # So W rows are (2*b, i) and (2*b+1, i).
        
        # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 2*b:2*b+2]
        # A part: rows 0,1,2,3 of U (i=0,1; j=0,1)
        # B part: rows 0,1,2,3 of V (j=0,1; k=2*b, 2*b+1)
        # C part: rows (2*b, 0), (2*b, 1), (2*b+1, 0), (2*b+1, 1) of W
        
        # Strassen Factors (adapted for local indices)
        # M1: (a+d)(x+v)
        U[0][offset+0] += 1; U[1][offset+0] += 1
        V[0][offset+0] += 1; V[1][offset+0] += 1 # B[0, 2b], B[0, 2b+1]
        W[2*b*n + 0][offset+0] += 1 # C[0, 2b]
        W[2*b*n + 1][offset+0] += 1 # C[0, 2b+1]
        W[(2*b+1)*n + 0][offset+0] += 1 # C[1, 2b]
        W[(2*b+1)*n + 1][offset+0] += 1 # C[1, 2b+1]
        
        # M2: (c+d)x
        U[2][offset+1] += 1; U[3][offset+1] += 1
        V[0][offset+1] += 1 # B[0, 2b]
        W[2*b*n + 0][offset+1] += 1
        W[(2*b+1)*n + 0][offset+1] += 1
        
        # M3: a(y-v)
        U[0][offset+2] += 1
        V[1][offset+2] += 1; V[0][offset+2] += -1 # B[0, 2b+1] - B[0, 2b]
        W[2*b*n + 1][offset+2] += 1
        W[(2*b+1)*n + 1][offset+2] += 1
        
        # M4: d(u-x)
        U[3][offset+3] += 1
        V[2][offset+3] += 1; V[0][offset+3] += -1 # B[1, 2b] - B[0, 2b]
        W[2*b*n + 0][offset+3] += 1
        W[(2*b+1)*n + 0][offset+3] += 1
        
        # M5: (a+b)v
        U[0][offset+4] += 1; U[1][offset+4] += 1
        V[1][offset+4] += 1 # B[0, 2b+1]
        W[2*b*n + 0][offset+4] += -1
        W[2*b*n + 1][offset+4] += 1
        
        # M6: (c+d)(v-y)
        U[2][offset+5] += 1; U[3][offset+5] += 1
        V[1][offset+5] += 1; V[0][offset+5] += -1 # B[0, 2b+1] - B[0, 2b]
        W[2*b*n + 1][offset+5] += 1
        W[(2*b+1)*n + 1][offset+5] += 1
        
        # M7: (a-b)(x+u)
        U[0][offset+6] += 1; U[1][offset+6] += -1
        V[0][offset+6] += 1; V[2][offset+6] += 1 # B[0, 2b] + B[1, 2b]
        W[2*b*n + 0][offset+6] += 1
        W[(2*b+1)*n + 0][offset+6] += 1
        
        # Part 2: Schoolbook for A[0:2, 2] * B[2, 2*b:2*b+2]
        # A part: rows 4,5 of U (i=0,1; j=2)
        # B part: row 4,5 of V (j=2; k=2*b, 2*b+1)
        # C part: same as above
        
        # Term 8: A[0,2] * B[2, 2b]
        U[4][offset+7] += 1
        V[4][offset+7] += 1 # B[2, 2b]
        W[2*b*n + 0][offset+7] += 1
        
        # Term 9: A[0,2] * B[2, 2b+1]
        U[4][offset+8] += 1
        V[5][offset+8] += 1 # B[2, 2b+1]
        W[2*b*n + 1][offset+8] += 1
        
        # Term 10: A[1,2] * B[2, 2b]
        U[5][offset+9] += 1
        V[4][offset+9] += 1
        W[(2*b+1)*n + 0][offset+9] += 1
        
        # Term 11: A[1,2] * B[2, 2b+1]
        U[5][offset+10] += 1
        V[5][offset+10] += 1
        W[(2*b+1)*n + 1][offset+10] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
