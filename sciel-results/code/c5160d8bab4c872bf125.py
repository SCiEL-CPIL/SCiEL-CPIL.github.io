"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def _strassen_2x2_terms():
    """Return the 7 Strassen terms for 2x2*2x2.
    Each term is a triple (u, v, w) where u,v,w are 4-element lists
    representing the factor contributions.
    u maps to A (n*m = 4), v to B (m*p = 4), w to C (p*n = 4)
    using index convention: u[i*m+j], v[j*p+k], w[k*n+i] with m=n=p=2.
    """
    # Indices for 2x2: A[i,j], B[j,k], C[k,i]
    # u: [A00, A01, A10, A11] (i*2+j)
    # v: [B00, B01, B10, B11] (j*2+k)
    # w: [C00, C01, C10, C11] (k*2+i)
    
    # M1 = (A00+A11)(B00+B11)
    u1 = [1, 0, 1, 1]
    v1 = [1, 0, 1, 1]
    w1 = [1, 0, 0, 1]
    
    # M2 = (A10+A11)B00
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    w2 = [0, 0, 1, 0]
    
    # M3 = A00(B10+B11)
    u3 = [1, 0, 0, 0]
    v3 = [0, 0, 1, 1]
    w3 = [0, 1, 0, 0]
    
    # M4 = A11(B01-B10)
    u4 = [0, 0, 0, 1]
    v4 = [0, 1, 1, 0]  # B01 - B10 -> [0, 1, -1, 0]... wait
    # Let me be more careful.
    # B01 is at index 1, B10 is at index 2.
    # B01 - B10 = [0, 1, -1, 0]
    v4 = [0, 1, -1, 0]
    w4 = [1, 0, 0, 0]
    
    # M5 = (A00+A01)B11
    u5 = [1, 1, 0, 0]
    v5 = [0, 0, 0, 1]
    w5 = [0, 0, 0, 1]
    
    # M6 = (A10-A00)(B00+B01)
    u6 = [1, 0, -1, 0]  # A00 - A10
    v6 = [1, 1, 0, 0]
    w6 = [0, 1, 1, 0]
    
    # M7 = (A11-A10)(B10+B11)
    u7 = [0, 0, 1, -1]  # A11 - A10
    v7 = [0, 0, 1, 1]
    w7 = [1, 0, 0, 0]
    
    return [(u1, v1, w1), (u2, v2, w2), (u3, v3, w3),
            (u4, v4, w4), (u5, v5, w5), (u6, v6, w6), (u7, w7 := w7)]

def _strassen_4x4_terms():
    """Build Strassen terms for 4x4*4x4 by recursively using 2x2 Strassen.
    Returns list of 49 triples (u, v, w) for 4x4 matrices.
    u: 16 entries (i*4+j), v: 16 entries (j*4+k), w: 16 entries (k*4+i)
    """
    # For 4x4, we split into 2x2 blocks of size 2.
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # The 7 Strassen combinations for the 2x2 block matrices:
    # M1 = (A00+A11)(B00+B11) -> contributes to C00, C01, C10, C11
    # M2 = (A10+A11)B00 -> contributes to C10, C11
    # M3 = A00(B10+B11) -> contributes to C00, C01
    # M4 = A11(B01-B10) -> contributes to C01
    # M5 = (A00+A01)B11 -> contributes to C01, C11
    # M6 = (A10-A00)(B00+B01) -> contributes to C10, C11
    # M7 = (A11-A10)(B10+B11) -> contributes to C10
    
    # Each Ai, Bi, Ci is 2x2, so we need to apply 2x2 Strassen to each.
    # This gives 7 * 7 = 49 terms.
    
    # Let's define the structure more carefully.
    # For the top-level Strassen:
    # M1 = (A00+A11) * (B00+B11)
    #   A00+A11 is 2x2, B00+B11 is 2x2
    #   Their product is 2x2, which needs 7 sub-terms
    #   So M1 contributes 7 terms to C
    # Similarly for others.
    
    # Let me build this programmatically.
    
    terms = []
    
    # Helper: get 2x2 Strassen terms
    s2_terms = _strassen_2x2_terms()
    
    # For each top-level Strassen combination, we need to express
    # the sum/difference of 2x2 blocks and multiply them.
    
    # Let's define the top-level combinations:
    # Each is (coeff_A_blocks, coeff_B_blocks, coeff_C_blocks)
    # where coeff_X_blocks is a dict from block_index to coefficient
    
    # Actually, let me think of it differently.
    # For 4x4, we have 4x4 matrices. We split into quadrants.
    # A = [A00 A01; A10 A11], B = [B00 B01; B10 B11]
    # C = A*B
    
    # Strassen for 4x4:
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C01 -= M4, C10 += M4+M7, C11 += M1-M7
    # Wait, that's not right. Let me use the standard recursive formulation.
    
    # Actually, for 4x4 using 2x2 Strassen recursively:
    # We treat A, B as 2x2 block matrices with 2x2 blocks.
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10+B11)
    # M4 = A11*(B01-B10)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A11-A10)*(B10+B11)
    
    # C00 = M1 + M2 - M3 + M7
    # C01 = M4 + M5
    # C10 = M3 + M6
    # C11 = M1 - M5 + M6 + M7
    
    # Now, each Mi is a 2x2 matrix product of 2x2 matrices.
    # So we need to compute each Mi using 2x2 Strassen.
    
    # Let's build the terms.
    # For each top-level M, we have:
    # M = (sum of A blocks with coeffs) * (sum of B blocks with coeffs)
    # We expand this and use 2x2 Strassen for each pair.
    
    # This is getting complex. Let me use a different approach.
    # I'll build the 4x4 Strassen terms by directly constructing the
    # tensor decomposition.
    
    # Actually, let me just use the known fact that 4x4 matrix
    # multiplication has a rank-49 decomposition using Strassen.
    # I'll construct it by nesting 2x2 Strassen.
    
    # Let me define helper to combine terms.
    
    # For 2x2 Strassen, the 7 terms are:
    # t1 = (A00+A11)*(B00+B11)
    # t2 = (A10+A11)*B00
    # t3 = A00*(B10+B11)
    # t4 = A11*(B01-B10)
    # t5 = (A00+A01)*B11
    # t6 = (A10-A00)*(B00+B01)
    # t7 = (A11-A10)*(B10+B11)
    
    # C00 = t1 + t2 - t3 + t7
    # C01 = t4 + t5
    # C10 = t3 + t6
    # C11 = t1 - t5 + t6 + t7
    
    # Now for 4x4, we apply this at the block level.
    # A00, A01, A10, A11 are each 2x2.
    # So (A00+A11) is a 2x2 matrix, (B00+B11) is a 2x2 matrix.
    # Their product needs 7 sub-terms (2x2 Strassen).
    
    # Let me build this step by step.
    
    # First, let's define the 2x2 Strassen in terms of indices.
    # For 2x2: A[i,j], B[j,k], C[k,i] with i,j,k in {0,1}
    # u index: i*2+j
    # v index: j*2+k  
    # w index: k*2+i
    
    # s2_terms[0] = t1: (A00+A11)*(B00+B11)
    # u: A00+A11 -> [1,0,1,1] (A00=1, A01=0, A10=1, A11=1)
    # v: B00+B11 -> [1,0,1,1]
    # w: C -> C00+C11 contribution... 
    # Actually, t1 contributes to C00 and C11.
    # C00 += t1_00, C11 += t1_11
    # Where t1 = (A00+A11)*(B00+B11) is 2x2.
    # t1_00 = (A00+A11)_00 * (B00+B11)_00 + ... 
    # This is getting complicated. Let me just hardcode the 4x4 Strassen terms.
    
    # Actually, I'll use a cleaner approach: build the 4x4 decomposition
    # by treating it as a block Strassen and expanding.
    
    pass  # Will implement below

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s out of 150s budget
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    rank = n * m * p  # 112
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Use Strassen for the 4x4x4 part (k=0..3), schoolbook for k=4..6
    # Strassen 4x4x4 has rank 49
    # Schoolbook for 4x4x3 has rank 4*4*3 = 48
    # Total rank: 49 + 48 = 97
    
    # Build Strassen 4x4 terms
    # For 4x4*4x4, using recursive Strassen (2 levels of 2x2)
    # This gives 7^2 = 49 terms
    
    # Let me construct the 4x4 Strassen terms properly.
    # I'll use the block structure.
    
    def strassen_4x4():
        """Return 49 triples (u, v, w) for 4x4 matrix multiplication.
        u: 16 entries indexed by i*4+j
        v: 16 entries indexed by j*4+k
        w: 16 entries indexed by k*4+i
        """
        terms = []
        
        # 2x2 Strassen template
        # For 2x2: u[i*2+j], v[j*2+k], w[k*2+i]
        s2 = [
            # t1: (A00+A11)(B00+B11)
            ([1,0,1,1], [1,0,1,1], [1,0,0,1]),
            # t2: (A10+A11)B00
            ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
            # t3: A00(B10+B11)
            ([1,0,0,0], [0,0,1,1], [0,1,0,0]),
            # t4: A11(B01-B10)
            ([0,0,0,1], [0,1,-1,0], [1,0,0,0]),
            # t5: (A00+A01)B11
            ([1,1,0,0], [0,0,0,1], [0,0,0,1]),
            # t6: (A10-A00)(B00+B01)
            ([1,0,-1,0], [1,1,0,0], [0,1,1,0]),
            # t7: (A11-A10)(B10+B11)
            ([0,0,1,-1], [0,0,1,1], [1,0,0,0]),
        ]
        
        # For 4x4, we have 2x2 blocks.
        # Block indices: 00, 01, 10, 11
        # A block ij has elements at positions (2i+r)*(4) + (2j+c) for r,c in {0,1}
        # i.e., A[2i+r, 2j+c] for r,c in {0,1}
        
        # u index for element A[row, col] = row*4 + col
        # v index for element B[row, col] = row*4 + col
        # w index for element C[row, col] = row*4 + col
        
        # For 2x2 block (bi, bj) of A, elements are at:
        # A[2*bi+r, 2*bj+c] for r,c in {0,1}
        # u index: (2*bi+r)*4 + (2*bj+c)
        
        def a_idx(bi, bj, r, c):
            return (2*bi + r) * 4 + (2*bj + c)
        
        def b_idx(bi, bj, r, c):
            return (2*bi + r) * 4 + (2*bj + c)
        
        def c_idx(bi, bj, r, c):
            return (2*bi + r) * 4 + (2*bj + c)
        
        # Top-level Strassen combinations for 4x4:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10+B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A11-A10)(B10+B11)
        
        # C00 = M1 + M2 - M3 + M7
        # C01 = M4 + M5
        # C10 = M3 + M6
        # C11 = M1 - M5 + M6 + M7
        
        # Each M is a 2x2 product of 2x2 blocks.
        # So each M needs 7 sub-terms from 2x2 Strassen.
        
        # For each top-level M, define:
        # A_blocks: dict from block_idx to coeff
        # B_blocks: dict from block_idx to coeff
        # C_contribution: dict from C_block_idx to coeff
        
        top_m = [
            # (A_blocks, B_blocks, C_contribution)
            # M1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            # M2 = (A10+A11)B00 -> C10 += 1, C11 += 1
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): 1}),
            # M3 = A00(B10+B11) -> C00 -= 1, C10 += 1
            ({(0,0): 1}, {(1,0): 1, (1,1): 1}, {(0,0): -1, (1,0): 1}),
            # M4 = A11(B01-B10) -> C01 += 1
            ({(1,1): 1}, {(0,1): 1, (1,0): -1}, {(0,1): 1}),
            # M5 = (A00+A01)B11 -> C01 += 1, C11 -= 1
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (1,1): -1}),
            # M6 = (A10-A00)(B00+B01) -> C10 += 1, C11 += 1
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,0): 1, (1,1): 1}),
            # M7 = (A11-A10)(B10+B11) -> C00 += 1, C11 += 1
            ({(1,1): 1, (1,0): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ]
        
        for a_blocks, b_blocks, c_contrib in top_m:
            # Expand (sum ai*A_bi) * (sum bj*B_bj)
            # = sum_i sum_j ai*bj * (A_bi * B_bj)
            # Each A_bi * B_bj is a 2x2*2x2 product, needs 7 sub-terms
            
            for (bi_ai, ai_coeff) in a_blocks.items():
                for (bi_bj, bj_coeff) in b_blocks.items():
                    coeff = ai_coeff * bj_coeff
                    if coeff == 0:
                        continue
                    
                    # Now compute (A_block_bi_ai) * (B_block_bi_bj) using 2x2 Strassen
                    # A block is at position (bi_ai[0], bi_ai[1])
                    # B block is at position (bi_bj[0], bi_bj[1])
                    
                    for (u2, v2, w2) in s2:
                        u = [0] * 16
                        v = [0] * 16
                        w = [0] * 16
                        
                        # Fill u from A block
                        for idx2, val in enumerate(u2):
                            r, c = divmod(idx2, 2)
                            ai = bi_ai[0] + r
                            aj = bi_ai[1] + c
                            u[a_idx(ai, aj, 0, 0)] = coeff * val
                            # Wait, this is wrong. The 2x2 Strassen u2 already
                            # encodes the linear combination. Let me redo.
                        
                        # Better approach: for each element of the 2x2 A block,
                        # the 2x2 Strassen u2 tells us the coefficient.
                        # u2[r*2+c] is the coefficient for A[r,c] in this sub-term.
                        # A[r,c] in block (ai,aj) is at global position (2*ai+r, 2*aj+c)
                        # u global index: (2*ai+r)*4 + (2*aj+c)
                        
                        for r in range(2):
                            for c in range(2):
                                idx2 = r * 2 + c
                                val = u2[idx2]
                                if val != 0:
                                    gi = 2 * bi_ai[0] + r
                                    gj = 2 * bi_ai[1] + c
                                    u[gi * 4 + gj] = coeff * val
                        
                        # Fill v from B block
                        for r in range(2):
                            for c in range(2):
                                idx2 = r * 2 + c
                                val = v2[idx2]
                                if val != 0:
                                    gj = 2 * bi_bj[0] + r
                                    gk = 2 * bi_bj[1] + c
                                    v[gj * 4 + gk] = coeff * val
                        
                        # Fill w from C contribution
                        for (cbi, cbj), c_coeff in c_contrib.items():
                            for r in range(2):
                                for c in range(2):
                                    idx2 = r * 2 + c
                                    val = w2[idx2]
                                    if val != 0:
                                        ck = 2 * cbi + r
                                        ci = 2 * cbj + c
                                        w[ck * 4 + ci] += coeff * c_coeff * val
                        
                        terms.append((u, v, w))
        
        return terms
    
    # Get Strassen 4x4 terms
    s4_terms = strassen_4x4()
    
    # Assign to U, V, W for k=0..3 part
    for r, (u, v, w) in enumerate(s4_terms):
        for i in range(n * m):
            U[i][r] = u[i]
        for j in range(m * p):
            # Map 4x4 V index to 4x7 V index
            # 4x4: j*4+k for k in 0..3
            # 4x7: j*7+k for k in 0..6
            # For the Strassen part, only k=0..3 are used
            jj = j // 4
            kk = j % 4
            if kk < 4:
                V[jj * 7 + kk][r] = v[j]
        for k in range(p * n):
            # Map 4x4 W index to 4x7 W index
            # 4x4: k*4+i for k in 0..3, i in 0..3
            # 4x7: k*4+i for k in 0..6, i in 0..3
            # Only k=0..3 are used
            W[k][r] = w[k]
    
    # Add schoolbook for k=4..6
    r = len(s4_terms)
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
