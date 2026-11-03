import numpy as np
import time

DIMENSIONS = [4, 4, 6]
DEADLINE_SECONDS = 140.0

def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    
    # Rank calculation:
    # 7 Strassen steps
    # Each step: 2x2 * 2x3
    # Split 2x3 into 2x2 (cols 0,1) and 2x1 (col 2)
    # 2x2 * 2x2 rank 7 (Strassen)
    # 2x2 * 2x1 rank 4 (Schoolbook: a00*b0, a01*b1, a10*b0, a11*b1)
    # Total per step: 11
    # Total rank: 7 * 11 = 77
    rank = 77
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen steps for 2x2 blocks
    # A blocks: 00, 01, 10, 11 (2x2 each)
    # B blocks: 00, 01, 10, 11 (2x3 each)
    # Note: B blocks are 2x3, so we split them into B_block_12 (2x2) and B_block_3 (2x1)
    
    steps = [
        ({'00': 1, '11': 1}, {'00': 1, '11': 1}, {'00': 1, '11': 1}),
        ({'10': 1, '11': 1}, {'00': 1}, {'10': 1, '11': -1}),
        ({'00': 1}, {'01': 1, '11': -1}, {'01': 1, '11': 1}),
        ({'11': 1}, {'10': 1, '00': -1}, {'00': 1, '10': 1}),
        ({'00': 1, '01': 1}, {'11': 1}, {'00': -1, '01': 1}),
        ({'10': 1, '00': -1}, {'00': 1, '01': 1}, {'11': 1}),
        ({'01': 1, '11': -1}, {'10': 1, '11': 1}, {'00': 1}),
    ]
    
    # Helper to get index in U (A_flat)
    # A is 4x4, blocks are 2x2.
    # Block '00' is rows 0-1, cols 0-1
    # Block '01' is rows 0-1, cols 2-3
    # Block '10' is rows 2-3, cols 0-1
    # Block '11' is rows 2-3, cols 2-3
    def get_u_row(block, i1, j1):
        if block == '00': i, j = i1, j1
        elif block == '01': i, j = i1, j1 + 2
        elif block == '10': i, j = i1 + 2, j1
        elif block == '11': i, j = i1 + 2, j1 + 2
        else: raise ValueError
        return i * m + j
        
    # Helper to get index in V (B_flat)
    # B is 4x6, blocks are 2x3.
    # Block '00' is rows 0-1, cols 0-2
    # Block '01' is rows 0-1, cols 3-5
    # Block '10' is rows 2-3, cols 0-2
    # Block '11' is rows 2-3, cols 3-5
    # We split B_block into B_block_12 (cols 0-1 of the block) and B_block_3 (col 2 of the block)
    # For V, we need to map to the full B_flat index.
    # B_flat index is j * p + k.
    # For a block (bi, bj), the local indices are j1 (0-1), k1 (0-2).
    # Global j = bi*2 + j1, global k = bj*3 + k1.
    def get_v_row(block, j1, k1):
        if block == '00': j, k = j1, k1
        elif block == '01': j, k = j1, k1 + 3
        elif block == '10': j, k = j1 + 2, k1
        elif block == '11': j, k = j1 + 2, k1 + 3
        else: raise ValueError
        return j * p + k
        
    # Helper to get index in W (C_flat)
    # C is 4x6, blocks are 2x3.
    # C_flat index is k * n + i.
    def get_w_row(block, i1, k1):
        if block == '00': i, k = i1, k1
        elif block == '01': i, k = i1, k1 + 3
        elif block == '10': i, k = i1 + 2, k1
        elif block == '11': i, k = i1 + 2, k1 + 3
        else: raise ValueError
        return k * n + i

    col = 0
    for s in range(7):
        alpha, beta, gamma = steps[s]
        # For each Strassen step, we have a 2x2 A' and 2x3 B'.
        # We split B' into B'_12 (2x2) and B'_3 (2x1).
        # We use Strassen for A' * B'_12 (rank 7).
        # We use schoolbook for A' * B'_3 (rank 2).
        
        # --- Part 1: A' * B'_12 (Strassen) ---
        # B'_12 corresponds to k1=0,1 in the block.
        # We need to define the Strassen steps for the 2x2 * 2x2 product.
        # The A' coefficients are the same as the outer Strassen steps.
        # The B' coefficients are restricted to k1=0,1.
        
        # Inner Strassen steps for 2x2 * 2x2
        inner_steps = [
            ({'00': 1, '11': 1}, {'00': 1, '11': 1}, {'00': 1, '11': 1}),
            ({'10': 1, '11': 1}, {'00': 1}, {'10': 1, '11': -1}),
            ({'00': 1}, {'01': 1, '11': -1}, {'01': 1, '11': 1}),
            ({'11': 1}, {'10': 1, '00': -1}, {'00': 1, '10': 1}),
            ({'00': 1, '01': 1}, {'11': 1}, {'00': -1, '01': 1}),
            ({'10': 1, '00': -1}, {'00': 1, '01': 1}, {'11': 1}),
            ({'01': 1, '11': -1}, {'10': 1, '11': 1}, {'00': 1}),
        ]
        
        for i1 in range(2):
            for j1 in range(2):
                # B'_12 uses k1=0,1
                for k1 in range(2):
                    # This is not quite right. The Strassen steps combine blocks.
                    # We need to iterate over the inner Strassen steps, and for each,
                    # add the contributions to U, V, W.
                    pass
        
        # Let's restructure.
        # For each outer Strassen step (s), we have a 2x2 A' and 2x3 B'.
        # We decompose A' * B' into A' * B'_12 + A' * B'_3.
        # A' * B'_12 is a 2x2 * 2x2 product. We use Strassen (rank 7).
        # A' * B'_3 is a 2x2 * 2x1 product. We use schoolbook (rank 2).
        
        # --- Part 1: A' * B'_12 (Strassen, rank 7) ---
        # The A' is defined by alpha.
        # The B'_12 is defined by beta, restricted to k1=0,1.
        # The result C'_12 is defined by gamma, restricted to k1=0,1.
        
        # We iterate over the 7 inner Strassen steps.
        for inner_s in range(7):
            i_alpha, i_beta, i_gamma = inner_steps[inner_s]
            # Note: The inner Strassen steps operate on the 2x2 blocks of A' and B'_12.
            # The "blocks" here are the scalar elements of the 2x2 matrices.
            # So we iterate over the scalar elements.
            
            # A' elements are indexed by (ai, aj) in 0..1.
            # B'_12 elements are indexed by (bj, bk) in 0..1.
            # C'_12 elements are indexed by (ci, ck) in 0..1.
            
            # The outer Strassen step defines which A' and B' blocks are used.
            # alpha is a dict of block->val.
            # beta is a dict of block->val.
            # gamma is a dict of block->val.
            
            # We need to combine the outer and inner coefficients.
            # For a given scalar element (ai, aj) in A', its coefficient in U is:
            # sum over outer blocks b_a in alpha of alpha[b_a] * (inner coefficient for (ai,aj) in i_alpha)
            # This is complex. Let's simplify by expanding the tensor.
            
            # Actually, we can just fill the U, V, W matrices directly.
            # For each outer step s, and each inner step t (0..6 for Strassen, 0..1 for schoolbook):
            # We determine the coefficients for U, V, W.
            
            pass
            
        # This nested loop is getting complicated. Let's use a simpler approach.
        # We will generate the rank-9 decomposition for 2x2 * 2x3 explicitly.
        
        # Rank 9 for 2x2 * 2x3:
        # 7 Strassen multiplications for the 2x2 * 2x2 part (cols 0,1).
        # 2 schoolbook multiplications for the 2x2 * 2x1 part (col 2).
        
        # We need to map the scalar elements of the 2x2 A' and 2x3 B' to the global indices.
        
        # Let's do it step by step.
        for i1 in range(2):
            for j1 in range(2):
                # This loop is not right. The Strassen steps combine blocks.
                pass

    # Let's restart the construction logic.
    # We will build the U, V, W matrices by iterating over the 7 outer Strassen steps.
    # For each step, we add 9 columns (7 for Strassen, 2 for schoolbook).
    
    col = 0
    for s in range(7):
        alpha, beta, gamma = steps[s]
        
        # --- Part 1: Strassen for 2x2 * 2x2 (cols 0,1 of B block) ---
        # We use the 7 inner Strassen steps.
        for inner_s in range(7):
            i_alpha, i_beta, i_gamma = inner_steps[inner_s]
            
            # For each scalar element in the 2x2 A' and 2x3 B'
            # We need to determine the coefficient in U, V, W.
            
            # A' is a 2x2 matrix. Its elements are (ai, aj) for ai,aj in 0..1.
            # The coefficient of A'[ai, aj] in the outer step s is:
            # sum over b in alpha: alpha[b] * (1 if (ai,aj) is in block b else 0)
            # But wait, the outer step s combines 2x2 blocks of the 4x4 A.
            # The "blocks" in alpha are '00', '01', '10', '11'.
            # The inner Strassen step combines scalar elements of the 2x2 block.
            
            # Let's define the mapping from (ai, aj) to the outer block.
            # The outer block is determined by i1, j1 (which block of the 4x4 matrix).
            # But the outer Strassen step s is fixed.
            # The alpha dict tells us which outer blocks are involved.
            # For a given outer block b, the local indices (ai, aj) correspond to global indices.
            
            # This is still confusing. Let's use a different strategy.
            # We will construct the U, V, W matrices for a single 2x2 * 2x3 product with rank 9.
            # Then we will map it to the global indices.
            
            pass
            
    # This is taking too long. Let's just use the schoolbook rank 12 for 2x2 * 2x3.
    # Total rank 84.
    # It is safe and correct.
    
    # Let's try to implement the rank 63 (7x9) scheme properly.
    
    # We will define a function that returns the U, V, W contributions for a 2x2 * 2x3 product.
    # The U, V, W will be of size (4, 9), (6, 9), (6, 9) locally? No.
    # U is (4, 9), V is (6, 9), W is (6, 4)? No.
    # U is (n*m, rank) = (16, 63).
    # V is (m*p, rank) = (24, 63).
    # W is (p*n, rank) = (24, 63).
    
    # Let's just use the schoolbook rank 84 for now. It's a valid baseline.
    # The improvement to 63 is significant, but I need to be sure.
    
    # Let's try to implement the 63 rank scheme.
    
    # For each of the 7 outer Strassen steps:
    # We have a 2x2 A' and 2x3 B'.
    # We decompose A' * B' into A' * B'_12 + A' * B'_3.
    
    # A' * B'_12 (2x2 * 2x2) is done with Strassen (rank 7).
    # A' * B'_3 (2x2 * 2x1) is done with schoolbook (rank 2).
    
    # We will iterate over the 7 outer steps.
    # For each step, we will add 9 columns to U, V, W.
    
    col = 0
    for s in range(7):
        alpha, beta, gamma = steps[s]
        
        # --- Part 1: Strassen for 2x2 * 2x2 (cols 0,1 of B block) ---
        # We use the 7 inner Strassen steps.
        for inner_s in range(7):
            i_alpha, i_beta, i_gamma = inner_steps[inner_s]
            
            # We need to fill U, V, W for this inner step.
            # The A' elements are (ai, aj) in 0..1.
            # The B'_12 elements are (bj, bk) in 0..1.
            # The C'_12 elements are (ci, ck) in 0..1.
            
            # The coefficient of A'[ai, aj] in U is:
            # sum over outer block b_a in alpha: alpha[b_a] * (1 if (ai,aj) is in block b_a else 0)
            # But (ai, aj) are local indices within the 2x2 block.
            # The outer block b_a is one of '00', '01', '10', '11'.
            # The local indices (ai, aj) correspond to the scalar elements of that block.
            
            # Let's iterate over the scalar elements.
            for ai in range(2):
                for aj in range(2):
                    # Determine the outer block for (ai, aj).
                    # This is not correct. The outer step s is fixed.
                    # The alpha dict tells us which outer blocks are used.
                    # For a given outer block b, the local indices (ai, aj) are 0..1.
                    
                    # We need to sum over all outer blocks b in alpha.
                    # For each b, if (ai, aj) is a valid local index for b,
                    # then the contribution is alpha[b] * (inner coefficient).
                    
                    # The inner coefficient for (ai, aj) is i_alpha.get(block_of(ai,aj), 0).
                    # But (ai, aj) are local indices. The "block" in i_alpha is '00', '01', etc.
                    # This implies that i_alpha operates on the 2x2 block of A'.
                    # So (ai, aj) correspond to the blocks '00', '01', '10', '11' of the 2x2 A'.
                    # Wait, the 2x2 A' has 4 scalar elements.
                    # The inner Strassen steps operate on these 4 scalar elements.
                    # So i_alpha is a dict mapping '00', '01', '10', '11' to coefficients.
                    # These '00', '01', etc. refer to the scalar positions in the 2x2 A'.
                    
                    # So for a given scalar position (ai, aj) in the 2x2 A',
                    # we find the corresponding key in i_alpha.
                    # If ai=0, aj=0, key is '00'.
                    # If ai=0, aj=1, key is '01'.
                    # If ai=1, aj=0, key is '10'.
                    # If ai=1, aj=1, key is '11'.
                    
                    inner_key = f"{ai}{aj}"
                    inner_coeff = i_alpha.get(inner_key, 0)
                    
                    if inner_coeff != 0:
                        # We need to add to U for all outer blocks b in alpha.
                        # For each b, the global index is get_u_row(b, ai, aj).
                        # The coefficient is alpha[b] * inner_coeff.
                        for b, val in alpha.items():
                            U[get_u_row(b, ai, aj), col] += val * inner_coeff
            
            # Similarly for V and W.
            for bj in range(2):
                for bk in range(2): # bk is 0,1 for B'_12
                    inner_key = f"{bj}{bk}"
                    inner_coeff = i_beta.get(inner_key, 0)
                    if inner_coeff != 0:
                        for b, val in beta.items():
                            V[get_v_row(b, bj, bk), col] += val * inner_coeff
            
            for ci in range(2):
                for ck in range(2): # ck is 0,1 for C'_12
                    inner_key = f"{ci}{ck}"
                    inner_coeff = i_gamma.get(inner_key, 0)
                    if inner_coeff != 0:
                        for b, val in gamma.items():
                            W[get_w_row(b, ci, ck), col] += val * inner_coeff
            
            col += 1
        
        # --- Part 2: Schoolbook for 2x2 * 2x1 (col 2 of B block) ---
        # The B block is 2x3. We used cols 0,1 for Strassen.
        # Now we handle col 2 (local index k1=2).
        # B'_3 is a 2x1 vector: [B[0,2], B[1,2]]^T (local indices j=0,1; k=2)
        # A' is 2x2.
        # C'_3 is a 2x1 vector: [C[0,2], C[1,2]]^T (local indices i=0,1; k=2)
        
        # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
        # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
        
        # We need 4 scalar multiplications to form this sum exactly in the tensor decomposition sense.
        # Rank 4 contribution.
        
        # Mult 1: A[0,0] * B[0,2] -> contributes to C[0,2]
        # U: A[0,0], V: B[0,2], W: C[0,2]
        for b, val in alpha.items():
            U[get_u_row(b, 0, 0), col] += val
        for b, val in beta.items():
            V[get_v_row(b, 0, 2), col] += val
        for b, val in gamma.items():
            W[get_w_row(b, 0, 2), col] += val
        col += 1
        
        # Mult 2: A[0,1] * B[1,2] -> contributes to C[0,2]
        # U: A[0,1], V: B[1,2], W: C[0,2]
        for b, val in alpha.items():
            U[get_u_row(b, 0, 1), col] += val
        for b, val in beta.items():
            V[get_v_row(b, 1, 2), col] += val
        for b, val in gamma.items():
            W[get_w_row(b, 0, 2), col] += val
        col += 1
        
        # Mult 3: A[1,0] * B[0,2] -> contributes to C[1,2]
        # U: A[1,0], V: B[0,2], W: C[1,2]
        for b, val in alpha.items():
            U[get_u_row(b, 1, 0), col] += val
        for b, val in beta.items():
            V[get_v_row(b, 0, 2), col] += val
        for b, val in gamma.items():
            W[get_w_row(b, 1, 2), col] += val
        col += 1
        
        # Mult 4: A[1,1] * B[1,2] -> contributes to C[1,2]
        # U: A[1,1], V: B[1,2], W: C[1,2]
        for b, val in alpha.items():
            U[get_u_row(b, 1, 1), col] += val
        for b, val in beta.items():
            V[get_v_row(b, 1, 2), col] += val
        for b, val in gamma.items():
            W[get_w_row(b, 1, 2), col] += val
        col += 1
            
    # Total rank should be 7 * (7 + 4) = 77.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
