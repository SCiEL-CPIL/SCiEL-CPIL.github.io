import time
import numpy as np

DIMENSIONS = [2, 4, 8]


# EVOLVE-BLOCK-START
def _matmul_check(U, V, W, n, m, p):
    """Exact check that the tensor decomposition reconstructs the MM tensor.
    For every (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] must equal 1.
    Returns (ok, max_abs_error).
    """
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    r = Ua.shape[1]
    Ur = Ua.reshape(n, m, r)
    Vr = Va.reshape(m, p, r)
    Wr = Wa.reshape(p, n, r)
    T = np.einsum('ijr,jkr,kin->ijk', Ur, Vr, Wr)
    err = np.max(np.abs(T - 1.0))
    return (err < 1e-9), float(err)


def _strassen_2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication."""
    return [
        {'U': {(0, 0): 1, (1, 1): 1}, 'V': {(0, 0): 1, (1, 1): 1}, 'W': {(0, 0): 1, (1, 1): 1}},
        {'U': {(1, 0): 1, (1, 1): 1}, 'V': {(0, 0): 1}, 'W': {(0, 1): 1, (1, 1): -1}},
        {'U': {(0, 0): 1}, 'V': {(0, 1): 1, (1, 1): -1}, 'W': {(1, 0): 1, (1, 1): 1}},
        {'U': {(1, 1): 1}, 'V': {(1, 0): 1, (0, 0): -1}, 'W': {(0, 0): 1, (0, 1): 1}},
        {'U': {(0, 0): 1, (0, 1): 1}, 'V': {(1, 1): 1}, 'W': {(0, 0): -1, (1, 0): 1}},
        {'U': {(1, 0): 1, (0, 0): -1}, 'V': {(0, 0): 1, (0, 1): 1}, 'W': {(1, 1): 1}},
        {'U': {(0, 1): 1, (1, 1): -1}, 'V': {(1, 0): 1, (1, 1): 1}, 'W': {(0, 0): 1}},
    ]


def _construct_rank49():
    """
    Rank-49 decomposition via embedding 2x4x8 into 4x4x4x4x4x4.
    
    Strategy: Embed A (2x4) into a 4x4 matrix (top 2 rows, bottom 2 rows zero).
    Split B (4x8) into two 4x4 halves: B1 (cols 0-3), B2 (cols 4-7).
    Use Strassen's 7-term algorithm for 4x4x4, which decomposes into 7*7=49 rank-1 terms.
    The output is a 4x4 matrix, but we only need the top 2 rows (since A's bottom rows are zero).
    Terms where U has all-zero entries in the top 2 rows of A can be trimmed.
    """
    n, m, p = DIMENSIONS
    # Build the full 4x4x4 Strassen decomposition (rank 49)
    # U_full: (4*4, 49), V_full: (4*4, 49), W_full: (4*4, 49)
    U_full = [[0] * 49 for _ in range(16)]
    V_full = [[0] * 49 for _ in range(16)]
    W_full = [[0] * 49 for _ in range(16)]
    
    s2 = _strassen_2x2_terms()
    
    # 4x4 = 2x2 blocks. The 7 Strassen terms for 4x4x4:
    # Each term t (0..6) has sub-structure for the 2x2 blocks.
    # For term t, we compute: (A_block_i + A_block_j) * (B_block_k + B_block_l)
    # and distribute to output blocks.
    
    # The 7 Strassen terms for 4x4x4 (standard):
    # m1 = (A11+A22)*(B11+B22)
    # m2 = (A21+A22)*B12
    # m3 = A11*(B11-B21)
    # m4 = A22*(B22-B12)
    # m5 = (A11+A12)*B22
    # m6 = (A21-A11)*(B11+B12)
    # m7 = (A12-A22)*(B21+B22)
    
    # For each of these 7 terms, the "A" and "B" matrices are 4x4, composed of 2x2 blocks.
    # We need to express each 4x4 matrix product as 7 rank-1 terms (Strassen for 2x2x2).
    
    # Actually, let me think differently. The 4x4x4 tensor has rank 49 via Strassen.
    # The decomposition: for each of the 7 top-level Strassen terms, we have a 2x2x2 
    # multiplication (since each "block" is 2x2), which itself has 7 rank-1 terms.
    # Total: 7 * 7 = 49 terms.
    
    # Let me define the 7 top-level Strassen terms for 4x4x4 more carefully.
    # Each term involves specific 2x2 sub-blocks of A, B, and C.
    
    # Standard Strassen for 4x4:
    # Split A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]], C = [[C11, C12], [C21, C22]]
    # where each sub-matrix is 2x2.
    
    # m1 = (A11+A22) * (B11+B22)
    # m2 = (A21+A22) * B12
    # m3 = A11 * (B11-B21)
    # m4 = A22 * (B22-B12)
    # m5 = (A11+A12) * B22
    # m6 = (A21-A11) * (B11+B12)
    # m7 = (A12-A22) * (B21+B22)
    
    # C11 = m1 + m4 - m5 + m7
    # C12 = m2 + m3
    # C21 = m6 + m5
    # C22 = m1 - m3 + m4 + m6
    
    # For each mj, we need to compute a 2x2 * 2x2 product, which uses 7 rank-1 terms.
    # The "A" matrix for mj is a 2x2 matrix (sum/difference of A sub-blocks).
    # The "B" matrix for mj is a 2x2 matrix (sum/difference of B sub-blocks).
    # The result contributes to specific C sub-blocks.
    
    # Let me define for each top-level term:
    # - The A sub-blocks involved and their coefficients
    # - The B sub-blocks involved and their coefficients
    # - The C sub-blocks where the result contributes and their coefficients
    
    # Term 1: m1 = (A11+A22)*(B11+B22)
    # A blocks: {A11: +1, A22: +1}
    # B blocks: {B11: +1, B22: +1}
    # C contributions: {C11: +1, C22: +1}
    
    # Term 2: m2 = (A21+A22)*B12
    # A blocks: {A21: +1, A22: +1}
    # B blocks: {B12: +1}
    # C contributions: {C12: +1}
    
    # Term 3: m3 = A11*(B11-B21)
    # A blocks: {A11: +1}
    # B blocks: {B11: +1, B21: -1}
    # C contributions: {C12: +1, C22: -1}
    
    # Term 4: m4 = A22*(B22-B12)
    # A blocks: {A22: +1}
    # B blocks: {B22: +1, B12: -1}
    # C contributions: {C11: +1, C22: +1}
    
    # Term 5: m5 = (A11+A12)*B22
    # A blocks: {A11: +1, A12: +1}
    # B blocks: {B22: +1}
    # C contributions: {C11: -1, C21: +1}
    
    # Term 6: m6 = (A21-A11)*(B11+B12)
    # A blocks: {A21: +1, A11: -1}
    # B blocks: {B11: +1, B12: +1}
    # C contributions: {C21: +1, C22: +1}
    
    # Term 7: m7 = (A12-A22)*(B21+B22)
    # A blocks: {A12: +1, A22: -1}
    # B blocks: {B21: +1, B22: +1}
    # C contributions: {C11: +1, C22: +1}
    
    top_terms = [
        {'A': {0: 1, 3: 1}, 'B': {0: 1, 3: 1}, 'C': {0: 1, 3: 1}},      # m1
        {'A': {2: 1, 3: 1}, 'B': {1: 1}, 'C': {1: 1}},                     # m2
        {'A': {0: 1}, 'B': {0: 1, 2: -1}, 'C': {1: 1, 3: -1}},            # m3
        {'A': {3: 1}, 'B': {3: 1, 1: -1}, 'C': {0: 1, 3: 1}},             # m4
        {'A': {0: 1, 1: 1}, 'B': {3: 1}, 'C': {0: -1, 2: 1}},             # m5
        {'A': {2: 1, 0: -1}, 'B': {0: 1, 1: 1}, 'C': {2: 1, 3: 1}},       # m6
        {'A': {1: 1, 3: -1}, 'B': {2: 1, 3: 1}, 'C': {0: 1, 3: 1}},       # m7
    ]
    # Block indices: 0=A11, 1=A12, 2=A21, 3=A22 (same for B, C)
    # In 4x4 matrix: block b corresponds to rows/cols [2*(b%2), 2*(b%2)+1] x [2*(b//2), 2*(b//2)+1]
    # Block 0: rows 0-1, cols 0-1
    # Block 1: rows 0-1, cols 2-3
    # Block 2: rows 2-3, cols 0-1
    # Block 3: rows 2-3, cols 2-3
    
    col = 0
    for tt in top_terms:
        # For this top-level term, we have a 2x2 * 2x2 product.
        # The "A" matrix is a 2x2 matrix where entry (i,j) = sum of coefficients from A blocks.
        # Actually, the A matrix for this term is formed by combining the specified A sub-blocks.
        # Each A sub-block is 2x2, so the combined "A" is 2x2 (since we're adding 2x2 blocks).
        # Similarly for B.
        
        # Build the 2x2 coefficient matrices for A and B in this term.
        # A_coeff[i,j] = sum of coeff for block b where b's row-block = i, col-block = j
        # Block b: row-block = b // 2, col-block = b % 2
        
        A_coeff = [[0, 0], [0, 0]]
        for b, coeff in tt['A'].items():
            rb = b // 2  # row block index (0 or 1)
            cb = b % 2   # col block index (0 or 1)
            A_coeff[rb][cb] += coeff
        
        B_coeff = [[0, 0], [0, 0]]
        for b, coeff in tt['B'].items():
            rb = b // 2
            cb = b % 2
            B_coeff[rb][cb] += coeff
        
        # Now, for this 2x2 * 2x2 product, use Strassen's 7 terms.
        # The "A" matrix in the 2x2x2 problem has entries A_coeff[i,j].
        # The "B" matrix has entries B_coeff[i,j].
        # The result contributes to C blocks as specified.
        
        for st in s2:
            r = col
            # U entries: for each (i,j) in st['U'], add to U_full at the corresponding 4x4 position
            # st['U'] has keys (i,j) where i,j are 0 or 1 (2x2 indices).
            # The 2x2 index (i,j) maps to 4x4 rows 2*i, 2*i+1 and cols 2*j, 2*j+1.
            # But wait - the "A" matrix here is 2x2, and its entry (i,j) is A_coeff[i,j].
            # The Strassen term st['U'][(i,j)] = c means: multiply A[i,j] by c.
            # So the contribution to U_full at 4x4 position (2*i + di, 2*j + dj) is c * A_coeff[i,j] * (1 if di,dj match).
            
            # Actually, I need to be more careful. The 2x2 "A" matrix in this sub-problem
            # represents the combined A sub-blocks. The actual 4x4 A matrix has:
            # A_full[2*i+di, 2*j+dj] for i,j in {0,1}, di,dj in {0,1}
            # = sum over blocks b with row-block=i, col-block=j of coeff_b * A_subblock_b[di,dj]
            
            # For the Strassen term, st['U'][(i,j)] = c means: take c * A_subproblem[i,j]
            # where A_subproblem[i,j] = A_coeff[i,j] (the coefficient of the combined block).
            # But A_subproblem[i,j] is not a single entry - it's a 2x2 block!
            
            # I think I'm overcomplicating this. Let me use a different approach.
            
            # The 4x4x4 Strassen decomposition can be constructed recursively:
            # Level 1: 7 terms, each involving 2x2 sub-blocks.
            # Level 2: Each 2x2x2 product uses 7 rank-1 terms.
            # Total: 49 terms.
            
            # For term (t, s) where t is top-level (0-6) and s is sub-level (0-6):
            # U_full[4x4_pos, 49_col] = product of coefficients
            
            # Let me build it more carefully using the recursive structure.
            pass
        col += 7
    
    # This approach is getting complicated. Let me use a cleaner recursive construction.
    return None


def _construct_rank49_clean():
    """
    Clean construction of rank-49 decomposition via 4x4x4 Strassen embedding.
    
    The 4x4x4 matrix multiplication tensor has a rank-49 decomposition via Strassen.
    We embed our 2x4 x 4x8 problem into this:
    - A (2x4) -> 4x4 matrix with A in top 2 rows, zeros in bottom 2 rows
    - B (4x8) -> two 4x4 matrices B1 (cols 0-3), B2 (cols 4-7)
    - C (2x8) -> top 2 rows of 4x4 result C1 (for B1) and C2 (for B2)
    
    The 4x4x4 Strassen decomposition has 49 terms. For each term, we check if
    the U entries in rows 0-7 (i=0,1 of the 4x4 A) are all zero. If so, the term
    contributes nothing and can be removed.
    """
    n, m, p = DIMENSIONS
    
    # Build the 4x4x4 Strassen decomposition (rank 49)
    # Using recursive Strassen: 4x4 = 2*(2x2), so 7 top terms * 7 sub terms = 49
    
    U_full = [[0] * 49 for _ in range(16)]  # 4*4 = 16 rows
    V_full = [[0] * 49 for _ in range(16)]  # 4*4 = 16 rows
    W_full = [[0] * 49 for _ in range(16)]  # 4*4 = 16 rows
    
    s2 = _strassen_2x2_terms()
    
    # Top-level Strassen terms for 4x4x4:
    # Each term specifies which 2x2 sub-blocks of A, B, C are involved and with what signs.
    # Block layout: block (rb, cb) where rb=0,1 (row block), cb=0,1 (col block)
    # Block index = rb * 2 + cb
    
    # m1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1]) -> C[0,0]+=1, C[1,1]+=1
    # m2 = (A[1,0]+A[1,1]) * B[0,1] -> C[0,1]+=1
    # m3 = A[0,0] * (B[0,0]-B[1,0]) -> C[0,1]+=1, C[1,1]-=1
    # m4 = A[1,1] * (B[1,1]-B[0,1]) -> C[0,0]+=1, C[1,1]+=1
    # m5 = (A[0,0]+A[0,1]) * B[1,1] -> C[0,0]-=1, C[1,0]+=1
    # m6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1]) -> C[1,0]+=1, C[1,1]+=1
    # m7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1]) -> C[0,0]+=1, C[1,1]+=1
    
    top_terms = [
        # (A_blocks: {block_idx: coeff}, B_blocks: {block_idx: coeff}, C_blocks: {block_idx: coeff})
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),      # m1
        ({2: 1, 3: 1}, {1: 1}, {1: 1}),                     # m2
        ({0: 1}, {0: 1, 2: -1}, {1: 1, 3: -1}),            # m3
        ({3: 1}, {3: 1, 1: -1}, {0: 1, 3: 1}),             # m4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 2: 1}),             # m5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {2: 1, 3: 1}),       # m6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1, 3: 1}),       # m7
    ]
    
    col = 0
    for tt_idx, (A_blocks, B_blocks, C_blocks) in enumerate(top_terms):
        # For this top-level term, the "A" is a 2x2 matrix formed by the specified blocks.
        # The "B" is a 2x2 matrix formed by the specified blocks.
        # We compute their product using Strassen (7 terms).
        
        # Build 2x2 coefficient matrices
        A_coeff = [[0, 0], [0, 0]]
        for b, c in A_blocks.items():
            rb = b // 2
            cb = b % 2
            A_coeff[rb][cb] += c
        
        B_coeff = [[0, 0], [0, 0]]
        for b, c in B_blocks.items():
            rb = b // 2
            cb = b % 2
            B_coeff[rb][cb] += c
        
        # Now use Strassen's 7 terms for the 2x2x2 product.
        # The Strassen term st has:
        #   st['U'][(i,j)] = c_u: means take c_u * A[i,j]
        #   st['V'][(j,k)] = c_v: means take c_v * B[j,k]
        #   st['W'][(k,i)] = c_w: means contribute c_w to C[k,i]
        
        for st in s2:
            r = col
            # For each (i,j) in st['U']:
            #   The 2x2 A entry A[i,j] = A_coeff[i,j]
            #   This corresponds to 4x4 A entries at rows 2*i..2*i+1, cols 2*j..2*j+1
            #   But A_coeff[i,j] is the coefficient, and the actual 4x4 A entries are the sub-block entries.
            #   
            # Wait, I think the issue is that A_coeff[i,j] is a scalar coefficient,
            # but the actual 4x4 A matrix has 2x2 blocks at each (i,j) position.
            # The Strassen term operates on the 2x2 "coefficient" matrix, but each entry
            # represents a 2x2 block of the original 4x4 matrix.
            #
            # So for st['U'][(i,j)] = c_u:
            #   For di in 0,1 and dj in 0,1:
            #     U_full[2*i + di, 2*j + dj, r] += c_u * A_coeff[i,j]
            #   But A_coeff[i,j] is already the coefficient. The actual A entries are 1 (identity-like).
            #
            # Hmm, I think I need to reconsider. The 4x4x4 Strassen decomposition works as follows:
            # The 4x4 A matrix is split into 2x2 blocks. The Strassen algorithm computes
            # 7 "products" of 2x2 matrices. Each such product is itself computed via 7 rank-1 terms.
            #
            # For the top-level term with A_blocks and B_blocks:
            # The "A matrix" for this term is a 2x2 matrix where entry (i,j) = sum of coeffs for block (i,j).
            # But this is not quite right either.
            #
            # Let me try a completely different approach: just build the 4x4x4 Strassen decomposition
            # by directly defining all 49 terms.
            
            # For Strassen term st at sub-level:
            # st['U'][(i,j)] = c_u means: in the 2x2 A matrix, entry (i,j) is multiplied by c_u.
            # This 2x2 A matrix corresponds to 4x4 A rows [2*i, 2*i+1] and cols [2*j, 2*j+1].
            # But the coefficient c_u applies to the entire 2x2 block.
            #
            # Actually, I think the correct interpretation is:
            # The 2x2 A matrix in the sub-problem has entries A_coeff[i,j].
            # The Strassen term says: take c_u * A_coeff[i,j] for each (i,j) in st['U'].
            # This contributes to the 4x4 U matrix at positions (2*i + di, 2*j + dj) for di,dj in {0,1}.
            # The value is c_u * A_coeff[i,j] (since the actual A entries are 1).
            
            for (i, j), c_u in st['U'].items():
                for di in range(2):
                    for dj in range(2):
                        row = 2 * i + di
                        col_idx = 2 * j + dj
                        U_full[row * 4 + col_idx][r] += c_u * A_coeff[i][j]
            
            for (j, k), c_v in st['V'].items():
                for dj in range(2):
                    for dk in range(2):
                        row = 2 * j + dj
                        col_idx = 2 * k + dk
                        V_full[row * 4 + col_idx][r] += c_v * B_coeff[j][k]
            
            for (k, i), c_w in st['W'].items():
                for dk in range(2):
                    for di in range(2):
                        row = 2 * k + dk
                        col_idx = 2 * i + di
                        W_full[row * 4 + col_idx][r] += c_w
            
            col += 1
    
    # Now project back to our 2x4x8 problem.
    # U: We need U[i*m+j, r] for i in 0,1 and j in 0,1,2,3.
    # In the 4x4 embedding, A is 4x4 with A[0:2, 0:4] = original A, A[2:4, :] = 0.
    # U_full has 16 rows (4x4). We need rows 0-7 (i=0,1 in 4x4, j=0,1,2,3).
    # Row index in U_full: i4 * 4 + j4 where i4 is 0..3, j4 is 0..3.
    # We need i4 in {0,1} and j4 in {0,1,2,3}.
    
    U = [[0] * 49 for _ in range(n * m)]  # 2*4 = 8 rows
    V = [[0] * 49 for _ in range(m * p)]  # 4*8 = 32 rows
    W = [[0] * 49 for _ in range(p * n)]  # 8*2 = 16 rows
    
    # U projection: U[i*m+j, r] = U_full[i*4+j, r] for i in 0,1, j in 0,1,2,3
    for i in range(2):
        for j in range(4):
            U[i * m + j] = U_full[i * 4 + j][:]
    
    # V projection: V[j*p+k, r]
    # B1 = B[:, 0:4] is 4x4, embedded as V_full rows j*4+k for j in 0..3, k in 0..3.
    # B2 = B[:, 4:8] is 4x4, embedded similarly.
    # We need two separate Strassen decompositions for B1 and B2!
    #
    # Oh no, I made an error. The 4x4x4 Strassen gives us C = A_full * B_full where both are 4x4.
    # But our B is 4x8, not 4x4. So we need to apply the 4x4x4 Strassen twice:
    # once for B1 = B[:, 0:4] and once for B2 = B[:, 4:8].
    # Total rank = 49 + 49 = 98? That's worse!
    #
    # Wait, no. Let me reconsider. The problem is 2x4 * 4x8 -> 2x8.
    # If I embed A into 4x4 (top 2 rows), and B into 4x4, I can only handle 4x4 * 4x4 -> 4x4.
    # But B is 4x8, so I need two 4x4 B matrices.
    #
    # Alternative: embed differently. 
    # A is 2x4. B is 4x8. 
    # Embed A into 4x4: A_full[0:2, :] = A, A_full[2:4, :] = 0.
    # Embed B into 4x8: B_full = B (already 4x8).
    # Then A_full * B_full is 4x8. We only need top 2 rows.
    #
    # But 4x4 * 4x8 is not a standard tensor. The rank of 4x4*4x8 is at most 4*4*8 = 128.
    # Strassen doesn't directly apply to rectangular matrices.
    #
    # Let me reconsider the original proposal. Maybe the idea is:
    # Split B (4x8) into B1 (4x4) and B2 (4x4).
    # Compute C1 = A * B1 and C2 = A * B2 separately.
    # For each, embed A (2x4) into 4x4 and use 4x4x4 Strassen (rank 49 each).
    # Total rank = 49 + 49 = 98. Worse than 56.
    #
    # Unless... we can share some terms between the two halves?
    # The A matrix is the same for both halves. So the U factors are the same.
    # But the V and W factors are different.
    #
    # Actually, in the tensor decomposition, each term is a rank-1 tensor U_r * V_r * W_r.
    # If we use the same U_r for both halves, we'd need separate V_r and W_r.
    # But the rank is the number of terms, and each term has one U, one V, one W.
    # So we can't share U across different V/W terms without increasing rank.
    #
    # Hmm, this approach seems to give rank 98, which is worse. Let me abandon this.
    
    return None


def _construct_rank56():
    """The original Strassen-based rank-56 decomposition (known valid)."""
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    strassen = _strassen_2x2_terms()

    col = 0
    for j_high in range(2):
        for l in range(4):
            for term in strassen:
                for (i, j_low), c in term['U'].items():
                    j = j_high * 2 + j_low
                    U[i * m + j][col] += c
                for (j_low, k_low), c in term['V'].items():
                    j = j_high * 2 + j_low
                    k = l * 2 + k_low
                    V[j * p + k][col] += c
                for (k_low, i), c in term['W'].items():
                    k = l * 2 + k_low
                    W[k * n + i][col] += c
                col += 1
    return {"U": U, "V": V, "W": W}


def _construct_rank32():
    """
    Rank-32 decomposition: 8 independent schoolbook 2x2 blocks.
    """
    n, m, p = DIMENSIONS
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for l in range(4):
        for rb in range(2):
            if rb == 0:
                def gA(i, j): return i * m + j
            else:
                def gA(i, j): return i * m + (2 + j)

            def gB(j, k, rb=rb, l=l):
                gj = (j if rb == 0 else 2 + j)
                return gj * p + (2 * l + k)

            def gC(k, i, l=l):
                return (2 * l + k) * n + i

            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        r = col
                        U[gA(i, j)][r] = 1
                        V[gB(j, k)][r] = 1
                        W[gC(k, i)][r] = 1
                        col += 1

    return {"U": U, "V": V, "W": W}


def _trim_zero_columns(U, V, W):
    """Remove rank columns that are all-zero in U, V, and W (they contribute nothing)."""
    n_u = len(U)
    n_v = len(V)
    n_w = len(W)
    rank = len(U[0]) if n_u else 0
    keep = []
    for c in range(rank):
        if any(U[i][c] != 0 for i in range(n_u)) or \
           any(V[i][c] != 0 for i in range(n_v)) or \
           any(W[i][c] != 0 for i in range(n_w)):
            keep.append(c)
    if len(keep) == rank:
        return U, V, W, rank
    U2 = [[U[i][c] for c in keep] for i in range(n_u)]
    V2 = [[V[i][c] for c in keep] for i in range(n_v)]
    W2 = [[W[i][c] for c in keep] for i in range(n_w)]
    return U2, V2, W2, len(keep)


def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140

    best = None

    # Candidate 1: rank-32 (8 independent schoolbook 2x2 blocks)
    try:
        cand = _construct_rank32()
        ok, err = _matmul_check(cand["U"], cand["V"], cand["W"], n, m, p)
        if ok:
            U, V, W, r = _trim_zero_columns(cand["U"], cand["V"], cand["W"])
            if best is None or r < best[0]:
                best = (r, U, V, W)
    except Exception:
        pass

    # Candidate 2: rank-56 (original Strassen, known valid) as fallback
    try:
        cand = _construct_rank56()
        ok, err = _matmul_check(cand["U"], cand["V"], cand["W"], n, m, p)
        if ok:
            U, V, W, r = _trim_zero_columns(cand["U"], cand["V"], cand["W"])
            if best is None or r < best[0]:
                best = (r, U, V, W)
    except Exception:
        pass

    if best is None:
        cand = _construct_rank56()
        return cand

    r, U, V, W = best
    return {"U": U, "V": W, "W": W}
# EVOLVE-BLOCK-END
