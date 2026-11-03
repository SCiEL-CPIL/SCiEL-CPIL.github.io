"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Strategy: Hybrid Strassen (4x4) + Schoolbook (4x3)
    # Total Rank = 7 (Strassen) + 12 (Schoolbook for 4*3) = 19? 
    # Wait, Strassen is for 2x2. For 4x4, we can use 4 blocks of 2x2.
    # 4 blocks * 7 terms = 28 terms for the 4x4 part.
    # The remaining 4x3 part is 4*3 = 12 terms.
    # Total rank = 28 + 12 = 40.
    # This is significantly better than 112.
    
    rank = 40
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # --- Part 1: Strassen for the 4x4 block (columns 0-3 of C, rows 0-3 of A, cols 0-3 of B) ---
    # We decompose A[0:4, 0:4] * B[0:4, 0:4]
    # A is 4x4, B is 4x4.
    # We split A into 2x2 blocks: A11, A12, A21, A22 (each 2x2)
    # We split B into 2x2 blocks: B11, B12, B21, B22 (each 2x2)
    # Result C[0:4, 0:4] has blocks C11, C12, C21, C22.
    
    # Strassen formulas for one 2x2 block multiplication:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We have 4 such 2x2 block multiplications to perform for the 4x4 matrix?
    # No, the 4x4 matrix is composed of four 2x2 sub-multiplications?
    # Actually, A[0:4, 0:4] * B[0:4, 0:4] is a single 4x4 multiplication.
    # We can view it as 4 independent 2x2 multiplications if we block it?
    # No, the blocks are coupled.
    # A = [[A11, A12], [A21, A22]] where each Ai is 2x2.
    # B = [[B11, B12], [B21, B22]] where each Bj is 2x2.
    # C = [[C11, C12], [C21, C22]]
    
    # So we apply Strassen ONCE to the 4x4 multiplication by treating the 2x2 blocks as scalars?
    # No, the entries are scalars. The blocks are matrices.
    # Strassen works over any ring. So we can use Strassen to compute the 4x4 product
    # by performing 7 "block multiplications".
    # Each "block multiplication" is a 2x2 matrix multiplication.
    # Each 2x2 matrix multiplication can be done in rank 7 (Strassen) or rank 4 (Schoolbook).
    # If we use Strassen for the outer 4x4, we need 7 block multiplies.
    # If we use Schoolbook for the inner 2x2, each block multiply costs 4. Total 28.
    # If we use Strassen for the inner 2x2, each block multiply costs 7. Total 49.
    # So rank 28 is the target for the 4x4 part.
    
    # Let's construct the 28 terms for the 4x4 part using Strassen on the outside and Schoolbook on the inside.
    
    # Indices for U: i in 0..3, j in 0..3 -> index i*4+j
    # Indices for V: j in 0..3, k in 0..3 -> index j*4+k
    # Indices for W: k in 0..3, i in 0..3 -> index k*4+i
    
    # We define the 7 Strassen terms M1..M7 for the 4x4 case.
    # Each Mk is a 2x2 matrix.
    # We decompose each Mk into 4 scalar terms (Schoolbook for 2x2).
    
    # Helper to add a term to U, V, W
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    # Strassen coefficients for 2x2 blocks (A11, A12, A21, A22)
    # We map 2x2 block indices to global indices.
    # Block (bi, bj) where bi, bj in {0,1} corresponds to rows/cols 2*bi..2*bi+1.
    # Global index for A: (2*bi + r)*4 + (2*bj + c)
    # Global index for B: (2*bj + r)*4 + (2*ck + c) ? No.
    # A is 4x4. B is 4x4.
    # A[i, j]. B[j, k]. C[i, k].
    
    # Let's define the 7 block products M1..M7.
    # Each M is a 2x2 matrix.
    # We iterate through the 7 Strassen terms.
    # For each term, we define the linear combinations of A-blocks and B-blocks.
    # Then for each of the 4 scalar multiplications within that block product, we add a term.
    
    # Strassen Term Definitions (for 2x2 blocks):
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # Block indices:
    # A11: bi=0, bj=0
    # A12: bi=0, bj=1
    # A21: bi=1, bj=0
    # A22: bi=1, bj=1
    
    # B11: bi=0, bj=0 (Rows 0-1 of B, Cols 0-1) -> Wait, B is multiplied on the right.
    # B[j, k].
    # Block B11 corresponds to j in 0-1, k in 0-1.
    # Block B12 corresponds to j in 0-1, k in 2-3.
    # Block B21 corresponds to j in 2-3, k in 0-1.
    # Block B22 corresponds to j in 2-3, k in 2-3.
    
    # Let's define the contributions of each Strassen term to C blocks.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We will iterate through the 7 terms.
    # For each term, we have a set of (A_block, coeff_A) and (B_block, coeff_B).
    # Actually, it's simpler to just list the scalar multiplications.
    
    # Term 1: M1 = (A11+A22)(B11+B22)
    # This expands to 4 scalar multiplications for each block pair?
    # No, M1 is a 2x2 matrix.
    # M1[r, c] = sum_{x,y} (A11[r,x] + A22[r,x]) * (B11[x,c] + B22[x,c])
    # = A11[r,x]B11[x,c] + A11[r,x]B22[x,c] + A22[r,x]B11[x,c] + A22[r,x]B22[x,c]
    # This is getting complicated to track coefficients.
    
    # Alternative: Just use the known rank-7 decomposition for 2x2 and embed it?
    # No, the standard efficient way is to use the "4x4 via 4x2x2" or similar.
    
    # Let's stick to the simplest valid improvement:
    # Use Strassen for 2x2 blocks, but we have 4 such blocks in the 4x4?
    # No, the 4x4 matrix product is a single entity.
    
    # Let's use the known fact: Rank(4x4) <= 4 * Rank(2x2) = 28.
    # We can achieve this by treating the 4x4 multiplication as 4 independent 2x2 multiplications
    # IF we block diagonalize? No, that's not how matrix multiplication works.
    
    # However, we CAN use Strassen's algorithm on the 4x4 matrix directly.
    # Strassen's algorithm reduces 4x4 to 7 multiplications of 2x2 matrices.
    # Each 2x2 matrix multiplication has rank 4 (schoolbook) or 7 (strassen).
    # If we use schoolbook for the 2x2 sub-multiplications, the total rank is 7 * 4 = 28.
    
    # So we need to generate the 28 terms.
    
    # Let's define the 7 Strassen intermediate matrices M1..M7.
    # Each Mi is 2x2.
    # We compute Mi using schoolbook (4 terms each).
    # Then we combine them to form C.
    
    # We need to map the scalar products in the computation of Mi to the global U, V, W.
    
    # Let's write a helper to add a scalar product term.
    # A term is defined by:
    # u_idx: index in A (0..15)
    # v_idx: index in B (0..27) -- Wait, B is 4x7.
    # For the 4x4 part, B is 4x4 (cols 0..3).
    # v_idx for 4x4 part: j*4 + k where j in 0..3, k in 0..3.
    # w_idx: index in C (0..15) for the 4x4 part.
    # w_idx: k*4 + i where k in 0..3, i in 0..3.
    
    # We will generate 28 terms for the 4x4 part.
    
    # Strassen Steps for 4x4:
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to decompose each Mi into scalar products.
    # Since we are using schoolbook for the 2x2 blocks, each Mi involves 4 scalar multiplications.
    # Total 7 * 4 = 28 scalar multiplications.
    
    # Let's iterate through the 7 terms.
    
    # Helper to get global index for A block (bi, bj) and local (r, c)
    def a_idx(bi, bj, r, c):
        return (2*bi + r)*4 + (2*bj + c)
        
    def b_idx(bi, bj, r, c):
        # B is 4x4 in this part.
        # Block (bi, bj) -> rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
        # Global index: row * 4 + col
        return (2*bi + r)*4 + (2*bj + c)
        
    def c_idx(bi, bj, r, c):
        # C is 4x4.
        # Block (bi, bj) -> rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
        # Global index: col * 4 + row (Trace convention W[k*n+i])
        # k is col, i is row.
        return (2*bj + c)*4 + (2*bi + r)
    
    # We define the 7 terms as lists of (a_block, b_block, coeff)
    # Actually, it's easier to just define the linear combinations for each scalar product.
    
    # Let's manually unroll the 28 terms.
    
    # Term 1: M1 = (A11+A22)(B11+B22)
    # M1 is 2x2.
    # M1[r,c] = sum_x (A11[r,x] + A22[r,x]) (B11[x,c] + B22[x,c])
    # We need to add terms for each r, c, x.
    # This creates 4 scalar products for M1?
    # No, M1 is a matrix. The "multiplication" is the dot product.
    # So for each r, c, we sum over x.
    # The term in the tensor decomposition is a product of a linear form in A and a linear form in B.
    # The linear form in A for M1[r,c] is sum_x (A11[r,x] + A22[r,x]).
    # The linear form in B for M1[r,c] is sum_x (B11[x,c] + B22[x,c]).
    # So we add ONE term for each r, c in the 2x2 block?
    # Yes, because the sum over x is internal to the linear forms.
    
    # So for M1, we add 4 terms (for r=0,1 and c=0,1).
    # Term 1.1: r=0, c=0
    # U: A11[0,0], A11[0,1], A22[0,0], A22[0,1]
    # V: B11[0,0], B11[1,0], B22[0,0], B22[1,0]
    # W: C11[0,0]
    
    # Let's generalize.
    # For each Strassen term M, and for each local position (r, c) in the 2x2 block:
    # We identify the linear combination of A entries and B entries.
    
    # Strassen Term 1: (A11+A22)(B11+B22)
    # A parts: A11, A22 (coeff 1)
    # B parts: B11, B22 (coeff 1)
    
    # Strassen Term 2: (A21+A22)B11
    # A parts: A21, A22 (coeff 1)
    # B parts: B11 (coeff 1)
    
    # Strassen Term 3: A11(B12-B22)
    # A parts: A11 (coeff 1)
    # B parts: B12 (coeff 1), B22 (coeff -1)
    
    # Strassen Term 4: A22(B21-B11)
    # A parts: A22 (coeff 1)
    # B parts: B21 (coeff 1), B11 (coeff -1)
    
    # Strassen Term 5: (A11+A12)B22
    # A parts: A11, A12 (coeff 1)
    # B parts: B22 (coeff 1)
    
    # Strassen Term 6: (A21-A11)(B11+B12)
    # A parts: A21 (coeff 1), A11 (coeff -1)
    # B parts: B11, B12 (coeff 1)
    
    # Strassen Term 7: (A12-A22)(B21+B22)
    # A parts: A12 (coeff 1), A22 (coeff -1)
    # B parts: B21, B22 (coeff 1)
    
    # We iterate through these 7 terms.
    # For each term, we iterate r in 0..1, c in 0..1.
    # We construct the U, V, W vectors for that scalar product.
    
    terms_def = [
        # (a_blocks: [(bi, bj, coeff)], b_blocks: [(bi, bj, coeff)])
        # Note: Block indices (bi, bj) refer to the 2x2 sub-blocks.
        # A11: (0,0), A12: (0,1), A21: (1,0), A22: (1,1)
        # B11: (0,0), B12: (0,1), B21: (1,0), B22: (1,1)
        
        # M1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2
        ([(1,0,1), (1,1,1)], [(0,0,1)]),
        # M3
        ([(0,0,1)], [(0,1,1), (1,1,-1)]),
        # M4
        ([(1,1,1)], [(1,0,1), (0,0,-1)]),
        # M5
        ([(0,0,1), (0,1,1)], [(1,1,1)]),
        # M6
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),
        # M7
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)])
    ]
    
    # Which C block does each M contribute to?
    # C11: M1(+), M4(+), M5(-), M7(+)
    # C12: M3(+), M5(+)
    # C21: M2(+), M4(+)
    # C22: M1(+), M2(-), M3(+), M6(+)
    
    c_block_contribs = [
        [(0,0,1), (1,0,1), (1,1,-1), (0,1,1)], # M1 -> C11(+), C21(+), C22(-), C12(+)
        [(1,0,1), (1,1,-1)],                   # M2 -> C21(+), C22(-)
        [(0,1,1), (1,1,1)],                    # M3 -> C12(+), C22(+)
        [(0,0,1), (1,0,1)],                    # M4 -> C11(+), C21(+)
        [(0,0,-1), (0,1,1)],                   # M5 -> C11(-), C12(+)
        [(1,1,1)],                             # M6 -> C22(+)
        [(0,0,1)]                              # M7 -> C11(+)
    ]
    
    for t_idx, (a_blocks, b_blocks) in enumerate(terms_def):
        # For each local r, c in 0..1
        for r in range(2):
            for c in range(2):
                # Initialize U, V, W for this term
                u_vec = {}
                v_vec = {}
                w_vec = {}
                
                # Populate U
                for bi, bj, coeff in a_blocks:
                    for xr in range(2):
                        for xc in range(2):
                            # A block (bi, bj), local (xr, xc)
                            # Wait, the dot product is over x.
                            # M[r,c] = sum_x A[r,x] B[x,c]
                            # So for a fixed r, c, we sum over x.
                            # The term in U is A[r,x].
                            # The term in V is B[x,c].
                            # So we need to be careful.
                            
                            # Actually, the linear form in A for M[r,c] is sum_x A[r,x].
                            # The linear form in B for M[r,c] is sum_x B[x,c].
                            
                            # So for each x, we add a product A[r,x] * B[x,c].
                            # But we are combining multiple blocks.
                            # U[r,x] gets coeff_A.
                            # V[x,c] gets coeff_B.
                            
                            pass # This logic is inside the x loop below.
                
                # Correct Logic:
                # For a fixed Strassen term t, and fixed local r, c:
                # We sum over x in 0..1.
                # For each x, we have a product of a linear form in A and a linear form in B.
                # But the linear forms are sums over blocks.
                # L_A = sum_{(bi,bj,ca) in a_blocks} ca * A_block(bi,bj)[r,x]
                # L_B = sum_{(bi,bj,cb) in b_blocks} cb * B_block(bi,bj)[x,c]
                
                # So the term is L_A * L_B.
                # This expands to sum_{a_block} sum_{b_block} ca*cb * A_block[r,x] * B_block[x,c].
                
                # We can add ONE term for each (x, a_block, b_block) combination?
                # No, that would be 2*2*2 = 8 terms per (r,c).
                # 7 terms * 4 (r,c) * 8 = 224 terms. Too many.
                
                # We should add ONE term for each (r,c,x) combination, where the U and V vectors
                # contain the linear combinations.
                
                # So for fixed r, c, x:
                # U vector has entries for all A[r,x] in the involved blocks.
                # V vector has entries for all B[x,c] in the involved blocks.
                
                # Let's do that.
                for x in range(2):
                    u_vals = []
                    v_vals = []
                    
                    for bi, bj, coeff in a_blocks:
                        idx = a_idx(bi, bj, r, x)
                        u_vals.append((idx, coeff))
                        
                    for bi, bj, coeff in b_blocks:
                        idx = b_idx(bi, bj, x, c)
                        v_vals.append((idx, coeff))
                        
                    # Now we have u_vals and v_vals.
                    # We add a term to U, V, W.
                    # W depends on which C blocks this M contributes to.
                    
                    for cb_i, cb_j, c_coeff in c_block_contribs[t_idx]:
                        w_idx = c_idx(cb_i, cb_j, r, c)
                        
                        # Add term
                        U_term = {idx: val for idx, val in u_vals}
                        V_term = {idx: val for idx, val in v_vals}
                        
                        # Combine W contributions?
                        # No, each (r,c,x) and each C block contribution is a separate scalar product?
                        # No, the product L_A * L_B is a single scalar.
                        # It contributes to multiple C entries.
                        # So we add the SAME U and V term, but with different W entries?
                        # No, W is a vector. We can put multiple 1s in W?
                        # No, W is a column vector.
                        # The tensor decomposition is sum_k U[:,k] V[:,k] W[:,k]^T.
                        # If L_A L_B contributes to C11 and C21, we need two terms?
                        # Or one term where W has non-zero entries in both C11 and C21?
                        # Yes! W can have multiple non-zero entries.
                        
                        # So for this (r,c,x), we add ONE term.
                        # U and V are fixed by a_blocks and b_blocks.
                        # W has non-zero entries for all C blocks that this M contributes to.
                        
                        w_vals = []
                        for cb_i, cb_j, c_coeff in c_block_contribs[t_idx]:
                            w_idx = c_idx(cb_i, cb_j, r, c)
                            w_vals.append((w_idx, c_coeff))
                            
                        # Add the term
                        for idx, val in u_vals:
                            U[idx][col] += val
                        for idx, val in v_vals:
                            V[idx][col] += val
                        for idx, val in w_vals:
                            W[idx][col] += val
                            
                        col += 1
                        break # Only add once per (r,c,x), not per C block. The W vector handles the distribution.
                        
    # Part 2: Schoolbook for the 4x3 part (cols 4-6 of C)
    # A is 4x4, B is 4x3 (cols 4-6 of original B).
    # C is 4x3.
    # Rank = 4 * 3 = 12.
    
    for i in range(4):
        for j in range(4):
            for k in range(3):
                u_idx = i*4 + j
                # B index: j in 0..3, k in 4..6
                v_idx = j*7 + (k+4)
                # W index: k in 4..6, i in 0..3
                w_idx = (k+4)*4 + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    # Verify col matches rank
    assert col == rank, f"Column count {col} does not match rank {rank}"
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    if time.time() < deadline:
        pass
        
    return best_result
# EVOLVE-BLOCK-END
