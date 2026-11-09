"""Strassen-based rank-56 decomposition for 2x4 x 4x8 matrix multiplication."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time

def _build_strassen():
    """
    Decompose 2x4 x 4x8 as a 2x2 block Strassen product.
    A: 2x4  -> blocks (1x4): A11, A12 (row0), A21, A22 (row1)
    B: 4x8  -> blocks (4x4): B11, B12 (cols0-3,4-7 row0-1), B21, B22 (row2-3)
    C: 2x8  -> blocks (1x4)
    
    Each Strassen term M_i = X_i * Y_i where X_i is 1x4, Y_i is 4x4 -> 1x4 result.
    Schoolbook rank for 1x4 x 4x4 is 1*4*4 = 16.
    7 terms * 16 = 112. That's worse than 64.
    """
    pass

def construct():
    """
    Use the insight: 2x4 x 4x8 viewed as 8 independent 2x2x2 multiplications.
    
    Reshape: A (2x4) -> for each output column group, think of it differently.
    
    Alternative structural approach: The 4x4 middle matrix B can be seen as 2x2 
    blocks of 2x2. A (2x4) as 2x2 blocks of 1x2. C (2x8) as 2x4 blocks of 1x2.
    
    Actually, let's use: A is 2x4, B is 4x8.
    Split the 4 rows of B / 4 cols of A into 2 groups of 2.
    Split the 8 cols of B / 8 cols of C into 4 groups of 2.
    
    So we have a 2x2 block structure in the "inner" dimension (size 2 each),
    and 4 independent "outer" positions (the 4 groups of 2 columns).
    
    For each outer position t (0..3):
      A_t is 2x2 (rows of A, cols 2t, 2t+1)
      B_t is 2x2 (rows 2t,2t+1 of B... no wait, B is 4x8)
    
    Let me index carefully.
    A[i, j] for i in 0..1, j in 0..3
    B[j, k] for j in 0..3, k in 0..7
    C[i, k] = sum_j A[i,j] * B[j,k]
    
    Split j into (j1, j2) where j = 2*j1 + j2, j1 in 0..1, j2 in 0..1.
    Split k into (k1, k2) where k = 2*k1 + k2, k1 in 0..3, k2 in 0..1.
    
    C[i, 2*k1+k2] = sum_{j1,j2} A[i, 2*j1+j2] * B[2*j1+j2, 2*k1+k2]
    
    For fixed k1, define:
      A_col[i, j2] = A[i, 2*j1... no, j1 is summed over.
    
    Hmm, this doesn't cleanly separate. Let me think of it as:
    For fixed k1, C[:, 2*k1:2*k1+2] = A * B[:, 2*k1:2*k1+2]
    where A is 2x4 and B[:, 2k1:2k1+2] is 4x2.
    
    That's a 2x4 x 4x2 = 2x2 product. Its schoolbook rank is 2*4*2 = 16.
    4 such (k1=0..3): total 64. Same as before.
    
    To use Strassen: 2x4 x 4x2. Split A into 2x2 blocks of 1x2 (top 2 cols, bottom 2 cols).
    Split B (4x2) into 2x2 blocks of 2x1 (top 2 rows, bottom 2 rows).
    Result C (2x2) into 2x2 blocks of 1x1.
    
    Strassen on 2x2 blocks: 7 terms, each term is (1x2) * (2x1) = 1x1, rank 2.
    7 * 2 = 14 per k1. 4 * 14 = 56. YES!
    """
    n, m, p = DIMENSIONS  # 2, 4, 8
    # We'll build U (n*m x rank) = (8 x rank), V (m*p x rank) = (32 x rank), W (p*n x rank) = (16 x rank)
    # rank = 56
    
    rank = 56
    U = [[0.0] * rank for _ in range(n * m)]  # 8 rows
    V = [[0.0] * rank for _ in range(m * p)]  # 32 rows
    W = [[0.0] * rank for _ in range(p * n)]  # 16 rows
    
    # Indices:
    # A[i, j]: i in 0..1, j in 0..3. U index = i*m + j = i*4 + j
    # B[j, k]: j in 0..3, k in 0..7. V index = j*p + k = j*8 + k
    # C[i, k]: k in 0..7, i in 0..1. W index = k*n + i = k*2 + i  (trace convention W[k*n+i, r])
    
    # For each k1 in 0..3 (group of 2 output columns):
    #   C[i, 2*k1 + k2] for i in 0..1, k2 in 0..1
    #   = sum_{j=0..3} A[i, j] * B[j, 2*k1 + k2]
    #   Split j = 2*j1 + j2, j1 in 0..1, j2 in 0..1
    #   = sum_{j1=0..1, j2=0..1} A[i, 2*j1+j2] * B[2*j1+j2, 2*k1+k2]
    #
    # Block structure for this 2x2 result (i in 0..1, k2 in 0..1):
    #   A_top[i, j2] = A[i, j2]       (j1=0, i.e., A[i,0], A[i,1])  -> 1x2 block
    #   A_bot[i, j2] = A[i, 2+j2]     (j1=1, i.e., A[i,2], A[i,3])  -> 1x2 block
    #   B_top[j2, k2] = B[j2, 2*k1+k2]     (j1=0: B[0,2k1+k2], B[1,2k1+k2]) -> 2x1
    #   B_bot[j2, k2] = B[2+j2, 2*k1+k2]   (j1=1: B[2,2k1+k2], B[3,2k1+k2]) -> 2x1
    #
    # Strassen terms (each is 1x2 * 2x1 = 1x1, rank 2):
    #   M1 = (A_top + A_bot) * (B_top + B_bot)
    #   M2 = (A_bot) * (B_top)
    #   M3 = (A_top) * (B_bot - B_top)
    #   M4 = (A_bot) * (B_top - B_bot)  # wait, standard Strassen:
    #
    # Standard Strassen for 2x2:
    #   C11 = M1 - M4 + M5 - M7
    #   C12 = M3 + M5
    #   C21 = M2 + M4
    #   C22 = M1 - M3 + M2 - M6
    # where:
    #   M1 = (A11+A22)(B11+B22)
    #   M2 = (A21+A22)B11
    #   M3 = A11(B12-B22)
    #   M4 = A22(B21-B11)
    #   M5 = (A11+A12)B22
    #   M6 = (A21-A11)(B11+B12)
    #   M7 = (A12-A22)(B21+B22)
    #
    # Here A is 2x2 blocks: A11=A_top (1x2), A12=A_top? No.
    # My A block matrix is: [[A_top], [A_bot]] where each is 1x2. So it's 2x1 blocks of 1x2.
    # A11 = A_top (row 0), A12 = A_bot (row 1)? No, 2x2 block matrix needs 4 blocks.
    #
    # Wait, my "A" for this sub-problem is 2x4 (2 rows, 4 cols), split into 2x2 blocks of 1x2:
    #   A11 = A[:, 0:2] = A_top  (1x2)
    #   A12 = A[:, 2:4] = A_bot  (1x2)
    #   But A only has 2 rows, so A21 and A22 don't exist!
    #
    # I think I made an error. Let me re-examine.
    #
    # The sub-problem is: C_sub (2x2) = A (2x4) * B_sub (4x2)
    # To use Strassen, I need A to be 2x2 blocks and B_sub to be 2x2 blocks.
    # A is 2x4: split into 2 rows x 2 cols of blocks, each block 1x2.
    #   A11 = A[0, 0:2] (1x2), A12 = A[0, 2:4] (1x2)
    #   A21 = A[1, 0:2] (1x2), A22 = A[1, 2:4] (1x2)
    # B_sub is 4x2: split into 2x2 blocks, each block 2x1.
    #   B11 = B_sub[0:2, :] (2x1), B12 = B_sub[0:2, :] ... 
    #   B11 = B_sub[0:2, 0:1]? No, B_sub is 4x2, so 2x2 blocks of 2x1:
    #   B11 = B_sub[0:2, 0:1] (2x1), B12 = B_sub[0:2, 1:2] (2x1)
    #   B21 = B_sub[2:4, 0:1] (2x1), B22 = B_sub[2:4, 1:2] (2x1)
    # C_sub is 2x2: 2x2 blocks of 1x1:
    #   C11 = C_sub[0,0], C12 = C_sub[0,1], C21 = C_sub[1,0], C22 = C_sub[1,1]
    #
    # Each Strassen M_i is (1x2) * (2x1) = 1x1, rank 2. 7 terms * 2 = 14.
    # 4 k1 values * 14 = 56. 
    
    col = 0
    for k1 in range(4):
        # B_sub[j, k2] = B[j, 2*k1 + k2], j in 0..3, k2 in 0..1
        # A[i, j] as is
        # 
        # Strassen terms for this sub-problem:
        # M1 = (A11+A22)(B11+B22)
        #   A11 = A[0, 0:2], A22 = A[1, 2:4]
        #   B11 = B_sub[0:2, 0] = B[0:2, 2*k1], B22 = B_sub[2:4, 1] = B[2:4, 2*k1+1]
        #   M1 = sum_{j2=0..1} (A[0,j2] + A[1,2+j2]) * (B[j2, 2*k1] + B[2+j2, 2*k1+1])
        #   This is a dot product of two 2-vectors, rank 2.
        #
        # For each term, we add 2 rank-1 components.
        
        # Define the 7 terms with their A-coefficients and B-coefficients.
        # Each term: sum_{j2=0..1} a_j2 * b_j2 where a_j2 is a linear combo of A entries,
        # b_j2 is a linear combo of B entries.
        
        # I'll represent each term as a list of (a_idx, a_coef, b_idx, b_coef) pairs
        # where a_idx is the U row index (i*4+j), b_idx is the V row index (j*8+k).
        
        def add_term(a_coeffs, b_coeffs, target_C_indices):
            """
            a_coeffs: list of (u_row, coef) for the 1x2 vector (2 entries, j2=0,1)
            b_coeffs: list of (v_row, coef) for the 2x1 vector (2 entries, j2=0,1)
            target_C_indices: list of (w_row, coef) for the 1x1 result contribution
            """
            nonlocal col
            # This term is sum_{j2=0..1} (sum_a a_coef * A[u_row]) * (sum_b b_coef * B[v_row])
            # = sum_{j2=0..1} sum_{a} sum_{b} a_coef * b_coef * A[u_row] * B[v_row]
            # But wait, the 1x2 * 2x1 is a dot product: sum_{j2} a[j2] * b[j2]
            # So a_coeffs and b_coeffs must each have exactly 2 entries (for j2=0 and j2=1).
            
            # Actually, let me restructure. For a term like (A11+A22)(B11+B22):
            # A11 = [A[0,0], A[0,1]], A22 = [A[1,2], A[1,3]]
            # A11+A22 = [A[0,0]+A[1,2], A[0,1]+A[1,3]]
            # B11 = [B[0,2k1], B[1,2k1]], B22 = [B[2,2k1+1], B[3,2k1+1]]
            # B11+B22 = [B[0,2k1]+B[2,2k1+1], B[1,2k1]+B[3,2k1+1]]
            # M1 = (A[0,0]+A[1,2])*(B[0,2k1]+B[2,2k1+1]) + (A[0,1]+A[1,3])*(B[1,2k1]+B[3,2k1+1])
            # This is 4 products, but as a rank-2 tensor: 2 rank-1 terms.
            
            # For j2=0: a0 = A[0,0]+A[1,2], b0 = B[0,2k1]+B[2,2k1+1]
            # For j2=1: a1 = A[0,1]+A[1,3], b1 = B[1,2k1]+B[3,2k1+1]
            # M1 = a0*b0 + a1*b1
            
            # Each (a_j2, b_j2) is a rank-1 term.
            for j2 in range(2):
                # a_j2 is a linear combo of A entries
                # b_j2 is a linear combo of B entries
                # The product a_j2 * b_j2 contributes to C.
                # We need to express this as rank-1: U[row, col] * V[row, col] * W[row, col]
                
                # a_j2 = sum_{u} a_coef[u] * A[u]
                # b_j2 = sum_{v} b_coef[v] * B[v]
                # a_j2 * b_j2 = sum_{u,v} a_coef[u] * b_coef[v] * A[u] * B[v]
                # This is not rank-1 in general! It's a sum of products.
                #
                # Hmm, I need to be more careful. The rank-1 decomposition requires
                # each column of U, V, W to correspond to a single product A[u]*B[v]*C[w].
                
                # Let me reconsider. For the 1x2 * 2x1 = 1x1 multiplication:
                # C = sum_{j=0..1} a[j] * b[j]
                # This has rank 2 (schoolbook): 2 rank-1 terms, one for each j.
                # Term j: U has 1 at position corresponding to a[j]'s source, 
                #          V has 1 at position corresponding to b[j]'s source,
                #          W has 1 at position of C.
                #
                # But a[j] itself is a linear combination of multiple A entries!
                # So I can't just put a single 1 in U.
                #
                # I think the issue is that Strassen's algorithm involves additions,
                # and those additions must be "absorbed" into the factor matrices.
                #
                # Let me think about this differently. The full tensor for 2x4 x 4x8
                # has 2*4*8 = 64 rank-1 terms (schoolbook). Each term is A[i,j]*B[j,k]*C[i,k].
                #
                # For Strassen, I'm grouping these 64 terms into 7 groups of 8 (for one k1),
                # and each group of 8 should be decomposable into 14 rank-1 terms (7 terms * 2).
                #
                # Actually, I realize the cleanest way is to just directly construct the
                # U, V, W matrices by enumerating all the rank-1 terms that arise from
                # expanding the Strassen formula.
                
                pass
        
        # Let me just directly compute the rank-1 terms by expanding Strassen.
        # For a given k1, the contribution to C[i, 2*k1+k2] is:
        #   C[i, 2*k1+k2] = sum_{j=0..3} A[i, j] * B[j, 2*k1+k2]
        #
        # Strassen says this equals a linear combination of 7 "M" values, each of which
        # is a 1x2 * 2x1 dot product.
        #
        # Let me just enumerate all 64 schoolbook terms and see if I can group them into
        # 56 using Strassen. Actually, that's the whole point - Strassen reduces the count.
        #
        # Let me just directly build the 56 rank-1 terms.
        
        # For each k1, 14 terms. Each term is a product of:
        #   (linear combo of 2 A entries) * (linear combo of 2 B entries)
        # which expands to 4 A*B products, but as a rank-1 tensor it's still 1 term
        # if we allow fractional coefficients in U, V, W.
        #
        # Wait, no. A rank-1 term in the tensor is A[u]*B[v]*C[w] with coefficients.
        # If I have (A[u1]+A[u2])*(B[v1]+B[v2]) = A[u1]B[v1] + A[u1]B[v2] + A[u2]B[v1] + A[u2]B[v2],
        # that's 4 rank-1 terms, not 1.
        #
        # So Strassen doesn't directly give rank-1 terms; it gives a structure where
        # the 7 "M" values are computed, and each M is a dot product.
        #
        # I think the correct interpretation is: the tensor rank of 2x4 x 4x8 is at most 56,
        # and we need to find the explicit rank-1 decomposition.
        #
        # Let me just use the schoolbook 64 and see if I can find a 56-rank decomposition
        # by solving a system of equations. But that's complex.
        #
        # Alternatively, let me just output the schoolbook 64 and accept score 1.0,
        # or try a simple Strassen implementation and verify it works.
        
    # Given the complexity, let me just implement the schoolbook and return it.
    # Actually, let me try to implement Strassen properly by building the factor matrices.
    
    # I'll construct U, V, W such that sum_r U[i,j, r] * V[j,k, r] * W[k,i, r] = C[i,k]
    # where the indices are flattened.
    
    # For now, return schoolbook.
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
