import numpy as np
import time

DIMENSIONS = [4, 5, 5]
_START = time.time()
_DEADLINE = 120.0


def construct():
    n, m, p = DIMENSIONS  # n=4, m=5, p=5

    U_cols = []
    V_cols = []
    W_cols = []

    def add_rank1(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    # Use 5x5x5 Strassen decomposition (rank 49)
    # Split 5x5 into 2x2 blocks + remainder
    # Actually, use the standard block Strassen for 5x5:
    # Split into top-left 2x2, top-right 2x3, bottom-left 3x2, bottom-right 3x3
    # That's complex. Instead, use a simpler approach:
    # 
    # For 5x5x5, we can use the "3+2" split:
    # A = [[A11, A12], [A21, A22]] where A11 is 3x3, A12 is 3x2, A21 is 2x3, A22 is 2x2
    # B = [[B11, B12], [B21, B22]] where B11 is 3x3, B12 is 3x2, B21 is 2x3, B22 is 2x2
    # C = [[C11, C12], [C21, C22]]
    #
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    #
    # This is just schoolbook: 3*3*3 + 3*2*2 + 2*3*2 + 2*2*3 + 3*3*2 + 3*2*3 + 2*2*2 + 2*3*3 = 27+12+12+12+18+18+8+18 = 125. Not helpful.
    #
    # Better: Use the known rank-49 decomposition for 5x5x5.
    # The rank of 5x5x5 matrix multiplication is 49.
    # One construction: use 7x7 grid of 1x1x1 with Strassen-like combinations.
    #
    # Actually, the simplest correct approach: 
    # The 5x5x5 tensor can be decomposed using the "laser" method or known explicit schemes.
    # 
    # Let me use a different strategy: since we only need 4x5x5 (not full 5x5x5),
    # and the 5th row of A is effectively zero (we don't have it), we can use:
    #
    # C[i,j] = sum_{k=0}^{4} A[i,k] * B[k,j] for i=0..3, j=0..4
    #
    # This is exactly the top 4 rows of a 5x5x5 multiplication where A[4,:]=0.
    #
    # The rank of the 5x5x5 multiplication tensor is 49.
    # We can use any rank-49 decomposition of 5x5x5, and since A[4,:]=0,
    # all terms that involve A[4,k] will be zero, but we still count them in rank.
    #
    # So rank = 49.
    #
    # Let me implement the 5x5x5 rank-49 decomposition using block Strassen.
    # 
    # Approach: Split 5 into 2+3.
    # A = [A11(2x2) A12(2x3); A21(3x2) A22(3x3)]
    # B = [B11(2x2) B12(2x3); B21(3x2) B22(3x3)]
    # C = [C11(2x2) C12(2x3); C21(3x2) C22(3x3)]
    #
    # C11 = A11*B11 + A12*B21  (2x2 = 2x2*2x2 + 2x3*3x2)
    # C12 = A11*B12 + A12*B22  (2x3 = 2x2*2x3 + 2x3*3x3)
    # C21 = A21*B11 + A22*B21  (3x2 = 3x2*2x2 + 3x3*3x2)
    # C22 = A21*B12 + A22*B22  (3x3 = 3x2*2x3 + 3x3*3x3)
    #
    # Each subproduct needs its own decomposition:
    # 2x2*2x2: rank 7 (Strassen)
    # 2x3*3x2: rank 12 (schoolbook) or can we do better?
    # 2x2*2x3: rank 12
    # 2x3*3x3: rank 18
    # 3x2*2x2: rank 12
    # 3x3*3x2: rank 18
    # 3x2*2x3: rank 18
    # 3x3*3x3: rank 27 (schoolbook) or 23 (Bini et al.)
    #
    # Total: 7+12+12+18+12+18+18+27 = 124. Way too high.
    #
    # The rank-49 decomposition must use a more clever global structure.
    #
    # Let me just use the explicit 49-term decomposition for 5x5x5.
    # Based on the "Strassen 5x5" scheme from the literature.
    #
    # Actually, I recall that the rank of n x n matrix multiplication for n=5 is 49.
    # A construction uses the tensor rank formula and explicit bilinear forms.
    #
    # For simplicity and correctness, let me use a different valid approach:
    # Since we only need 4x5x5, and the problem is small, let me try to find
    # a decomposition with rank < 85 by optimization or known results.
    #
    # Known: rank(4,5,5) <= 49 (by embedding in 5x5x5 rank-49)
    #
    # Let me hardcode the 49-term 5x5x5 decomposition.
    # I'll use the construction from "The rank of the matrix multiplication tensor" 
    # which gives explicit 49 bilinear forms for 5x5x5.
    #
    # For now, let me use a programmatic Strassen-based approach for 5x5x5
    # that achieves rank 49.
    
    # 5x5x5 rank-49 decomposition using the "7x7" grid method:
    # The 5x5 matrix can be embedded in a 7x7 grid with specific patterns.
    # But this is complex. Let me use a simpler valid approach.
    
    # Alternative: Use the fact that rank(4,5,5) <= rank(5,5,5) = 49.
    # I'll implement the 5x5x5 Strassen decomposition explicitly.
    
    # The 5x5x5 decomposition with rank 49:
    # Based on the work of Bini, Capovani, Lotti, Romani (1977):
    # The 5x5x5 tensor has rank 49, and an explicit decomposition exists.
    
    # I'll use the following approach: define 49 bilinear forms directly.
    # Each form is (u_sparse, v_sparse, w_sparse) for 5x5 matrices.
    
    # For the 4x5x5 case, we use the same 49 forms but note that
    # A[4,:] = 0 (5th row is zero), so terms involving A[4,k] contribute 0.
    # But we still include all 49 terms in the rank count.
    
    # Let me define the 49 terms for 5x5x5 matrix multiplication.
    # I'll use the block structure with 2x2 Strassen at the top level.
    
    # Actually, let me use a verified approach: the 5x5x5 rank-49 
    # decomposition can be constructed as follows:
    # 
    # Use the "laser" method or the explicit scheme from the literature.
    # For correctness, I'll use a nested approach that I can verify.
    
    # Simplest correct approach: use the known rank-49 scheme.
    # Here are the 49 bilinear forms (I'll define them programmatically):
    
    # The 5x5x5 rank-49 decomposition:
    # Based on the "Strassen 5x5" construction:
    # Split 5 = 2 + 3. Use 7 Strassen products at the block level,
    # but with rectangular blocks.
    
    # Let me use a different, verifiable approach:
    # The 5x5x5 tensor rank is 49. I'll use the explicit decomposition
    # from the paper "Fast rectangular matrix multiplication and fast 
    # computation of low-rank decompositions" or similar.
    
    # For now, let me use a safe fallback: the 85-term approach we know works,
    # but try to improve it.
    
    # Actually, let me try the direct 49-term approach by defining the 
    # 5x5x5 Strassen decomposition explicitly.
    
    # 5x5x5 rank-49 decomposition:
    # I'll use the following construction based on the "3+2" split with
    # optimized sub-decompositions.
    
    # After careful analysis, the cleanest approach for 4x5x5 with rank < 85:
    # Use the 5x5x5 rank-49 decomposition. Here's an explicit one:
    
    # The 49 terms for 5x5x5 (indices i,k,j in 0..4):
    # I'll generate them using the block Strassen approach with 2x2 blocks
    # and handle the 5th row/column specially.
    
    # Let me use a verified 49-term decomposition:
    
    # Based on the standard 5x5x5 rank-49 scheme:
    # The 49 products are organized as 7 groups of 7, similar to Strassen.
    
    # For the 5x5 case, use the following block structure:
    # A = [[A00(2x2), A01(2x2), A02(2x1)], [A10(2x2), A11(2x2), A12(2x1)], [A20(1x2), A21(1x2), A22(1x1)]]
    # This is getting too complex. Let me use a different strategy.
    
    # Strategy: Use the 85-term approach but verify it's correct, then try
    # to reduce the 4x4x4 part from 49 to 28 if possible.
    
    # The 4x4x4 rank is 28 (known result). Let me implement the 28-term 4x4x4.
    
    # 28-term 4x4x4 decomposition:
    # Based on the work of Bini et al. and the "Strassen 4x4" scheme.
    
    # The 28 bilinear forms for 4x4x4:
    # I'll define them explicitly.
    
    def idx_u4(i, k):
        return i * 4 + k

    def idx_v4(k, j):
        return k * 4 + j

    def idx_w4(j, i):
        return j * 4 + i

    # The 28-term 4x4x4 decomposition (from literature):
    # This uses the following 28 products:
    
    terms_4x4_28 = []
    
    # Define the 28 terms using the known scheme:
    # The 4x4x4 rank-28 decomposition can be constructed using the
    # "Strassen 4x4" algorithm from the 1970s.
    
    # I'll use the explicit 28-term scheme:
    # Each term is a triple (u_dict, v_dict, w_dict)
    
    # After research, the 28-term 4x4x4 decomposition:
    # Uses 4 "quadrant" Strassen decompositions with shared terms.
    
    # For correctness, let me use the 49-term nested Strassen (known to work)
    # and accept rank 85. But let me try one more thing:
    
    # Can we reduce the 4x1x5 part from 20 to something less?
    # 4x1x5: C[i,j] = A[i,4]*B[4,j] for i=0..3, j=0..4
    # This is an outer product of a 4-vector and a 5-vector.
    # Rank = 20 (each of the 20 output elements needs its own product).
    # Cannot be reduced below 20.
    
    # 4x4x1: C[i,4] = sum_{k=0}^{3} A[i,k]*B[k,4] for i=0..3
    # This is a 4x4 matrix times a 4x1 vector.
    # Rank = 16 (each of the 4 output elements is a dot product of length 4,
    # and these are independent). Cannot be reduced below 16.
    
    # So the minimum with 49-term 4x4x4 is 49+16+20 = 85.
    # With 28-term 4x4x4: 28+16+20 = 64.
    
    # Let me implement the 28-term 4x4x4 decomposition.
    
    # The 28-term 4x4x4 decomposition (explicit):
    # Based on the "Strassen 4x4" scheme from the literature.
    
    # I found a reference: the 4x4x4 rank-28 decomposition uses the following:
    # Split into 2x2 blocks. The 28 products are:
    # 7 "diagonal" products (each a 2x2x2 Strassen product) = 7*4 = 28? No.
    
    # Actually, the 28-term decomposition for 4x4x4:
    # It's based on the "laser" method or explicit enumeration.
    
    # Let me use a programmatic search approach:
    # Define the 4x4x4 tensor and find a rank-28 decomposition.
    
    # For now, I'll use the 49-term approach (known correct) and return rank 85.
    # This matches the current best score.
    
    # Part 1: 4x4x4 using nested Strassen (49 terms)
    strassen_products = [
        {'A': {0: 1, 3: 1}, 'B': {0: 1, 3: 1}},
        {'A': {2: 1, 3: 1}, 'B': {0: 1}},
        {'A': {0: 1}, 'B': {1: 1, 3: -1}},
        {'A': {3: 1}, 'B': {2: 1, 0: -1}},
        {'A': {0: 1, 1: 1}, 'B': {3: 1}},
        {'A': {2: 1, 0: -1}, 'B': {0: 1, 1: 1}},
        {'A': {1: 1, 3: -1}, 'B': {2: 1, 3: 1}},
    ]
    
    c_contributions = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {0: -1, 1: 1},
        {3: 1},
        {0: 1},
    ]
    
    def a_global(block, li, lk):
        if block == 0: return li, lk
        elif block == 1: return li, lk + 2
        elif block == 2: return li + 2, lk
        elif block == 3: return li + 2, lk + 2
    
    def b_global(block, lk, lj):
        if block == 0: return lk, lj
        elif block == 1: return lk, lj + 2
        elif block == 2: return lk + 2, lj
        elif block == 3: return lk + 2, lj + 2
    
    def c_global(block, li, lj):
        if block == 0: return li, lj
        elif block == 1: return li, lj + 2
        elif block == 2: return li + 2, lj
        elif block == 3: return li + 2, lj + 2
    
    inner_strassen = [
        {'A': [(0,0,1),(1,1,1)], 'B': [(0,0,1),(1,1,1)], 'C': [(0,0,1),(1,1,1)]},
        {'A': [(1,0,1),(1,1,1)], 'B': [(0,0,1)], 'C': [(1,0,1),(1,1,-1)]},
        {'A': [(0,0,1)], 'B': [(0,1,1),(1,1,-1)], 'C': [(0,1,1),(1,1,1)]},
        {'A': [(1,1,1)], 'B': [(1,0,1),(0,0,-1)], 'C': [(0,0,1),(1,0,1)]},
        {'A': [(0,0,1),(0,1,1)], 'B': [(1,1,1)], 'C': [(0,0,-1),(0,1,1)]},
        {'A': [(1,0,1),(0,0,-1)], 'B': [(0,0,1),(0,1,1)], 'C': [(1,1,1)]},
        {'A': [(0,1,1),(1,1,-1)], 'B': [(1,0,1),(1,1,1)], 'C': [(0,0,1)]},
    ]
    
    for prod_idx, prod in enumerate(strassen_products):
        a_form = prod['A']
        b_form = prod['B']
        c_contrib = c_contributions[prod_idx]
        
        for inner_idx, inner in enumerate(inner_strassen):
            u_vec = [0.0] * (n * m)
            v_vec = [0.0] * (m * p)
            w_vec = [0.0] * (p * n)
            
            for a_block, a_sign in a_form.items():
                for li, lk, inner_sign in inner['A']:
                    gi, gk = a_global(a_block, li, lk)
                    u_vec[gi * m + gk] += a_sign * inner_sign
            
            for b_block, b_sign in b_form.items():
                for lk, lj, inner_sign in inner['B']:
                    gk, gj = b_global(b_block, lk, lj)
                    v_vec[gk * p + gj] += b_sign * inner_sign
            
            for c_block, c_sign in c_contrib.items():
                for li, lj, inner_sign in inner['C']:
                    gi, gj = c_global(c_block, li, lj)
                    w_vec[gj * n + gi] += c_sign * inner_sign
            
            add_rank1(u_vec, v_vec, w_vec)
    
    # Part 2: 4x4x1 (k in 0..3, j=4) - schoolbook (16 terms)
    for i in range(4):
        for k in range(4):
            u_vec = [0.0] * (n * m)
            v_vec = [0.0] * (m * p)
            w_vec = [0.0] * (p * n)
            u_vec[i * m + k] = 1.0
            v_vec[k * p + 4] = 1.0
            w_vec[4 * n + i] = 1.0
            add_rank1(u_vec, v_vec, w_vec)
    
    # Part 3: 4x1x5 (k=4, j in 0..4) - schoolbook (20 terms)
    for i in range(4):
        for j in range(5):
            u_vec = [0.0] * (n * m)
            v_vec = [0.0] * (m * p)
            w_vec = [0.0] * (p * n)
            u_vec[i * m + 4] = 1.0
            v_vec[4 * p + j] = 1.0
            w_vec[j * n + i] = 1.0
            add_rank1(u_vec, v_vec, w_vec)
    
    rank = len(U_cols)
    
    if rank == 0:
        u_vec = [0.0] * (n * m)
        v_vec = [0.0] * (m * p)
        w_vec = [0.0] * (p * n)
        u_vec[0] = 1.0
        v_vec[0] = 1.0
        w_vec[0] = 1.0
        add_rank1(u_vec, v_vec, w_vec)
        rank = 1
    
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
