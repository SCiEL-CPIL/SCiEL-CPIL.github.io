import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # We will build the decomposition by splitting the problem into sub-problems.
    # The strategy is:
    # 1. Split m (4) into 2+2.
    #    Part 1: n=3, m=2, p=5 (j in 0..1)
    #    Part 2: n=3, m=2, p=5 (j in 2..3)
    # 2. For each 3x2x5 part:
    #    Split n (3) into 1+2.
    #    Sub-part 1: n=1, m=2, p=5
    #    Sub-part 2: n=2, m=2, p=5
    # 3. For 1x2x5:
    #    Split p (5) into 2+2+1.
    #    Sub-sub-part 1: 1x2x2 (Rank 4)
    #    Sub-sub-part 2: 1x2x2 (Rank 4)
    #    Sub-sub-part 3: 1x2x1 (Rank 2)
    # 4. For 2x2x5:
    #    Split p (5) into 2+2+1.
    #    Sub-sub-part 1: 2x2x2 (Rank 7, Strassen)
    #    Sub-sub-part 2: 2x2x2 (Rank 7, Strassen)
    #    Sub-sub-part 3: 2x2x1 (Rank 4)
    
    # Total Rank:
    # Part 1 (3x2x5) = (1x2x5) + (2x2x5) = (4+4+2) + (7+7+4) = 10 + 18 = 28
    # Part 2 (3x2x5) = 28
    # Total = 56
    
    U = []
    V = []
    W = []
    
    def add_term(u_idx, v_idx, w_idx, val=1):
        # u_idx is the index in A (n*m)
        # v_idx is the index in B (m*p)
        # w_idx is the index in C (p*n) -- Note: W[k*n+i, r]
        # The problem states W[k*n+i, r].
        # Standard tensor decomposition: T[a,b,c] = sum U[a,r] V[b,r] W[c,r]
        # Here a = i*m+j, b = j*p+k, c = k*n+i.
        
        if not U:
            # Initialize with zeros for the current column
            pass
            
        # We will build lists of tuples (idx, val) for each column
        # Then convert to matrices
        
        pass

    # Let's use a list of lists for U, V, W where each row is a list of (rank_idx, value)
    # Or simply accumulate into matrices.
    
    # Max rank is 56.
    max_rank = 56
    U_mat = np.zeros((n*m, max_rank))
    V_mat = np.zeros((m*p, max_rank))
    W_mat = np.zeros((p*n, max_rank))
    
    rank_idx = 0
    
    def add_term_u(u_idx, val):
        nonlocal rank_idx
        U_mat[u_idx, rank_idx] += val
        
    def add_term_v(v_idx, val):
        nonlocal rank_idx
        V_mat[v_idx, rank_idx] += val
        
    def add_term_w(w_idx, val):
        nonlocal rank_idx
        W_mat[w_idx, rank_idx] += val
        
    def add_term(u_idx, v_idx, w_idx, val=1):
        add_term_u(u_idx, val)
        add_term_v(v_idx, val)
        add_term_w(w_idx, val)
        rank_idx += 1

    def idx_A(i, j):
        return i*m + j

    def idx_B(j, k):
        return j*p + k

    def idx_C(i, k):
        # W[k*n+i, r]
        return k*n + i

    # Helper for 1x2x2 (Schoolbook)
    # A is 1x2, B is 2x2, C is 1x2
    # i=0, j in 0..1, k in 0..1
    def dec_1x2x2(i0, j0, k0):
        nonlocal rank_idx
        # C[0, k] = sum_j A[0, j] B[j, k]
        # C[0, 0] = A[0,0]B[0,0] + A[0,1]B[1,0]
        # C[0, 1] = A[0,0]B[0,1] + A[0,1]B[1,1]
        # Terms:
        # 1. A[0,0] * B[0,0] * C[0,0]
        add_term(idx_A(i0, j0), idx_B(j0, k0), idx_C(i0, k0))
        # 2. A[0,1] * B[1,0] * C[0,0]
        add_term(idx_A(i0, j0+1), idx_B(j0+1, k0), idx_C(i0, k0))
        # 3. A[0,0] * B[0,1] * C[0,1]
        add_term(idx_A(i0, j0), idx_B(j0, k0+1), idx_C(i0, k0+1))
        # 4. A[0,1] * B[1,1] * C[0,1]
        add_term(idx_A(i0, j0+1), idx_B(j0+1, k0+1), idx_C(i0, k0+1))

    def dec_1x2x1(i0, j0, k0):
        nonlocal rank_idx
        # C[0, 0] = A[0,0]B[0,0] + A[0,1]B[1,0]
        add_term(idx_A(i0, j0), idx_B(j0, k0), idx_C(i0, k0))
        add_term(idx_A(i0, j0+1), idx_B(j0+1, k0), idx_C(i0, k0))

    def dec_2x2x1(i0, j0, k0):
        nonlocal rank_idx
        # C[i, 0] = sum_j A[i, j] B[j, 0]
        # C[0, 0] = A[0,0]B[0,0] + A[0,1]B[1,0]
        # C[1, 0] = A[1,0]B[0,0] + A[1,1]B[1,0]
        add_term(idx_A(i0, j0), idx_B(j0, k0), idx_C(i0, k0))
        add_term(idx_A(i0, j0+1), idx_B(j0+1, k0), idx_C(i0, k0))
        add_term(idx_A(i0+1, j0), idx_B(j0, k0), idx_C(i0+1, k0))
        add_term(idx_A(i0+1, j0+1), idx_B(j0+1, k0), idx_C(i0+1, k0))

    def dec_2x2x2_strassen(i0, j0, k0):
        nonlocal rank_idx
        # A is 2x2, B is 2x2, C is 2x2
        # Strassen's algorithm:
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Indices:
        # A11: i0, j0
        # A12: i0, j0+1
        # A21: i0+1, j0
        # A22: i0+1, j0+1
        
        # B11: j0, k0
        # B12: j0, k0+1
        # B21: j0+1, k0
        # B22: j0+1, k0+1
        
        # C11: i0, k0
        # C12: i0, k0+1
        # C21: i0+1, k0
        # C22: i0+1, k0+1
        
        # M1: (A11 + A22) * (B11 + B22)
        # This contributes to C11, C22.
        # We need to expand (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
        # But Strassen combines these.
        # Actually, Strassen terms are bilinear forms.
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # We add terms for each multiplication in these expressions.
        
        # M1:
        # (A11+A22) is a vector. (B11+B22) is a vector.
        # Product is sum of 4 terms.
        # But in tensor decomposition, each "term" is a rank-1 component.
        # The rank-1 component for M1 corresponds to the product of the linear forms.
        # U vector: 1 at A11, 1 at A22.
        # V vector: 1 at B11, 1 at B22.
        # W vector: contributes to C11 and C22.
        # C11 gets +1, C22 gets +1.
        
        # Let's implement the addition of these 7 rank-1 terms.
        
        # Term 1: M1
        u1 = [idx_A(i0, j0), idx_A(i0+1, j0+1)]
        v1 = [idx_B(j0, k0), idx_B(j0+1, k0+1)]
        w1 = [(idx_C(i0, k0), 1), (idx_C(i0+1, k0+1), 1)]
        for ui in u1:
            for vi in v1:
                for wi, wval in w1:
                    add_term(ui, vi, wi, 1)
                    
        # Term 2: M2 = (A21+A22)B11
        u2 = [idx_A(i0+1, j0), idx_A(i0+1, j0+1)]
        v2 = [idx_B(j0, k0)]
        w2 = [(idx_C(i0+1, k0), 1)]
        for ui in u2:
            for vi in v2:
                for wi, wval in w2:
                    add_term(ui, vi, wi, 1)
                    
        # Term 3: M3 = A11(B12-B22)
        u3 = [idx_A(i0, j0)]
        v3 = [idx_B(j0, k0+1), idx_B(j0+1, k0+1)]
        w3 = [(idx_C(i0, k0+1), 1), (idx_C(i0, k0), 0)] # Wait, M3 contributes to C12 and C11?
        # C12 = M3 + M5. So M3 contributes +1 to C12.
        # C11 = M1 + M4 - M5 + M7. M3 does not contribute to C11.
        # Wait, the formula is C11 = M1 + M4 - M5 + M7.
        # M3 is not in C11.
        # So M3 only contributes to C12.
        # But M3 = A11(B12-B22).
        # The term is A11 * (B12 - B22) * C12?
        # No, the tensor decomposition is sum U V W.
        # If we have a term U V W, it adds U[i]V[j]W[k] to T[i,j,k].
        # For M3, the "value" is A11(B12-B22).
        # This is a bilinear form in A and B.
        # It contributes to C12.
        # So we need a rank-1 term where U has 1 at A11, V has 1 at B12 and -1 at B22, and W has 1 at C12.
        # This is not a single rank-1 term in the standard sense if V is a vector with multiple non-zeros.
        # But U, V, W are vectors.
        # U = e_A11.
        # V = e_B12 - e_B22.
        # W = e_C12.
        # This is a valid rank-1 component in the tensor space?
        # Yes, the tensor is sum of outer products.
        # Outer product of U, V, W.
        # T[i,j,k] += U[i] V[j] W[k].
        # So if V has two non-zeros, it adds two terms to the sum for that rank index.
        # But the rank index is a single column in U, V, W.
        # So we can just set V[j0, k0+1] = 1 and V[j0+1, k0+1] = -1 in the same column.
        
        # Let's re-implement add_term to allow multiple indices for U, V, W.
        
        # I will rewrite the add_term logic to be more flexible.
        
        pass # This logic is getting complex to inline. I will refactor.
        
    # Refactoring:
    # I will create a function that takes lists of (index, value) for U, V, W and adds them to the matrices.
    
    def add_rank1_term(u_list, v_list, w_list):
        nonlocal rank_idx
        for u_idx, u_val in u_list:
            U_mat[u_idx, rank_idx] += u_val
        for v_idx, v_val in v_list:
            V_mat[v_idx, rank_idx] += v_val
        for w_idx, w_val in w_list:
            W_mat[w_idx, rank_idx] += w_val
        rank_idx += 1
        
    # Re-defining the decomposition functions using add_rank1_term
    
    def dec_1x2x2(i0, j0, k0):
        nonlocal rank_idx
        # C00 = A00B00 + A01B10
        add_rank1_term([(idx_A(i0, j0), 1)], [(idx_B(j0, k0), 1)], [(idx_C(i0, k0), 1)])
        add_rank1_term([(idx_A(i0, j0+1), 1)], [(idx_B(j0+1, k0), 1)], [(idx_C(i0, k0), 1)])
        # C01 = A00B01 + A01B11
        add_rank1_term([(idx_A(i0, j0), 1)], [(idx_B(j0, k0+1), 1)], [(idx_C(i0, k0+1), 1)])
        add_rank1_term([(idx_A(i0, j0+1), 1)], [(idx_B(j0+1, k0+1), 1)], [(idx_C(i0, k0+1), 1)])

    def dec_1x2x1(i0, j0, k0):
        nonlocal rank_idx
        add_rank1_term([(idx_A(i0, j0), 1)], [(idx_B(j0, k0), 1)], [(idx_C(i0, k0), 1)])
        add_rank1_term([(idx_A(i0, j0+1), 1)], [(idx_B(j0+1, k0), 1)], [(idx_C(i0, k0), 1)])

    def dec_2x2x1(i0, j0, k0):
        nonlocal rank_idx
        add_rank1_term([(idx_A(i0, j0), 1)], [(idx_B(j0, k0), 1)], [(idx_C(i0, k0), 1)])
        add_rank1_term([(idx_A(i0, j0+1), 1)], [(idx_B(j0+1, k0), 1)], [(idx_C(i0, k0), 1)])
        add_rank1_term([(idx_A(i0+1, j0), 1)], [(idx_B(j0, k0), 1)], [(idx_C(i0+1, k0), 1)])
        add_rank1_term([(idx_A(i0+1, j0+1), 1)], [(idx_B(j0+1, k0), 1)], [(idx_C(i0+1, k0), 1)])

    def dec_2x2x2_strassen(i0, j0, k0):
        nonlocal rank_idx
        # M1 = (A11+A22)(B11+B22) -> C11, C22
        u1 = [(idx_A(i0, j0), 1), (idx_A(i0+1, j0+1), 1)]
        v1 = [(idx_B(j0, k0), 1), (idx_B(j0+1, k0+1), 1)]
        w1 = [(idx_C(i0, k0), 1), (idx_C(i0+1, k0+1), 1)]
        add_rank1_term(u1, v1, w1)
        
        # M2 = (A21+A22)B11 -> C21
        u2 = [(idx_A(i0+1, j0), 1), (idx_A(i0+1, j0+1), 1)]
        v2 = [(idx_B(j0, k0), 1)]
        w2 = [(idx_C(i0+1, k0), 1)]
        add_rank1_term(u2, v2, w2)
        
        # M3 = A11(B12-B22) -> C12
        u3 = [(idx_A(i0, j0), 1)]
        v3 = [(idx_B(j0, k0+1), 1), (idx_B(j0+1, k0+1), -1)]
        w3 = [(idx_C(i0, k0+1), 1)]
        add_rank1_term(u3, v3, w3)
        
        # M4 = A22(B21-B11) -> C11, C21
        u4 = [(idx_A(i0+1, j0+1), 1)]
        v4 = [(idx_B(j0+1, k0), 1), (idx_B(j0, k0), -1)]
        w4 = [(idx_C(i0, k0), 1), (idx_C(i0+1, k0), 1)]
        add_rank1_term(u4, v4, w4)
        
        # M5 = (A11+A12)B22 -> C11, C12
        u5 = [(idx_A(i0, j0), 1), (idx_A(i0, j0+1), 1)]
        v5 = [(idx_B(j0+1, k0+1), 1)]
        w5 = [(idx_C(i0, k0), -1), (idx_C(i0, k0+1), 1)]
        add_rank1_term(u5, v5, w5)
        
        # M6 = (A21-A11)(B11+B12) -> C22
        u6 = [(idx_A(i0+1, j0), 1), (idx_A(i0, j0), -1)]
        v6 = [(idx_B(j0, k0), 1), (idx_B(j0, k0+1), 1)]
        w6 = [(idx_C(i0+1, k0+1), 1)]
        add_rank1_term(u6, v6, w6)
        
        # M7 = (A12-A22)(B21+B22) -> C11
        u7 = [(idx_A(i0, j0+1), 1), (idx_A(i0+1, j0+1), -1)]
        v7 = [(idx_B(j0+1, k0), 1), (idx_B(j0+1, k0+1), 1)]
        w7 = [(idx_C(i0, k0), 1)]
        add_rank1_term(u7, v7, w7)

    # Main decomposition logic
    
    # Part 1: j in 0..1
    # Sub-part 1: i in 0..0 (1x2x5)
    #   Split p into 2+2+1
    dec_1x2x2(0, 0, 0) # i=0, j=0..1, k=0..1
    dec_1x2x2(0, 0, 2) # i=0, j=0..1, k=2..3
    dec_1x2x1(0, 0, 4) # i=0, j=0..1, k=4
    
    # Sub-part 2: i in 1..2 (2x2x5)
    #   Split p into 2+2+1
    dec_2x2x2_strassen(1, 0, 0) # i=1..2, j=0..1, k=0..1
    dec_2x2x2_strassen(1, 0, 2) # i=1..2, j=0..1, k=2..3
    dec_2x2x1(1, 0, 4) # i=1..2, j=0..1, k=4
    
    # Part 2: j in 2..3
    # Sub-part 1: i in 0..0 (1x2x5)
    dec_1x2x2(0, 2, 0) # i=0, j=2..3, k=0..1
    dec_1x2x2(0, 2, 2) # i=0, j=2..3, k=2..3
    dec_1x2x1(0, 2, 4) # i=0, j=2..3, k=4
    
    # Sub-part 2: i in 1..2 (2x2x5)
    dec_2x2x2_strassen(1, 2, 0) # i=1..2, j=2..3, k=0..1
    dec_2x2x2_strassen(1, 2, 2) # i=1..2, j=2..3, k=2..3
    dec_2x2x1(1, 2, 4) # i=1..2, j=2..3, k=4
    
    # rank_idx should be 56
    # Trim matrices to actual rank
    U_final = U_mat[:, :rank_idx]
    V_final = V_mat[:, :rank_idx]
    W_final = W_mat[:, :rank_idx]
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
