import numpy as np

def construct():
    """
    Rank-49 exact tensor decomposition for 4x4 matrix multiplication.
    
    Convention: U[i*m+k, r] for A[i,k], V[k*p+j, r] for B[k,j], W[j*n+i, r] for C[i,j]
    Tensor: T[i,k, k,j, j,i] = sum_r U[i*4+k, r] * V[k*4+j, r] * W[j*4+i, r] = delta[i,k]*delta[k,j]... 
    Actually: C[i,j] = sum_k A[i,k]*B[k,j], so T[i,k,k,j] = 1 iff we're computing C[i,j]
    
    The tensor is: T[a,b,c,d] where a=row of A, b=col of A (=row of B), c=col of B, d=row of C
    C[d, ?] ... let me use the trace convention: W[k*n+i, r] means W index = k*4+i where k is the 
    "middle" index and i is the row of C.
    
    So: sum_r U[i*4+k, r] * V[k*4+j, r] * W[j*4+i, r] should equal 1 if this contributes to C[i,j]
    which happens when... actually the tensor element T[i,k,j] = 1 for the matmul tensor means
    C[i,j] = sum_k A[i,k] B[k,j].
    
    With trace convention W[k*n+i, r]: the W factor has index (k, i) meaning it connects B's column 
    index j... hmm let me re-read: "W[k*n+i,r]" where the tensor is T[i*4+j, j*4+k, k*4+i]
    
    So U has index i*4+j (A[i,j]), V has index j*4+k (B[j,k]), W has index k*4+i (C[i,k] transposed?)
    
    The tensor: T[i,j, j,k, k,i] = sum_r U[i*4+j,r] V[j*4+k,r] W[k*4+i,r] = 1
    This means: for each (i,j,k), the sum over r should give 1. This is the matmul tensor where
    C[i,k] = sum_j A[i,j] B[j,k].
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    U = [[0.0] * rank for _ in range(n * m)]  # 16 x 49, index i*4+j
    V = [[0.0] * rank for _ in range(m * p)]  # 16 x 49, index j*4+k  
    W = [[0.0] * rank for _ in range(p * n)]  # 16 x 49, index k*4+i
    
    # Use 2x2 block Strassen recursively.
    # Split A into blocks A00, A01, A10, A11 (each 2x2)
    # Split B into blocks B00, B01, B10, B11 (each 2x2)
    # C = A*B has blocks C00, C01, C10, C11
    #
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # Block-level Strassen:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    #
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # For each Pi = Li * Ri (2x2 * 2x2), apply Strassen to get 7 scalar mults.
    # Total: 7 * 7 = 49.
    
    # Helper: add coefficient to U, V, W for a scalar multiplication
    # U_idx = i*4+j for A[i,j], V_idx = j*4+k for B[j,k], W_idx = k*4+i for C[i,k]
    
    def add_term(rank_idx, a_coeffs, b_coeffs, c_coeffs):
        """
        a_coeffs: list of (i, j, coeff) for A elements
        b_coeffs: list of (j, k, coeff) for B elements  
        c_coeffs: list of (i, k, coeff) for C elements (W index = k*4+i)
        """
        for (i, j, coeff) in a_coeffs:
            U[i * 4 + j][rank_idx] += coeff
        for (j, k, coeff) in b_coeffs:
            V[j * 4 + k][rank_idx] += coeff
        for (i, k, coeff) in c_coeffs:
            W[k * 4 + i][rank_idx] += coeff
    
    # Define the 7 block Strassen terms.
    # Each term: L is 2x2 with entries as lists of (global_i, global_j, coeff) for A
    #             R is 2x2 with entries as lists of (global_j, global_k, coeff) for B
    #             contrib: for each (br, bc) in 2x2 output, list of (global_i, global_k, coeff)
    
    # Block (bi, bj) of A: rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    # Element (r,c) in block -> global (2*bi+r, 2*bj+c)
    
    def A_block_elem(bi, bj, r, c):
        return (2*bi + r, 2*bj + c)
    
    def B_block_elem(bi, bj, r, c):
        return (2*bi + r, 2*bj + c)
    
    def C_block_elem(bi, bj, r, c):
        return (2*bi + r, 2*bj + c)
    
    # Build L and R for each of the 7 terms
    terms = []
    
    # Term 0: P1 = (A00+A11)(B00+B11)
    L0 = [[[] for _ in range(2)] for _ in range(2)]
    R0 = [[[] for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            i1, j1 = A_block_elem(0, 0, r, c)
            i2, j2 = A_block_elem(1, 1, r, c)
            L0[r][c] = [(i1, j1, 1.0), (i2, j2, 1.0)]
            j1, k1 = B_block_elem(0, 0, r, c)
            j2, k2 = B_block_elem(1, 1, r, c)
            R0[r][c] = [(j1, k1, 1.0), (j2, k2, 1.0)]
    contrib0 = {}
    for r in range(2):
        for c in range(2):
            i1, k1 = C_block_elem(0, 0, r, c)
            i2, k2 = C_block_elem(1, 1, r, c)
            contrib0[(r, c)] = [(i1, k1, 1.0), (i2, k2, 1.0)]
    terms.append((L0, R0, contrib0))
    
    # Term 1: P2 = (A10+A11)B00
    L1 = [[[] for _ in range(2)] for _ in range(2)]
    R1 = [[[] for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            i1, j1 = A_block_elem(1, 0, r, c)
            i2, j2 = A_block_elem(1, 1, r, c)
            L1[r][c] = [(i1, j1, 1.0), (i2, j2, 1.0)]
            j1, k1 = B_block_elem(0, 0, r, c)
            R1[r][c] = [(j1, k1, 1.0)]
    contrib1 = {}
    for r in range(2):
        for c in range(2):
            i1, k1 = C_block_elem(1, 0, r, c)
            i2, k2 = C_block_elem(1, 1, r, c)
            contrib1[(r, c)] = [(i1, k1, 1.0), (i2, k2, -1.0)]
    terms.append((L1, R1, contrib1))
    
    # Term 2: P3 = A00(B01-B11)
    L2 = [[[] for _ in range(2)] for _ in range(2)]
    R2 = [[[] for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            i1, j1 = A_block_elem(0, 0, r, c)
            L2[r][c] = [(i1, j1, 1.0)]
            j1, k1 = B_block_elem(0, 1, r, c)
            j2, k2 = B_block_elem(1, 1, r, c)
            R2[r][c] = [(j1, k1, 1.0), (j2, k2, -1.0)]
    contrib2 = {}
    for r in range(2):
        for c in range(2):
            i1, k1 = C_block_elem(0, 1, r, c)
            i2, k2 = C_block_elem(1, 1, r, c)
            contrib2[(r, c)] = [(i1, k1, 1.0), (i2, k2, 1.0)]
    terms.append((L2, R2, contrib2))
    
    # Term 3: P4 = A11(B10-B00)
    L3 = [[[] for _ in range(2)] for _ in range(2)]
    R3 = [[[] for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            i1, j1 = A_block_elem(1, 1, r, c)
            L3[r][c] = [(i1, j1, 1.0)]
            j1, k1 = B_block_elem(1, 0, r, c)
            j2, k2 = B_block_elem(0, 0, r, c)
            R3[r][c] = [(j1, k1, 1.0), (j2, k2, -1.0)]
    contrib3 = {}
    for r in range(2):
        for c in range(2):
            i1, k1 = C_block_elem(0, 0, r, c)
            i2, k2 = C_block_elem(1, 0, r, c)
            contrib3[(r, c)] = [(i1, k1, 1.0), (i2, k2, 1.0)]
    terms.append((L3, R3, contrib3))
    
    # Term 4: P5 = (A00+A01)B11
    L4 = [[[] for _ in range(2)] for _ in range(2)]
    R4 = [[[] for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            i1, j1 = A_block_elem(0, 0, r, c)
            i2, j2 = A_block_elem(0, 1, r, c)
            L4[r][c] = [(i1, j1, 1.0), (i2, j2, 1.0)]
            j1, k1 = B_block_elem(1, 1, r, c)
            R4[r][c] = [(j1, k1, 1.0)]
    contrib4 = {}
    for r in range(2):
        for c in range(2):
            i1, k1 = C_block_elem(0, 0, r, c)
            i2, k2 = C_block_elem(0, 1, r, c)
            contrib4[(r, c)] = [(i1, k1, -1.0), (i2, k2, 1.0)]
    terms.append((L4, R4, contrib4))
    
    # Term 5: P6 = (A10-A00)(B00+B01)
    L5 = [[[] for _ in range(2)] for _ in range(2)]
    R5 = [[[] for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            i1, j1 = A_block_elem(1, 0, r, c)
            i2, j2 = A_block_elem(0, 0, r, c)
            L5[r][c] = [(i1, j1, 1.0), (i2, j2, -1.0)]
            j1, k1 = B_block_elem(0, 0, r, c)
            j2, k2 = B_block_elem(0, 1, r, c)
            R5[r][c] = [(j1, k1, 1.0), (j2, k2, 1.0)]
    contrib5 = {}
    for r in range(2):
        for c in range(2):
            i1, k1 = C_block_elem(1, 1, r, c)
            contrib5[(r, c)] = [(i1, k1, 1.0)]
    terms.append((L5, R5, contrib5))
    
    # Term 6: P7 = (A01-A11)(B10+B11)
    L6 = [[[] for _ in range(2)] for _ in range(2)]
    R6 = [[[] for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            i1, j1 = A_block_elem(0, 1, r, c)
            i2, j2 = A_block_elem(1, 1, r, c)
            L6[r][c] = [(i1, j1, 1.0), (i2, j2, -1.0)]
            j1, k1 = B_block_elem(1, 0, r, c)
            j2, k2 = B_block_elem(1, 1, r, c)
            R6[r][c] = [(j1, k1, 1.0), (j2, k2, 1.0)]
    contrib6 = {}
    for r in range(2):
        for c in range(2):
            i1, k1 = C_block_elem(0, 0, r, c)
            contrib6[(r, c)] = [(i1, k1, 1.0)]
    terms.append((L6, R6, contrib6))
    
    # Strassen sub-product contributions for 2x2 result:
    # M1=(a+d)(e+h): P11+1, P22+1
    # M2=(c+d)e:     P21+1, P22-1
    # M3=a(f-h):      P12+1, P22+1
    # M4=d(g-e):      P11+1, P21+1
    # M5=(a+b)h:      P11-1, P12+1
    # M6=(c-a)(e+f):  P22+1
    # M7=(b-d)(g+h):  P11+1
    sub_contrib = [
        [((0,0), 1.0), ((1,1), 1.0)],
        [((1,0), 1.0), ((1,1), -1.0)],
        [((0,1), 1.0), ((1,1), 1.0)],
        [((0,0), 1.0), ((1,0), 1.0)],
        [((0,0), -1.0), ((0,1), 1.0)],
        [((1,1), 1.0)],
        [((0,0), 1.0)]
    ]
    
    # For each term t and sub-product k, build the scalar multiplication
    for t in range(7):
        L_t, R_t, contrib_t = terms[t]
        
        # Extract a,b,c,d from L_t and e,f,g,h from R_t
        # Each is a list of (idx1, idx2, coeff)
        a = L_t[0][0]  # top-left
        b = L_t[0][1]  # top-right
        c = L_t[1][0]  # bottom-left
        d = L_t[1][1]  # bottom-right
        e = R_t[0][0]
        f = R_t[0][1]
        g = R_t[1][0]
        h = R_t[1][1]
        
        # Define 7 sub-products as (x_list, y_list) where x_list is for A, y_list for B
        # M1: X=a+d, Y=e+h
        def combine_add(list1, list2):
            result = {}
            for (i, j, coeff) in list1:
                result[(i, j)] = result.get((i, j), 0.0) + coeff
            for (i, j, coeff) in list2:
                result[(i, j)] = result.get((i, j), 0.0) + coeff
            return [(k[0], k[1], v) for k, v in result.items() if v != 0.0]
        
        def combine_sub(list1, list2):
            result = {}
            for (i, j, coeff) in list1:
                result[(i, j)] = result.get((i, j), 0.0) + coeff
            for (i, j, coeff) in list2:
                result[(i, j)] = result.get((i, j), 0.0) - coeff
            return [(k[0], k[1], v) for k, v in result.items() if v != 0.0]
        
        sub_products = [
            (combine_add(a, d), combine_add(e, h)),      # M1
            (combine_add(c, d), e),                       # M2
            (a, combine_sub(f, h)),                       # M3
            (d, combine_sub(g, e)),                       # M4
            (combine_add(a, b), h),                       # M5
            (combine_sub(c, a), combine_add(e, f)),       # M6
            (combine_sub(b, d), combine_add(g, h))        # M7
        ]
        
        for k in range(7):
            rank_idx = t * 7 + k
            x_list, y_list = sub_products[k]
            
            # Add to U and V
            for (i, j, coeff) in x_list:
                U[i * 4 + j][rank_idx] += coeff
            for (j, k_idx, coeff) in y_list:
                V[j * 4 + k_idx][rank_idx] += coeff
            
            # Add to W
            for ((pr, pc), sub_coeff) in sub_contrib[k]:
                if (pr, pc) in contrib_t:
                    for (i, k_idx, block_coeff) in contrib_t[(pr, pc)]:
                        W[k_idx * 4 + i][rank_idx] += sub_coeff * block_coeff
    
    return {"U": U, "V": V, "W": W}
