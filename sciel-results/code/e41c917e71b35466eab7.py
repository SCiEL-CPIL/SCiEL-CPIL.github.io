import numpy as np

def construct():
    rank = 49
    U = [[0.0] * rank for _ in range(16)]
    V = [[0.0] * rank for _ in range(16)]
    W = [[0.0] * rank for _ in range(16)]
    
    # We'll build the decomposition by direct enumeration.
    # The tensor is: T[i,j,k,l] = delta_{j,k}
    # U[i,j,r] * V[k,l,r] * W[l,i,r] summed over r should give delta_{j,k}
    
    # Using recursive Strassen (2 levels of 2x2 Strassen):
    # Level 1: Split 4x4 into 2x2 blocks, apply Strassen (7 block products)
    # Level 2: Each block product is 2x2, apply Strassen (7 scalar products)
    # Total: 49 scalar products
    
    # I'll use a helper to build 2x2 Strassen terms and compose them.
    
    # For a 2x2 matrix product with elements as linear combos:
    # If A = [[a,b],[c,d]] and B = [[e,f],[g,h]] where a,b,c,d,e,f,g,h are linear combos,
    # then the 7 Strassen terms are:
    # T1: U=(a+d), V=(e+h), W contributes to C00:+1, C11:+1
    # T2: U=(c+d), V=e, W contributes to C10:+1, C11:-1
    # T3: U=a, V=(f-h), W contributes to C01:+1, C11:+1
    # T4: U=d, V=(g-e), W contributes to C00:+1, C10:+1
    # T5: U=(a+b), V=h, W contributes to C00:-1, C01:+1
    # T6: U=(c-a), V=(e+f), W contributes to C11:+1
    # T7: U=(b-d), V=(g+h), W contributes to C00:+1
    
    # For the block level, each "element" a,b,c,d,e,f,g,h is itself a 2x2 matrix
    # (a block). So when we apply Strassen at the block level, each term Tk has
    # U_block and V_block which are 2x2 matrices of linear combos.
    # The product U_block * V_block is a 2x2 matrix product, decomposed with rank-7.
    
    # Let me define everything in terms of global indices.
    
    def A_idx(i, j):
        return i * 4 + j
    
    def B_idx(i, j):
        return i * 4 + j
    
    def C_idx(i, j):
        return j * 4 + i  # trace convention W[k*n+i] where k=j, i=i
    
    # Block (bi, bj) covers rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    # Element (r, c) within block: global row = 2*bi+r, col = 2*bj+c
    
    # For each of the 7 block-level Strassen terms, define:
    # - The 2x2 U-block matrix (each entry is a dict {A_global_idx: coeff})
    # - The 2x2 V-block matrix (each entry is a dict {B_global_idx: coeff})
    # - The contribution pattern: which C blocks this term's result contributes to
    
    # Block-level Strassen terms (Lt, Rt) where L and R are 2x2 block matrices:
    # Term 0: L=A11+A22, R=B11+B22 -> contributes to C00:+1, C11:+1
    # Term 1: L=A21+A22, R=B11 -> contributes to C10:+1, C11:-1
    # Term 2: L=A11, R=B12-B22 -> contributes to C01:+1, C11:+1
    # Term 3: L=A22, R=B21-B11 -> contributes to C00:+1, C10:+1
    # Term 4: L=A11+A12, R=B22 -> contributes to C00:-1, C01:+1
    # Term 5: L=A21-A11, R=B11+B12 -> contributes to C11:+1
    # Term 6: L=A12-A22, R=B21+B22 -> contributes to C00:+1
    
    # For each term, L and R are uniform 2x2 block matrices (all 4 entries same linear combo)
    
    def make_uniform_block(combo, is_A):
        """Create a 2x2 matrix where each entry is the linear combo expanded to global indices"""
        result = [[{} for _ in range(2)] for _ in range(2)]
        for block, coeff in combo.items():
            bi = block // 2
            bj = block % 2
            for r in range(2):
                for c in range(2):
                    gidx = A_idx(2*bi+r, 2*bj+c) if is_A else B_idx(2*bi+r, 2*bj+c)
                    result[r][c][gidx] = result[r][c].get(gidx, 0) + coeff
        return result
    
    block_terms = [
        ({0:1, 3:1}, {0:1, 3:1}, [(0,0,1), (1,1,1)]),
        ({2:1, 3:1}, {0:1}, [(1,0,1), (1,1,-1)]),
        ({0:1}, {1:1, 3:-1}, [(0,1,1), (1,1,1)]),
        ({3:1}, {2:1, 0:-1}, [(0,0,1), (1,0,1)]),
        ({0:1, 1:1}, {3:1}, [(0,0,-1), (0,1,1)]),
        ({2:1, 0:-1}, {0:1, 1:1}, [(1,1,1)]),
        ({1:1, 3:-1}, {2:1, 3:1}, [(0,0,1)]),
    ]
    
    for t, (L_combo, R_combo, contribs) in enumerate(block_terms):
        # Create 2x2 matrices for L and R
        L_mat = make_uniform_block(L_combo, True)
        R_mat = make_uniform_block(R_combo, False)
        
        # Extract elements
        L00, L01 = L_mat[0][0], L_mat[0][1]
        L10, L11 = L_mat[1][0], L_mat[1][1]
        R00, R01 = R_mat[0][0], R_mat[0][1]
        R10, R11 = R_mat[1][0], R_mat[1][1]
        
        def add_d(d1, d2, s2=1):
            r = {}
            for k, v in d1.items(): r[k] = r.get(k, 0) + v
            for k, v in d2.items(): r[k] = r.get(k, 0) + s2 * v
            return r
        
        # 7 Strassen sub-products for L_mat * R_mat
        subs = [
            (add_d(L00, L11), add_d(R00, R11), {(0,0):1, (1,1):1}),
            (add_d(L10, L11), R00, {(1,0):1, (1,1):-1}),
            (L00, add_d(R01, R11, -1), {(0,1):1, (1,1):1}),
            (L11, add_d(R10, R00, -1), {(0,0):1, (1,0):1}),
            (add_d(L00, L01), R11, {(0,0):-1, (0,1):1}),
            (add_d(L10, L00, -1), add_d(R00, R01), {(1,1):1}),
            (add_d(L01, L11, -1), add_d(R10, R11), {(0,0):1}),
        ]
        
        for k, (Ud, Vd, Wpat) in enumerate(subs):
            ridx = t * 7 + k
            for idx, coeff in Ud.items():
                if coeff != 0: U[idx][ridx] += coeff
            for idx, coeff in Vd.items():
                if coeff != 0: V[idx][ridx] += coeff
            # W contribution: for each (lbr, lbc, wcoeff) in Wpat,
            # and for each (cbr, cbc, bcoeff) in contribs:
            #   total = wcoeff * bcoeff
            #   The local 2x2 result at position (lbr, lbc) maps to C block (cbr, cbc)
            #   Each element (r,c) of the local 2x2 at (lbr,lbc) maps to C element
            #   at global position (2*cbr+r, 2*cbc+c) with coefficient total
            for (lbr, lbc), wcoeff in Wpat.items():
                for (cbr, cbc, bcoeff) in contribs:
                    tc = wcoeff * bcoeff
                    if tc == 0: continue
                    # The local 2x2 at (lbr,lbc) has elements (r,c) for r,c in {0,1}
                    # Each maps to C element (2*cbr+r, 2*cbc+c)
                    for r in range(2):
                        for c in range(2):
                            widx = C_idx(2*cbr+r, 2*cbc+c)
                            W[widx][ridx] += tc
    
    return {"U": U, "V": V, "W": W}
