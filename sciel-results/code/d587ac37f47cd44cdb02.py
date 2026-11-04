def _embed_2x2_to_4x4(U2, V2, W2):
    """Embed a 2x2x2 decomposition into a 4x4x4 decomposition using block Strassen."""
    n = 4
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    # Block-level Strassen terms
    block_terms = [
        # M1 = (A00+A11)(B00+B11) -> C00 +=, C11 +=
        {'a': [(0,0,1), (1,1,1)], 'b': [(0,0,1), (1,1,1)], 'c': [(0,0,1), (1,1,1)]},
        # M2 = (A10+A11)B00 -> C10 +=, C11 -=
        {'a': [(1,0,1), (1,1,1)], 'b': [(0,0,1)], 'c': [(1,0,1), (1,1,-1)]},
        # M3 = A00(B10-B11) -> C00 -=, C10 +=
        {'a': [(0,0,1)], 'b': [(1,0,1), (1,1,-1)], 'c': [(0,0,-1), (1,0,1)]},
        # M4 = A11(B01-B00) -> C00 +=, C11 +=
        {'a': [(1,1,1)], 'b': [(0,1,1), (0,0,-1)], 'c': [(0,0,1), (1,1,1)]},
        # M5 = (A00+A01)B11 -> C00 -=, C01 +=
        {'a': [(0,0,1), (0,1,1)], 'b': [(1,1,1)], 'c': [(0,0,-1), (0,1,1)]},
        # M6 = (A10-A00)(B00+B01) -> C01 +=, C10 -=
        {'a': [(1,0,1), (0,0,-1)], 'b': [(0,0,1), (0,1,1)], 'c': [(0,1,1), (1,0,-1)]},
        # M7 = (A01-A11)(B10+B11) -> C00 +=, C01 -=
        {'a': [(0,1,1), (1,1,-1)], 'b': [(1,0,1), (1,1,1)], 'c': [(0,0,1), (0,1,-1)]},
    ]
    
    col = 0
    for term in block_terms:
        for r in range(7):
            for lr in range(2):
                for lc in range(2):
                    # A part
                    for a_br, a_bc, a_sign in term['a']:
                        i = 2*a_br + lr
                        j = 2*a_bc + lc
                        U[i*4+j, col+r] += a_sign * U2[lr*2+lc, r]
                        
                    # B part
                    for b_br, b_bc, b_sign in term['b']:
                        j2 = 2*b_br + lr
                        k = 2*b_bc + lc
                        V[j2*4+k, col+r] += b_sign * V2[lr*2+lc, r]
                        
                    # C part
                    for c_br, c_bc, c_sign in term['c']:
                        i2 = 2*c_br + lr
                        k2 = 2*c_bc + lc
                        W[k2*4+i2, col+r] += c_sign * W2[lr*2+lc, r]
                        
        col += 7
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
