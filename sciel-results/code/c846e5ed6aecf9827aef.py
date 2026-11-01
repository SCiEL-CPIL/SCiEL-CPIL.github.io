def _strassen_4x4():
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Block combinations for 4x4 Strassen
    # M1 = (A00+A11)(B00+B11)
    # ...
    
    block_combos = [
        # M1
        {'A': [((0,0), 1), ((1,1), 1)], 'B': [((0,0), 1), ((1,1), 1)]},
        # M2
        {'A': [((1,0), 1), ((1,1), 1)], 'B': [((0,0), 1)]},
        # M3
        {'A': [((0,0), 1)], 'B': [((1,0), 1), ((1,1), -1)]},
        # M4
        {'A': [((1,1), 1)], 'B': [((0,1), 1), ((0,0), -1)]},
        # M5
        {'A': [((0,0), 1), ((0,1), 1)], 'B': [((1,1), 1)]},
        # M6
        {'A': [((1,0), 1), ((0,0), -1)], 'B': [((0,0), 1), ((0,1), 1)]},
        # M7
        {'A': [((0,1), 1), ((1,1), -1)], 'B': [((1,0), 1), ((1,1), 1)]},
    ]
    
    # 2x2 Strassen decomposition
    # S1: u=a00+a11, v=b00+b11, c00+=1, c11+=1
    # S2: u=a10+a11, v=b00, c10+=1, c11-=1
    # S3: u=a00, v=b10-b11, c01+=1, c11+=1
    # S4: u=a11, v=b01-b00, c00+=1, c10+=1
    # S5: u=a00+a01, v=b11, c00-=1, c01+=1
    # S6: u=a10-a00, v=b00+b01, c11+=1
    # S7: u=a01-a11, v=b10+b11, c00+=1
    
    s2x2 = [
        {'u': [((0,0), 1), ((1,1), 1)], 'v': [((0,0), 1), ((1,1), 1)], 'c': [((0,0), 1), ((1,1), 1)]},
        {'u': [((1,0), 1), ((1,1), 1)], 'v': [((0,0), 1)], 'c': [((1,0), 1), ((1,1), -1)]},
        {'u': [((0,0), 1)], 'v': [((1,0), 1), ((1,1), -1)], 'c': [((0,1), 1), ((1,1), 1)]},
        {'u': [((1,1), 1)], 'v': [((0,1), 1), ((0,0), -1)], 'c': [((0,0), 1), ((1,0), 1)]},
        {'u': [((0,0), 1), ((0,1), 1)], 'v': [((1,1), 1)], 'c': [((0,0), -1), ((0,1), 1)]},
        {'u': [((1,0), 1), ((0,0), -1)], 'v': [((0,0), 1), ((0,1), 1)], 'c': [((1,1), 1)]},
        {'u': [((0,1), 1), ((1,1), -1)], 'v': [((1,0), 1), ((1,1), 1)], 'c': [((0,0), 1)]},
    ]
    
    col = 0
    for k in range(7):
        bA = block_combos[k]['A']
        bB = block_combos[k]['B']
        
        for j in range(7):
            s = s2x2[j]
            
            # Build U coefficients
            u_coeffs = {}
            for (a_br, a_bc), a_sign in bA:
                for (li, lj), l_sign in s['u']:
                    gi = 2*a_br + li
                    gj = 2*a_bc + lj
                    u_coeffs[(gi, gj)] = u_coeffs.get((gi, gj), 0) + a_sign * l_sign
            
            # Build V coefficients
            v_coeffs = {}
            for (b_br, b_bc), b_sign in bB:
                for (li, lj), l_sign in s['v']:
                    gi = 2*b_br + li
                    gj = 2*b_bc + lj
                    v_coeffs[(gi, gj)] = v_coeffs.get((gi, gj), 0) + b_sign * l_sign
            
            # Fill U
            for (gi, gj), coeff in u_coeffs.items():
                U[gi*m + gj, col] += coeff
            
            # Fill V
            for (gi, gj), coeff in v_coeffs.items():
                V[gi*p + gj, col] += coeff
            
            # Fill W
            for (c_br, c_bc), c_sign in s['c']:
                for li in range(2):
                    for lj in range(2):
                        gi = 2*c_br + li
                        gj = 2*c_bc + lj
                        # Trace convention: W[k*n + i, r] where C[i,k]
                        # C index is (gi, gj). So i=gi, k=gj.
                        # W row index: gj * n + gi
                        W[gj*n + gi, col] += c_sign
                        
            col += 1
            
    return {'U': U, 'V': V, 'W': W}
