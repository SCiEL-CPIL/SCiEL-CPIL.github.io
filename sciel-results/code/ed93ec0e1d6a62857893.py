"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # rank is 3 blocks * (7 Strassen + 1 Rank1) = 24
    rank = 24
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We define the terms for the 2x2 part (j=0,1)
    # Indices in local 2x2: i in {0,1}, j in {0,1}, k' in {0,1}
    # U entries: (i, j)
    # V entries: (j, k')
    # W entries: (i, k')
    
    strassen_terms = []
    
    # Term 0: M1
    # U: (0,0)=1, (1,1)=1
    # V: (0,0)=1, (0,1)=1, (1,0)=1, (1,1)=1
    # W: (0,0)=1, (0,1)=1, (1,0)=1, (1,1)=1
    u = {(0,0): 1, (1,1): 1}
    v = {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}
    w = {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}
    strassen_terms.append((u, v, w))
    
    # Term 1: M2 = (c+d)e
    # U: (1,0)=1, (1,1)=1
    # V: (0,0)=1, (0,1)=1
    # W: (0,0)=1, (0,1)=1, (1,0)=1, (1,1)=1
    u = {(1,0): 1, (1,1): 1}
    v = {(0,0): 1, (0,1): 1}
    w = {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}
    strassen_terms.append((u, v, w))
    
    # Term 2: M3 = a(f-h)
    # U: (0,0)=1
    # V: (0,1)=1, (1,1)=-1
    # W: (0,0)=1, (0,1)=1, (1,0)=1, (1,1)=1
    u = {(0,0): 1}
    v = {(0,1): 1, (1,1): -1}
    w = {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}
    strassen_terms.append((u, v, w))
    
    # Term 3: M4 = d(g-e)
    # U: (1,1)=1
    # V: (1,0)=1, (0,0)=-1
    # W: (0,0)=1, (0,1)=1, (1,0)=1, (1,1)=1
    u = {(1,1): 1}
    v = {(1,0): 1, (0,0): -1}
    w = {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}
    strassen_terms.append((u, v, w))
    
    # Term 4: M5 = (a+b)h
    # U: (0,0)=1, (0,1)=1
    # V: (1,0)=1, (1,1)=1
    # W: (0,0)=1, (0,1)=1, (1,0)=1, (1,1)=1
    u = {(0,0): 1, (0,1): 1}
    v = {(1,0): 1, (1,1): 1}
    w = {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}
    strassen_terms.append((u, v, w))
    
    # Term 5: M6 = (c-a)(e+f)
    # U: (1,0)=1, (0,0)=-1
    # V: (0,0)=1, (0,1)=1
    # W: (0,0)=1, (0,1)=1, (1,0)=1, (1,1)=1
    u = {(1,0): 1, (0,0): -1}
    v = {(0,0): 1, (0,1): 1}
    w = {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}
    strassen_terms.append((u, v, w))
    
    # Term 6: M7 = (b-d)(g+h)
    # U: (0,1)=1, (1,1)=-1
    # V: (1,0)=1, (1,1)=1
    # W: (0,0)=1, (0,1)=1, (1,0)=1, (1,1)=1
    u = {(0,1): 1, (1,1): -1}
    v = {(1,0): 1, (1,1): 1}
    w = {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}
    strassen_terms.append((u, v, w))
    
    # We need to map these local (i,j,k') to global indices and combine W coefficients.
    # The W coefficients above are for the local C matrix.
    # We need to accumulate the W values for each global (i, k') pair.
    # Actually, the Strassen algorithm defines C as a linear combination of Ms.
    # So for each term r, W_{i,k',r} is the coefficient of M_r in C_{i,k'}.
    
    # Let's re-define the terms with explicit W coefficients.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Term 0 (M1): W00=1, W01=0, W10=0, W11=1
    # Term 1 (M2): W00=0, W01=0, W10=1, W11=-1
    # Term 2 (M3): W00=0, W01=1, W10=0, W11=1
    # Term 3 (M4): W00=1, W01=0, W10=1, W11=0
    # Term 4 (M5): W00=-1, W01=1, W10=0, W11=0
    # Term 5 (M6): W00=0, W01=0, W10=0, W11=1
    # Term 6 (M7): W00=1, W01=0, W10=0, W11=0
    
    # Wait, my previous W dict was all 1s. That was wrong.
    # I need to use the correct coefficients.
    
    strassen_data = [
        # (u_dict, v_dict, w_dict)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}), # M1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 1}, {(1,0): 1, (1,1): -1}), # M2
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}), # M3
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}), # M4
        ({(0,0): 1, (0,1): 1}, {(1,0): 1, (1,1): 1}, {(0,0): -1, (0,1): 1}), # M5
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}), # M6
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}) # M7
    ]
    
    # Rank 1 term for j=2
    # U: (i, 2) = 1 for all i
    # V: (2, k') = 1 for all k'
    # W: (i, k') = 1 for all i, k'
    rank1_data = (
        {(0,2): 1, (1,2): 1},
        {(2,0): 1, (2,1): 1},
        {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}
    )
    
    all_terms = strassen_data + [rank1_data]
    
    for b in range(3):
        offset = b * 8
        for r_idx, (u_dict, v_dict, w_dict) in enumerate(all_terms):
            r = offset + r_idx
            for (i, j), val in u_dict.items():
                U[i * m + j][r] = val
            for (j, k_prime), val in v_dict.items():
                k_global = 2 * b + k_prime
                V[j * p + k_global][r] = val
            for (i, k_prime), val in w_dict.items():
                k_global = 2 * b + k_prime
                W[k_global * n + i][r] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
