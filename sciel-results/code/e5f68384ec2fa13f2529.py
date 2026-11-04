def construct():
    n, m, p = 2, 4, 8
    
    # Rank of M_{2,2,2} is 7 (Strassen)
    # Rank of M_{1,2,4} is 8 (4 independent dot products of length 2, each rank 2)
    # Total rank = 7 * 8 = 56
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2
    # U, V, W for M_{2,2,2}
    # Indices: i,j,k in {0,1}
    # U size 4x7, V size 4x7, W size 4x7
    
    strassen_U = [
        [1, 0, 1, 0, 0, 0, 0], # A00
        [0, 0, 0, 1, 0, 0, 0], # A01
        [0, 1, 0, 0, 1, 0, 0], # A10
        [0, 0, 0, 0, 0, 1, 0], # A11
    ]
    # Wait, standard Strassen:
    # M1: (A00+A11)(B00+B11) -> U: A00(1), A11(1). V: B00(1), B11(1).
    # M2: (A10+A11)B00 -> U: A10(1), A11(1). V: B00(1).
    # M3: A00(B01-B11) -> U: A00(1). V: B01(1), B11(-1).
    # M4: A11(B10-B00) -> U: A11(1). V: B10(1), B00(-1).
    # M5: (A00+A01)B11 -> U: A00(1), A01(1). V: B11(1).
    # M6: (A10-A00)(B00+B01) -> U: A10(1), A00(-1). V: B00(1), B01(1).
    # M7: (A01-A11)(B10+B11) -> U: A01(1), A11(-1). V: B10(1), B11(1).
    
    # Let's build the 7 terms explicitly for 2x2
    terms_2x2 = []
    
    # P1
    terms_2x2.append({
        "U": [(0,0,1), (1,1,1)],
        "V": [(0,0,1), (1,1,1)],
        "W": [(0,0,1), (1,1,1)]
    })
    # P2
    terms_2x2.append({
        "U": [(1,0,1), (1,1,1)],
        "V": [(0,0,1)],
        "W": [(1,0,1), (1,1,-1)]
    })
    # P3
    terms_2x2.append({
        "U": [(0,0,1)],
        "V": [(0,1,1), (1,1,-1)],
        "W": [(0,1,1), (1,1,1)]
    })
    # P4
    terms_2x2.append({
        "U": [(1,1,1)],
        "V": [(1,0,1), (0,0,-1)],
        "W": [(0,0,1), (1,0,1)]
    })
    # P5
    terms_2x2.append({
        "U": [(0,0,1), (0,1,1)],
        "V": [(1,1,1)],
        "W": [(0,0,-1), (0,1,1)]
    })
    # P6
    terms_2x2.append({
        "U": [(1,0,1), (0,0,-1)],
        "V": [(0,0,1), (0,1,1)],
        "W": [(1,1,1)]
    })
    # P7
    terms_2x2.append({
        "U": [(0,1,1), (1,1,-1)],
        "V": [(1,0,1), (1,1,1)],
        "W": [(0,0,1)]
    })
    
    # Trivial terms for 1x2x4 (Rank 8)
    # A' is 1x2 (indices j'=0,1)
    # B' is 2x4 (indices j'=0,1; k'=0,1,2,3)
    # C' is 1x4 (indices k'=0,1,2,3)
    # Terms:
    # For each k' in 0..3:
    #   T_k0: A'_0 B'_{0,k'} -> C'_{k'}
    #   T_k1: A'_1 B'_{1,k'} -> C'_{k'}
    
    terms_1x2x4 = []
    for k_prime in range(4):
        terms_1x2x4.append({
            "U": [(0, 0, 1)], # A'_0 (i'=0, j'=0)
            "V": [(0, k_prime, 1)], # B'_{0, k'}
            "W": [(k_prime, 0, 1)] # C'_{k'} (k', i'=0)
        })
        terms_1x2x4.append({
            "U": [(0, 1, 1)], # A'_1 (i'=0, j'=1)
            "V": [(1, k_prime, 1)], # B'_{1, k'}
            "W": [(k_prime, 0, 1)] # C'_{k'}
        })
        
    # Combine
    col = 0
    for t1 in terms_2x2:
        for t2 in terms_1x2x4:
            # Map indices
            # A: i (0..1), j (0..3). j = 2*j1 + j2
            # B: j (0..3), k (0..7). j = 2*j1 + j2, k = 2*k1 + k2
            # C: i (0..1), k (0..7). k = 2*k1 + k2
            
            for (i, j1, val_u) in t1["U"]:
                for (i2, j2, val_u2) in t2["U"]:
                    # i2 is always 0
                    # j_global = 2*j1 + j2
                    jg = 2*j1 + j2
                    U[i*4 + jg][col] += val_u * val_u2
            
            for (j1, k1, val_v) in t1["V"]:
                for (j2, k2, val_v2) in t2["V"]:
                    # j_global = 2*j1 + j2
                    # k_global = 2*k1 + k2
                    jg = 2*j1 + j2
                    kg = 2*k1 + k2
                    V[jg*8 + kg][col] += val_v * val_v2
            
            for (i, k1, val_w) in t1["W"]:
                for (k2, i2, val_w2) in t2["W"]:
                    # i is from t1, i2 is 0
                    # k_global = 2*k1 + k2
                    kg = 2*k1 + k2
                    W[kg*2 + i][col] += val_w * val_w2
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
