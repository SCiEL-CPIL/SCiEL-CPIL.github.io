def construct():
    n, m, p = 2, 4, 8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen terms for 2x2x2 multiplication
    # Local indices: i in {0,1}, j in {0,1}, k in {0,1}
    # A(i,j), B(j,k), C(i,k)
    
    strassen_terms = [
        # P1: (A00+A11)(B00+B11) -> C00+, C11+
        {
            "U": [(0,0,1), (1,1,1)],
            "V": [(0,0,1), (1,1,1)],
            "W": [(0,0,1), (1,1,1)]
        },
        # P2: (A10+A11)B00 -> C10+, C11-
        {
            "U": [(1,0,1), (1,1,1)],
            "V": [(0,0,1)],
            "W": [(1,0,1), (1,1,-1)]
        },
        # P3: A00(B01-B11) -> C01+, C11+
        {
            "U": [(0,0,1)],
            "V": [(0,1,1), (1,1,-1)],
            "W": [(0,1,1), (1,1,1)]
        },
        # P4: A11(B10-B00) -> C00+, C10+
        {
            "U": [(1,1,1)],
            "V": [(1,0,1), (0,0,-1)],
            "W": [(0,0,1), (1,0,1)]
        },
        # P5: (A00+A01)B11 -> C00-, C01+
        {
            "U": [(0,0,1), (0,1,1)],
            "V": [(1,1,1)],
            "W": [(0,0,-1), (0,1,1)]
        },
        # P6: (A10-A00)(B00+B01) -> C11+
        {
            "U": [(1,0,1), (0,0,-1)],
            "V": [(0,0,1), (0,1,1)],
            "W": [(1,1,1)]
        },
        # P7: (A01-A11)(B10+B11) -> C00+
        {
            "U": [(0,1,1), (1,1,-1)],
            "V": [(1,0,1), (1,1,1)],
            "W": [(0,0,1)]
        },
    ]
    
    # Iterate over blocks of m (size 2) and blocks of p (size 2)
    # m_block in {0, 1} corresponds to rows 0-1 and 2-3 of B, cols 0-1 and 2-3 of A
    # p_block in {0, 1, 2, 3} corresponds to cols 0-1, 2-3, 4-5, 6-7 of B and C
    
    for m_block in range(2):
        for p_block in range(4):
            for term in strassen_terms:
                # Map local indices to global
                # A: i in {0,1}, j_local in {0,1} -> j_global = 2*m_block + j_local
                # B: j_local in {0,1}, k_local in {0,1} -> j_global = 2*m_block + j_local, k_global = 2*p_block + k_local
                # C: i in {0,1}, k_local in {0,1} -> k_global = 2*p_block + k_local
                
                for (i, jl, val) in term["U"]:
                    jg = 2 * m_block + jl
                    U[i * 4 + jg][col] += val
                
                for (jl, kl, val) in term["V"]:
                    jg = 2 * m_block + jl
                    kg = 2 * p_block + kl
                    V[jg * 8 + kg][col] += val
                
                for (i, kl, val) in term["W"]:
                    kg = 2 * p_block + kl
                    W[kg * 2 + i][col] += val
                
                col += 1

    return {"U": U, "V": V, "W": W}
