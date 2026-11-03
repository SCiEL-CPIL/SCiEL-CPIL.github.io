"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    
    # Strassen decomposition for 2x2 matrices
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    strassen_terms = [
        # Term 1: P1 = (A00 + A11)(B00 + B11)
        {
            'U': {0: 1, 3: 1},
            'V': {0: 1, 3: 1},
            'W': {0: 1, 3: 1}
        },
        # Term 2: P2 = (A10 + A11)B00
        {
            'U': {2: 1, 3: 1},
            'V': {0: 1},
            'W': {2: 1, 3: -1}
        },
        # Term 3: P3 = (A00 - A10)(B10 + B11)
        {
            'U': {0: 1, 2: -1},
            'V': {2: 1, 3: 1},
            'W': {1: 1, 3: 1}
        },
        # Term 4: P4 = A11(B10 - B00)
        {
            'U': {3: 1},
            'V': {2: 1, 0: -1},
            'W': {0: 1, 2: 1}
        },
        # Term 5: P5 = A00(B00 - B10)
        {
            'U': {0: 1},
            'V': {0: 1, 2: -1},
            'W': {0: -1, 1: 1}
        },
        # Term 6: P6 = (A01 + A11)B11
        {
            'U': {1: 1, 3: 1},
            'V': {3: 1},
            'W': {3: 1}
        },
        # Term 7: P7 = (A00 + A10)(B01 - B11)
        {
            'U': {0: 1, 2: 1},
            'V': {1: 1, 3: -1},
            'W': {0: -1}
        }
    ]
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t1 in range(7):
        outer_term = strassen_terms[t1]
        for t2 in range(7):
            inner_term = strassen_terms[t2]
            
            # Process U
            for I in range(2):
                for J in range(2):
                    u_outer = outer_term['U'].get((I, J), 0)
                    if u_outer == 0:
                        continue
                    for i in range(2):
                        for j in range(2):
                            u_inner = inner_term['U'].get((i, j), 0)
                            if u_inner == 0:
                                continue
                            coeff = u_outer * u_inner
                            row = I * 2 + i
                            col_idx = J * 2 + j
                            u_idx = row * 5 + col_idx
                            U[u_idx][col] += coeff
            
            # Process V
            for J in range(2):
                for K in range(2):
                    v_outer = outer_term['V'].get((J, K), 0)
                    if v_outer == 0:
                        continue
                    for j in range(2):
                        for k in range(2):
                            v_inner = inner_term['V'].get((j, k), 0)
                            if v_inner == 0:
                                continue
                            coeff = v_outer * v_inner
                            row = J * 2 + j
                            col_idx = K * 2 + k
                            v_idx = row * 5 + col_idx
                            V[v_idx][col] += coeff
            
            # Process W
            for K in range(2):
                for I in range(2):
                    w_outer = outer_term['W'].get((K, I), 0)
                    if w_outer == 0:
                        continue
                    for k in range(2):
                        for i in range(2):
                            w_inner = inner_term['W'].get((k, i), 0)
                            if w_inner == 0:
                                continue
                            coeff = w_outer * w_inner
                            row = K * 2 + k
                            col_idx = I * 2 + i
                            w_idx = row * 5 + col_idx
                            W[w_idx][col] += coeff
                            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
