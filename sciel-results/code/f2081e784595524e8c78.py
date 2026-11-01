import numpy as np

def construct():
    n, m, p = 2, 2, 10
    num_blocks = p // 2
    rank = num_blocks * 7
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen terms for 2x2 multiplication
    # Each term is (u_dict, v_dict, w_dict)
    # u_dict: (i, j) -> val
    # v_dict: (j, k) -> val
    # w_dict: (k, i) -> val
    
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    strassen_terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (1,0): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    for r in range(num_blocks):
        base_col = r * 7
        for t in range(7):
            col = base_col + t
            u_entries, v_entries, w_entries = strassen_terms[t]
            
            for (i, j), val in u_entries.items():
                U[i * m + j, col] = val
            
            for (j, k), val in v_entries.items():
                # B index: j * p + (2*r + k)
                V[j * p + (2 * r + k), col] = val
            
            for (k, i), val in w_entries.items():
                # W index: (2*r + k) * n + i
                W[(2 * r + k) * n + i, col] = val
                
    return {"U": U, "V": V, "W": W}
