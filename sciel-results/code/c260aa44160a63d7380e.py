import numpy as np

def construct():
    n, m, p = 2, 4, 8
    R = 56
    
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)
    
    col = 0
    
    # Strassen terms (u, v, w)
    # u: coeffs for A (00, 01, 10, 11)
    # v: coeffs for B (00, 01, 10, 11)
    # w: coeffs for C (00, 01, 10, 11)
    strassen_terms = [
        ((1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 1)),  # M1
        ((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, -1)),  # M2
        ((1, 0, 0, 0), (0, 0, 1, -1), (0, 1, 0, 1)),   # M3
        ((0, 0, 0, 1), (0, 1, -1, 0), (1, 0, 1, 0)),   # M4
        ((1, 1, 0, 0), (0, 0, 0, 1), (-1, 1, 0, 0)),   # M5
        ((-1, 0, 1, 0), (1, 1, 0, 0), (0, 0, 0, 1)),   # M6
        ((0, 1, 0, -1), (0, 0, 1, 1), (1, 0, 0, 0))    # M7
    ]
    
    def add_product(u_map, v_map, w_map):
        nonlocal col
        for u, v, w in strassen_terms:
            # u, v, w are tuples of 4 values
            # u_map is dict: 0->idx00, 1->idx01, 2->idx10, 3->idx11
            # We need to add u[k] to U[u_map[k], col]
            for k in range(4):
                if u[k] != 0:
                    U[u_map[k], col] += u[k]
                if v[k] != 0:
                    V[v_map[k], col] += v[k]
                if w[k] != 0:
                    W[w_map[k], col] += w[k]
            col += 1

    for j in range(4):
        # Product 1: A0 * B0,j
        # A0: A[:, 0:2]
        u_map_1 = {0: 0, 1: 1, 2: 4, 3: 5}
        # B0,j: B[0:2, 2j:2j+2]
        v_map_1 = {0: 2*j, 1: 2*j+1, 2: 8+2*j, 3: 8+2*j+1}
        # C block j: C[:, 2j:2j+2]
        w_map_1 = {0: 4*j, 1: 4*j+2, 2: 4*j+1, 3: 4*j+3}
        
        add_product(u_map_1, v_map_1, w_map_1)
        
        # Product 2: A1 * B1,j
        # A1: A[:, 2:4]
        u_map_2 = {0: 2, 1: 3, 2: 6, 3: 7}
        # B1,j: B[2:4, 2j:2j+2]
        v_map_2 = {0: 16+2*j, 1: 16+2*j+1, 2: 24+2*j, 3: 24+2*j+1}
        # C block j: same as above
        w_map_2 = {0: 4*j, 1: 4*j+2, 2: 4*j+1, 3: 4*j+3}
        
        add_product(u_map_2, v_map_2, w_map_2)
        
    return {"U": U, "V": V, "W": W}
