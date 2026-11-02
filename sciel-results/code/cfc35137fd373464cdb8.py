import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # We will use rank 28
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper to add a term
    def add_term(u_coeffs, v_coeffs, w_coeffs, i0, j0, k0, n_b, m_b, p_b):
        nonlocal col
        # u_coeffs: dict or list of (li, lj, coeff)
        # v_coeffs: dict or list of (lj, lk, coeff)
        # w_coeffs: dict or list of (lk, li, coeff)
        
        for li, lj, cu in u_coeffs:
            gi = i0 + li
            gj = j0 + lj
            U[gi * m + gj, col] += cu
            
        for lj, lk, cv in v_coeffs:
            gj = j0 + lj
            gk = k0 + lk
            V[gj * p + gk, col] += cv
            
        for lk, li, cw in w_coeffs:
            gk = k0 + lk
            gi = i0 + li
            W[gk * n + gi, col] += cw
            
        col += 1

    # Strassen 2x2x2
    def add_strassen(i0, j0, k0):
        # Terms defined above
        terms = [
            # 1
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
            # 2
            ([(1,0,1), (1,1,1)], [(0,0,1)], [(0,1,1), (1,1,1)]),
            # 3
            ([(0,0,1)], [(0,1,1), (1,1,-1)], [(1,0,1)]),
            # 4
            ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),
            # 5
            ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,1), (1,0,-1)]),
            # 6
            ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
            # 7
            ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1), (1,1,1)])
        ]
        for u, v, w in terms:
            add_term(u, v, w, i0, j0, k0, 2, 2, 2)

    # Schoolbook 2x2x1
    def add_221(i0, j0, k0):
        terms = [
            # a00 b00 -> c00
            ([(0,0,1)], [(0,0,1)], [(0,0,1)]),
            # a01 b10 -> c00
            ([(0,1,1)], [(1,0,1)], [(0,0,1)]),
            # a10 b00 -> c10
            ([(1,0,1)], [(0,0,1)], [(0,1,1)]),
            # a11 b10 -> c10
            ([(1,1,1)], [(1,0,1)], [(0,1,1)])
        ]
        for u, v, w in terms:
            add_term(u, v, w, i0, j0, k0, 2, 2, 1)

    # Schoolbook 2x1x5
    def add_215(i0, j0, k0):
        for i in range(2):
            for k in range(5):
                u = [(i, 0, 1)]
                v = [(0, k, 1)]
                w = [(k, i, 1)]
                add_term(u, v, w, i0, j0, k0, 2, 1, 5)

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2]
    add_strassen(0, 0, 0)
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4]
    add_strassen(0, 0, 2)
    
    # Block 3: A[0:2, 0:2] * B[0:2, 4:5]
    add_221(0, 0, 4)
    
    # Block 4: A[0:2, 2:3] * B[2:3, 0:5]
    add_215(0, 2, 0)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
