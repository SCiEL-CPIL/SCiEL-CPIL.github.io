import numpy as np

DIMENSIONS = [2, 2, 7]

def construct():
    n, m, p = DIMENSIONS
    # U rows: i*m + j
    # V rows: j*p + k
    # W rows: k*n + i
    
    rank = 25
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def add_term(u_idx_coeff, v_idx_coeff, w_idx_coeff, r):
        for idx, coeff in u_idx_coeff:
            U[idx, r] += coeff
        for idx, coeff in v_idx_coeff:
            V[idx, r] += coeff
        for idx, coeff in w_idx_coeff:
            W[idx, r] += coeff
            
    r = 0
    
    # Strassen blocks for k in {0,1}, {2,3}, {4,5}
    for b in range(3):
        k0 = 2*b
        k1 = 2*b+1
        
        # M1: (A00+A11)(B0k0+B1k1) -> C0k0 + C1k1
        # A: 00 (0), 11 (3)
        # B: 0k0 (0*7+k0), 1k1 (1*7+k1)
        # C: 0k0 (k0*2+0), 1k1 (k1*2+1)
        add_term(
            [(0, 1), (3, 1)],
            [(k0, 1), (7+k1, 1)],
            [(k0*2, 1), (k1*2+1, 1)],
            r
        )
        r += 1
        
        # M2: (A10+A11)B0k0 -> C1k0
        # A: 10 (2), 11 (3)
        # B: 0k0 (k0)
        # C: 1k0 (k0*2+1)
        add_term(
            [(2, 1), (3, 1)],
            [(k0, 1)],
            [(k0*2+1, 1)],
            r
        )
        r += 1
        
        # M3: A00(B1k0-B1k1) -> C0k1
        # A: 00 (0)
        # B: 1k0 (7+k0), 1k1 (7+k1)
        # C: 0k1 (k1*2)
        add_term(
            [(0, 1)],
            [(7+k0, 1), (7+k1, -1)],
            [(k1*2, 1)],
            r
        )
        r += 1
        
        # M4: A11(B0k0-B0k1) -> C0k0 + C1k0
        # A: 11 (3)
        # B: 0k0 (k0), 0k1 (k1)
        # C: 0k0 (k0*2), 1k0 (k0*2+1)
        add_term(
            [(3, 1)],
            [(k0, 1), (k1, -1)],
            [(k0*2, 1), (k0*2+1, 1)],
            r
        )
        r += 1
        
        # M5: (A00+A01)B1k1 -> C0k1 - C0k0
        # A: 00 (0), 01 (1)
        # B: 1k1 (7+k1)
        # C: 0k1 (k1*2), 0k0 (k0*2)
        add_term(
            [(0, 1), (1, 1)],
            [(7+k1, 1)],
            [(k1*2, 1), (k0*2, -1)],
            r
        )
        r += 1
        
        # M6: (A10-A00)(B0k0+B0k1) -> C1k1
        # A: 10 (2), 00 (0)
        # B: 0k0 (k0), 0k1 (k1)
        # C: 1k1 (k1*2+1)
        add_term(
            [(2, 1), (0, -1)],
            [(k0, 1), (k1, 1)],
            [(k1*2+1, 1)],
            r
        )
        r += 1
        
        # M7: (A01-A11)(B1k0+B1k1) -> C0k0 + C1k1
        # A: 01 (1), 11 (3)
        # B: 1k0 (7+k0), 1k1 (7+k1)
        # C: 0k0 (k0*2), 1k1 (k1*2+1)
        add_term(
            [(1, 1), (3, -1)],
            [(7+k0, 1), (7+k1, 1)],
            [(k0*2, 1), (k1*2+1, 1)],
            r
        )
        r += 1

    # Block 3: k=6
    k = 6
    # M8: A00 B06 -> C06
    add_term(
        [(0, 1)],
        [(k, 1)],
        [(k*2, 1)],
        r
    )
    r += 1
    
    # M9: A01 B16 -> C06
    add_term(
        [(1, 1)],
        [(7+k, 1)],
        [(k*2, 1)],
        r
    )
    r += 1
    
    # M10: A10 B06 -> C16
    add_term(
        [(2, 1)],
        [(k, 1)],
        [(k*2+1, 1)],
        r
    )
    r += 1
    
    # M11: A11 B16 -> C16
    add_term(
        [(3, 1)],
        [(7+k, 1)],
        [(k*2+1, 1)],
        r
    )
    r += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
