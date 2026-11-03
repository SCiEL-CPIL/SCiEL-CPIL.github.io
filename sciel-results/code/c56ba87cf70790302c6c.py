import time
import numpy as np

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=8
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    
    # We'll build the decomposition term by term
    # Each term is a tuple (U_dict, V_dict, W_dict) where each is a dict mapping index->value
    terms = []
    
    # Helper to add a schoolbook term
    def add_schoolbook(i, j, k):
        terms.append({
            'U': {(i*m + j): 1},
            'V': {(j*p + k): 1},
            'W': {(k*n + i): 1}
        })
    
    # Helper to add a Strassen block
    # i0, j0, k0 are the global offsets for the 2x2x2 block
    # Local indices a,b,c in {0,1} map to global i=i0+a, j=j0+b, k=k0+c
    def add_strassen(i0, j0, k0):
        # 7 Strassen terms
        # Each term: (U_coeffs, V_coeffs, W_coeffs)
        # U is indexed by local (a,b) -> global (i0+a, j0+b) -> index (i0+a)*m + (j0+b)
        # V is indexed by local (b,c) -> global (j0+b, k0+c) -> index (j0+b)*p + (k0+c)
        # W is indexed by local (c,a) -> global (k0+c, i0+a) -> index (k0+c)*n + (i0+a)
        
        def U_idx(a, b):
            return (i0+a)*m + (j0+b)
        
        def V_idx(b, c):
            return (j0+b)*p + (k0+c)
        
        def W_idx(c, a):
            return (k0+c)*n + (i0+a)
        
        # M1 = (x00+x11)*(y00+y11) -> z00 += 1, z11 += 1
        terms.append({
            'U': {U_idx(0,0): 1, U_idx(1,1): 1},
            'V': {V_idx(0,0): 1, V_idx(1,1): 1},
            'W': {W_idx(0,0): 1, W_idx(1,1): 1}
        })
        
        # M2 = (x10+x11)*y00 -> z10 += 1, z11 -= 1
        terms.append({
            'U': {U_idx(1,0): 1, U_idx(1,1): 1},
            'V': {V_idx(0,0): 1},
            'W': {W_idx(0,1): 1, W_idx(1,1): -1}
        })
        
        # M3 = x00*(y01-y11) -> z01 += 1, z11 += 1
        terms.append({
            'U': {U_idx(0,0): 1},
            'V': {V_idx(0,1): 1, V_idx(1,1): -1},
            'W': {W_idx(1,0): 1, W_idx(1,1): 1}
        })
        
        # M4 = x11*(y10-y00) -> z00 += 1, z10 += 1
        terms.append({
            'U': {U_idx(1,1): 1},
            'V': {V_idx(1,0): 1, V_idx(0,0): -1},
            'W': {W_idx(0,0): 1, W_idx(0,1): 1}
        })
        
        # M5 = (x00+x01)*y11 -> z00 -= 1, z01 += 1
        terms.append({
            'U': {U_idx(0,0): 1, U_idx(0,1): 1},
            'V': {V_idx(1,1): 1},
            'W': {W_idx(0,0): -1, W_idx(1,0): 1}
        })
        
        # M6 = (x10-x00)*(y00+y01) -> z11 += 1
        terms.append({
            'U': {U_idx(0,0): -1, U_idx(1,0): 1},
            'V': {V_idx(0,0): 1, V_idx(0,1): 1},
            'W': {W_idx(1,1): 1}
        })
        
        # M7 = (x01-x11)*(y10+y11) -> z00 += 1
        terms.append({
            'U': {U_idx(0,1): 1, U_idx(1,1): -1},
            'V': {V_idx(1,0): 1, V_idx(1,1): 1},
            'W': {W_idx(0,0): 1}
        })
    
    # Schoolbook blocks
    # 1. A11*B11: i=0, j=0, k=0..3
    for k in range(4):
        add_schoolbook(0, 0, k)
    
    # 2. A12*B21: i=0, j=1,2, k=0..3
    for j in range(1, 3):
        for k in range(4):
            add_schoolbook(0, j, k)
    
    # 3. A11*B12: i=0, j=0, k=4..7
    for k in range(4, 8):
        add_schoolbook(0, 0, k)
    
    # 4. A12*B22: i=0, j=1,2, k=4..7
    for j in range(1, 3):
        for k in range(4, 8):
            add_schoolbook(0, j, k)
    
    # 5. A21*B11: i=1,2, j=0, k=0..3
    for i in range(1, 3):
        for k in range(4):
            add_schoolbook(i, 0, k)
    
    # 6. A21*B12: i=1,2, j=0, k=4..7
    for i in range(1, 3):
        for k in range(4, 8):
            add_schoolbook(i, 0, k)
    
    # Strassen blocks
    # 7. A22*B21a: i0=1, j0=1, k0=0 (2x2x2)
    add_strassen(1, 1, 0)
    
    # 8. A22*B21b: i0=1, j0=1, k0=2 (2x2x2)
    add_strassen(1, 1, 2)
    
    # 9. A22*B22a: i0=1, j0=1, k0=4 (2x2x2)
    add_strassen(1, 1, 4)
    
    # 10. A22*B22b: i0=1, j0=1, k0=6 (2x2x2)
    add_strassen(1, 1, 6)
    
    # Now build U, V, W matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, term in enumerate(terms):
        for idx, val in term['U'].items():
            U[idx][r] = val
        for idx, val in term['V'].items():
            V[idx][r] = val
        for idx, val in term['W'].items():
            W[idx][r] = val
    
    return {"U": U, "V": V, "W": W}
