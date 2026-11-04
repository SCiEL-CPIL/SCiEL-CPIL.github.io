import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Total rank
    rank = 78
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Helper to add a schoolbook term for M[n_block, m_block, p_block]
    def add_schoolbook(n_start, m_start, p_start, n_len, m_len, p_len, current_col):
        # Returns new col
        for i in range(n_len):
            for j in range(m_len):
                for k in range(p_len):
                    gi = n_start + i
                    gj = m_start + j
                    gk = p_start + k
                    
                    U[gi * m + gj, current_col] = 1
                    V[gj * p + gk, current_col] = 1
                    W[gk * n + gi, current_col] = 1
                    current_col += 1
        return current_col

    # Helper to add Strassen terms for M[2, 2, 2]
    def add_strassen(n_start, m_start, p_start, current_col):
        # n_len=2, m_len=2, p_len=2
        # Indices:
        # a: (0,0), b: (0,1)
        # c: (1,0), d: (1,1)
        # e: (0,0), f: (0,1)
        # g: (1,0), h: (1,1)
        
        # P1 = (a+d)(e+h)
        # U: a=1, d=1. V: e=1, h=1. W: C11+=1, C12+=0, C21+=0, C22+=1?
        # Wait, Strassen formulas:
        # C11 = P1 + P4 - P5 + P7
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 + P3 + P6
        
        # We need to decompose each P into a rank-1 tensor?
        # No, each P is a product of a linear form in A and a linear form in B.
        # So each P corresponds to one column in the decomposition.
        
        # P1:
        # A_lin: a=1, d=1. (Indices (0,0), (1,1))
        # B_lin: e=1, h=1. (Indices (0,0), (1,1))
        # C_lin: C11=1, C22=1. (Indices (0,0), (1,1) in output C)
        # Output C indices:
        # C11 is at k=0, i=0 -> W[0*2 + 0] = W[0]
        # C12 is at k=1, i=0 -> W[1*2 + 0] = W[2]
        # C21 is at k=0, i=1 -> W[0*2 + 1] = W[1]
        # C22 is at k=1, i=1 -> W[1*2 + 1] = W[3]
        
        def add_term(a_coeffs, b_coeffs, c_coeffs, current_col):
            # a_coeffs: dict or list of (i,j, coeff)
            # b_coeffs: dict or list of (j,k, coeff)
            # c_coeffs: dict or list of (k,i, coeff)
            for (i, j, coeff) in a_coeffs:
                U[(n_start + i) * m + (m_start + j), current_col] = coeff
            for (j, k, coeff) in b_coeffs:
                V[(m_start + j) * p + (p_start + k), current_col] = coeff
            for (k, i, coeff) in c_coeffs:
                W[(p_start + k) * n + (n_start + i), current_col] = coeff
            return current_col + 1
        
        # P1
        current_col = add_term(
            [(0,0,1), (1,1,1)],
            [(0,0,1), (1,1,1)],
            [(0,0,1), (1,1,1)], # C11, C22
            current_col
        )
        
        # P2 = (c+d)e
        # A: c=1, d=1. B: e=1. C: C21=1, C22=-1
        current_col = add_term(
            [(1,0,1), (1,1,1)],
            [(0,0,1)],
            [(0,1,1), (1,1,-1)], # C21, C22
            current_col
        )
        
        # P3 = a(f-h)
        # A: a=1. B: f=1, h=-1. C: C12=1, C22=1
        current_col = add_term(
            [(0,0,1)],
            [(0,1,1), (1,1,-1)],
            [(1,0,1), (1,1,1)], # C12, C22
            current_col
        )
        
        # P4 = d(g-f)
        # A: d=1. B: g=1, f=-1. C: C11=1, C21=1
        current_col = add_term(
            [(1,1,1)],
            [(1,0,1), (0,1,-1)],
            [(0,0,1), (0,1,1)], # C11, C21
            current_col
        )
        
        # P5 = (a+b)h
        # A: a=1, b=1. B: h=1. C: C11=-1, C12=1
        current_col = add_term(
            [(0,0,1), (0,1,1)],
            [(1,1,1)],
            [(0,0,-1), (1,0,1)], # C11, C12
            current_col
        )
        
        # P6 = (c-a)(e+f)
        # A: c=1, a=-1. B: e=1, f=1. C: C22=1
        current_col = add_term(
            [(1,0,1), (0,0,-1)],
            [(0,0,1), (0,1,1)],
            [(1,1,1)], # C22
            current_col
        )
        
        # P7 = (b-d)(g+h)
        # A: b=1, d=-1. B: g=1, h=1. C: C11=1
        current_col = add_term(
            [(0,1,1), (1,1,-1)],
            [(1,0,1), (1,1,1)],
            [(0,0,1)], # C11
            current_col
        )
        
        return current_col

    # 
    # Block 1: m in [0, 2)
    # 
    # Sub-block 1.1: n in [0, 1) -> M[1, 2, 7] Schoolbook
    col = add_schoolbook(0, 0, 0, 1, 2, 7, col) # 1*2*7 = 14 terms
    
    # Sub-block 1.2: n in [1, 3) -> M[2, 2, 7]
    # Split p into 2, 2, 2, 1
    # p in [0, 2) -> M[2, 2, 2] Strassen
    col = add_strassen(1, 0, 0, col) # 7 terms
    # p in [2, 4) -> M[2, 2, 2] Strassen
    col = add_strassen(1, 0, 2, col) # 7 terms
    # p in [4, 6) -> M[2, 2, 2] Strassen
    col = add_strassen(1, 0, 4, col) # 7 terms
    # p in [6, 7) -> M[2, 2, 1] Schoolbook
    col = add_schoolbook(1, 0, 6, 2, 2, 1, col) # 2*2*1 = 4 terms
    
    # 
    # Block 2: m in [2, 4)
    # 
    # Sub-block 2.1: n in [0, 1) -> M[1, 2, 7] Schoolbook
    # Note: m_start is 2
    col = add_schoolbook(0, 2, 0, 1, 2, 7, col) # 14 terms
    
    # Sub-block 2.2: n in [1, 3) -> M[2, 2, 7]
    # p in [0, 2)
    col = add_strassen(1, 2, 0, col) # 7 terms
    # p in [2, 4)
    col = add_strassen(1, 2, 2, col) # 7 terms
    # p in [4, 6)
    col = add_strassen(1, 2, 4, col) # 7 terms
    # p in [6, 7)
    col = add_schoolbook(1, 2, 6, 2, 2, 1, col) # 4 terms
    
    if col != rank:
        # This should not happen if calculations are correct
        # 14 + 7+7+7+4 + 14 + 7+7+7+4 = 14 + 25 + 14 + 25 = 78
        pass

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
