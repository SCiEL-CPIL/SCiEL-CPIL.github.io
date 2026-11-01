import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # U shape: (n*m, rank) = (6, rank)
    # V shape: (m*p, rank) = (21, rank)
    # W shape: (p*n, rank) = (14, rank)
    
    # We don't know rank in advance, so we'll accumulate terms in lists
    U_terms = []
    V_terms = []
    W_terms = []
    
    def add_term(u_indices, v_indices, w_indices, val=1):
        # u_indices: list of (index, coeff)
        # v_indices: list of (index, coeff)
        # w_indices: list of (index, coeff)
        u_vec = [0.0] * (n * m)
        v_vec = [0.0] * (m * p)
        w_vec = [0.0] * (p * n)
        
        for idx, coeff in u_indices:
            u_vec[idx] = coeff
        for idx, coeff in v_indices:
            v_vec[idx] = coeff
        for idx, coeff in w_indices:
            w_vec[idx] = coeff
            
        U_terms.append(u_vec)
        V_terms.append(v_vec)
        W_terms.append(w_vec)
        
    def add_strassen_block(a_idx, b_idx, c_idx):
        # a_idx: dict mapping '00','01','10','11' to global U index
        # b_idx: dict mapping '00','01','10','11' to global V index
        # c_idx: dict mapping '00','01','10','11' to global W index
        
        # M1 = (A00+A11)(B00+B11)
        add_term([(a_idx['00'], 1), (a_idx['11'], 1)],
                 [(b_idx['00'], 1), (b_idx['11'], 1)],
                 [(c_idx['00'], 1), (c_idx['11'], 1)])
                 
        # M2 = (A10+A11)B00
        add_term([(a_idx['10'], 1), (a_idx['11'], 1)],
                 [(b_idx['00'], 1)],
                 [(c_idx['10'], 1)])
                 
        # M3 = A00(B01-B11)
        add_term([(a_idx['00'], 1)],
                 [(b_idx['01'], 1), (b_idx['11'], -1)],
                 [(c_idx['01'], 1), (c_idx['11'], 1)])
                 
        # M4 = A11(B10-B00)
        add_term([(a_idx['11'], 1)],
                 [(b_idx['10'], 1), (b_idx['00'], -1)],
                 [(c_idx['00'], 1), (c_idx['10'], 1)])
                 
        # M5 = (A00+A01)B11
        add_term([(a_idx['00'], 1), (a_idx['01'], 1)],
                 [(b_idx['11'], 1)],
                 [(c_idx['00'], -1), (c_idx['01'], 1)])
                 
        # M6 = (A10-A00)(B00+B01)
        add_term([(a_idx['10'], 1), (a_idx['00'], -1)],
                 [(b_idx['00'], 1), (b_idx['01'], 1)],
                 [(c_idx['11'], 1)])
                 
        # M7 = (A01-A11)(B10+B11)
        add_term([(a_idx['01'], 1), (a_idx['11'], -1)],
                 [(b_idx['10'], 1), (b_idx['11'], 1)],
                 [(c_idx['00'], 1), (c_idx['11'], 1)])

    # Indices for A (2x3)
    # A[i, j] -> U[i*3 + j]
    A_idx = {
        '00': 0*3+0, '01': 0*3+1, '02': 0*3+2,
        '10': 1*3+0, '11': 1*3+1, '12': 1*3+2
    }
    
    # Indices for B (3x7)
    # B[j, k] -> V[j*7 + k]
    # Indices for C (2x7)
    # C[i, k] -> W[k*2 + i]
    
    # Block 1: A2 (col 2) * B2 (row 2)
    # A2: U indices A_idx['02'], A_idx['12']
    # B2: V indices for j=2, k=0..6
    # C: all W indices
    
    for i in range(2):
        for k in range(7):
            u_idx = A_idx[str(i)+'2']
            v_idx = 2*7 + k
            w_idx = k*2 + i
            add_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])
            
    # Block 2: A1 (cols 0,1) * B1a (rows 0,1; cols 0..3)
    # A1: U indices A_idx['00'], A_idx['01'], A_idx['10'], A_idx['11']
    
    # Sub-block 2a: B1a cols 0,1
    b_idx_2a = {
        '00': 0*7+0, '01': 0*7+1,
        '10': 1*7+0, '11': 1*7+1
    }
    c_idx_2a = {
        '00': 0*2+0, '01': 1*2+0,
        '10': 0*2+1, '11': 1*2+1
    }
    # Wait, C index is k*2 + i.
    # C[0,0] -> 0*2+0 = 0
    # C[0,1] -> 1*2+0 = 2
    # C[1,0] -> 0*2+1 = 1
    # C[1,1] -> 1*2+1 = 3
    
    c_idx_2a = {
        '00': 0, # C[0,0]
        '01': 2, # C[0,1]
        '10': 1, # C[1,0]
        '11': 3  # C[1,1]
    }
    
    a_idx_1 = {
        '00': A_idx['00'], '01': A_idx['01'],
        '10': A_idx['10'], '11': A_idx['11']
    }
    
    add_strassen_block(a_idx_1, b_idx_2a, c_idx_2a)
    
    # Sub-block 2b: B1a cols 2,3
    b_idx_2b = {
        '00': 0*7+2, '01': 0*7+3,
        '10': 1*7+2, '11': 1*7+3
    }
    c_idx_2b = {
        '00': 2*2+0, # C[0,2] = 4
        '01': 3*2+0, # C[0,3] = 6
        '10': 2*2+1, # C[1,2] = 5
        '11': 3*2+1  # C[1,3] = 7
    }
    c_idx_2b = {
        '00': 4,
        '01': 6,
        '10': 5,
        '11': 7
    }
    
    add_strassen_block(a_idx_1, b_idx_2b, c_idx_2b)
    
    # Block 3: A1 (cols 0,1) * B1b (rows 0,1; cols 4..6)
    # Sub-block 3a: B1b cols 4,5
    b_idx_3a = {
        '00': 0*7+4, '01': 0*7+5,
        '10': 1*7+4, '11': 1*7+5
    }
    c_idx_3a = {
        '00': 4*2+0, # C[0,4] = 8
        '01': 5*2+0, # C[0,5] = 10
        '10': 4*2+1, # C[1,4] = 9
        '11': 5*2+1  # C[1,5] = 11
    }
    c_idx_3a = {
        '00': 8,
        '01': 10,
        '10': 9,
        '11': 11
    }
    
    add_strassen_block(a_idx_1, b_idx_3a, c_idx_3a)
    
    # Sub-block 3b: B1b col 6
    # Schoolbook for 2x2 by 2x1
    # A10 B06 -> C16
    # A11 B16 -> C16
    # A00 B06 -> C06
    # A01 B16 -> C06
    
    # C[0,6] = 6*2+0 = 12
    # C[1,6] = 6*2+1 = 13
    
    add_term([(A_idx['00'], 1)], [(0*7+6, 1)], [(12, 1)])
    add_term([(A_idx['01'], 1)], [(1*7+6, 1)], [(12, 1)])
    add_term([(A_idx['10'], 1)], [(0*7+6, 1)], [(13, 1)])
    add_term([(A_idx['11'], 1)], [(1*7+6, 1)], [(13, 1)])
    
    rank = len(U_terms)
    
    U = [term for term in U_terms]
    V = [term for term in V_terms]
    W = [term for term in W_terms]
    
    return {"U": U, "V": V, "W": W}

