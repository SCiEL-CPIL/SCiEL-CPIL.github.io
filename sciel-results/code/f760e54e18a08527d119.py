import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x3 matrix by a 3x9 matrix.
    
    Returns: dict with 'U', 'V', 'W' factor matrices
    """
    n, m, p = DIMENSIONS  # 2, 3, 9
    
    # We will collect terms as triples (u_idx, v_idx, w_idx, coeff)
    # U has size n*m = 6
    # V has size m*p = 27
    # W has size p*n = 18
    
    terms = []
    
    def add_term(u, v, w, coeff=1):
        if coeff != 0:
            terms.append((u, v, w, coeff))
            
    # Indices in U (flattened A, 2x3):
    # A[0,0]=0, A[0,1]=1, A[0,2]=2
    # A[1,0]=3, A[1,1]=4, A[1,2]=5
    
    # Indices in V (flattened B, 3x9):
    # B[i, j] = i * 9 + j
    
    # Indices in W (flattened C, 9x2):
    # C[j, i] = j * 2 + i
    
    # Part 1: Contribution from A[:, 0:2] and B[0:2, :]
    # A_sub is 2x2, B_sub is 2x9
    # We split B_sub into 4 blocks of 2 columns and 1 block of 1 column.
    
    # Strassen coefficients for 2x2 * 2x2
    # Let A_sub = [[a, b], [c, d]]
    # Let B_block = [[e, f], [g, h]] (columns k, k+1)
    # C_block = [[C00, C01], [C10, C11]]
    
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    u_a = 0  # A[0,0]
    u_b = 1  # A[0,1]
    u_c = 3  # A[1,0]
    u_d = 4  # A[1,1]
    
    for b_idx in range(4):
        k0 = 2 * b_idx
        k1 = 2 * b_idx + 1
        
        v_e = 0 * 9 + k0  # B[0, k0]
        v_f = 0 * 9 + k1  # B[0, k1]
        v_g = 1 * 9 + k0  # B[1, k0]
        v_h = 1 * 9 + k1  # B[1, k1]
        
        w_00 = k0 * 2 + 0
        w_01 = k1 * 2 + 0
        w_10 = k0 * 2 + 1
        w_11 = k1 * 2 + 1
        
        # M1: (a+d)(e+h)
        add_term(u_a, v_e, w_00, 1)
        add_term(u_d, v_e, w_00, 1)
        add_term(u_a, v_h, w_00, 1)
        add_term(u_d, v_h, w_00, 1)
        
        add_term(u_a, v_e, w_11, 1)
        add_term(u_d, v_e, w_11, 1)
        add_term(u_a, v_h, w_11, 1)
        add_term(u_d, v_h, w_11, 1)
        
        # M2: (c+d)e
        add_term(u_c, v_e, w_10, 1)
        add_term(u_d, v_e, w_10, 1)
        
        add_term(u_c, v_e, w_11, -1)
        add_term(u_d, v_e, w_11, -1)
        
        # M3: a(f-h)
        add_term(u_a, v_f, w_01, 1)
        add_term(u_a, v_h, w_01, -1)
        
        add_term(u_a, v_f, w_11, 1)
        add_term(u_a, v_h, w_11, -1)
        
        # M4: d(g-e)
        add_term(u_d, v_g, w_00, 1)
        add_term(u_d, v_e, w_00, -1)
        
        add_term(u_d, v_g, w_10, 1)
        add_term(u_d, v_e, w_10, -1)
        
        # M5: (a+b)h
        add_term(u_a, v_h, w_00, -1)
        add_term(u_b, v_h, w_00, -1)
        
        add_term(u_a, v_h, w_01, 1)
        add_term(u_b, v_h, w_01, 1)
        
        # M6: (c-a)(e+f)
        add_term(u_c, v_e, w_11, 1)
        add_term(u_a, v_e, w_11, -1)
        add_term(u_c, v_f, w_11, 1)
        add_term(u_a, v_f, w_11, -1)
        
        # M7: (b-d)(g+h)
        add_term(u_b, v_g, w_00, 1)
        add_term(u_d, v_g, w_00, -1)
        add_term(u_b, v_h, w_00, 1)
        add_term(u_d, v_h, w_00, -1)

    # Last block: 2x2 * 2x1 (column 8)
    k8 = 8
    v_e = 0 * 9 + k8
    v_g = 1 * 9 + k8
    
    w_08 = k8 * 2 + 0
    w_18 = k8 * 2 + 1
    
    # Direct multiplication for 2x2 * 2x1
    # C00 = a*e
    add_term(u_a, v_e, w_08, 1)
    # C01 = b*g (Wait, B is 2x1, so it's a vector [e, g]. A is 2x2. 
    # C = [[a, b], [c, d]] * [[e], [g]] = [[a*e + b*g], [c*e + d*g]]
    # So C[0,8] = a*e + b*g
    # C[1,8] = c*e + d*g
    
    add_term(u_b, v_g, w_08, 1)
    add_term(u_c, v_e, w_18, 1)
    add_term(u_d, v_g, w_18, 1)
    
    # Part 2: Contribution from A[:, 2] and B[2, :]
    # A_vec is 2x1 (column 2), B_vec is 1x9 (row 2)
    # C += A_vec * B_vec (outer product)
    # C[i, j] += A[i, 2] * B[2, j]
    
    u_a2 = 2  # A[0, 2]
    u_a1 = 5  # A[1, 2]
    
    for j in range(9):
        v_b2 = 2 * 9 + j  # B[2, j]
        
        # C[0, j] += A[0,2] * B[2,j]
        add_term(u_a2, v_b2, j * 2 + 0, 1)
        
        # C[1, j] += A[1,2] * B[2,j]
        add_term(u_a1, v_b2, j * 2 + 1, 1)
        
    # Determine rank
    rank = len(terms)
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=np.int32)
    V = np.zeros((m * p, rank), dtype=np.int32)
    W = np.zeros((p * n, rank), dtype=np.int32)
    
    # Fill matrices
    for i, (u, v, w, coeff) in enumerate(terms):
        U[u][i] += coeff
        V[v][i] += coeff
        W[w][i] += coeff
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", res["U"][0].__len__() if isinstance(res["U"][0], list) else len(res["U"]))
