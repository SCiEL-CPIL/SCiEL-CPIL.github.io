import numpy as np

def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    Clean implementation of the 49 terms.
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Indices for A (16), B (16), C (16)
    # A[i, j] -> i*4 + j
    # B[i, j] -> i*4 + j
    # C[i, j] -> j*4 + i (Trace convention W[k*n+i] with k=col, i=row)
    
    def a_idx(i, j): return i * 4 + j
    def b_idx(i, j): return i * 4 + j
    def c_idx(i, j): return j * 4 + i

    # Linear form representation: list of (idx, coeff)
    
    def add_form(f1, f2):
        d = {}
        for i, c in f1: d[i] = d.get(i, 0) + c
        for i, c in f2: d[i] = d.get(i, 0) + c
        return [(i, c) for i, c in d.items() if c != 0]

    def sub_form(f1, f2):
        d = {}
        for i, c in f1: d[i] = d.get(i, 0) + c
        for i, c in f2: d[i] = d.get(i, 0) - c
        return [(i, c) for i, c in d.items() if c != 0]

    # Get linear form for A block (br, bc)
    # A_block[br, bc] is 2x2.
    # L1 is 2x2. L1[i, j] = A[2*br+i, 2*bc+j].
    
    def get_a_block_form(br, bc):
        # Returns a 2x2 list of linear forms
        forms = []
        for i in range(2):
            row = []
            for j in range(2):
                idx = a_idx(2*br+i, 2*bc+j)
                row.append([(idx, 1)])
            forms.append(row)
        return forms

    def get_b_block_form(br, bc):
        forms = []
        for i in range(2):
            row = []
            for j in range(2):
                idx = b_idx(2*br+i, 2*bc+j)
                row.append([(idx, 1)])
            forms.append(row)
        return forms

    def add_2x2_forms(M1, M2):
        return [[add_form(M1[i][j], M2[i][j]) for j in range(2)] for i in range(2)]

    def sub_2x2_forms(M1, M2):
        return [[sub_form(M1[i][j], M2[i][j]) for j in range(2)] for i in range(2)]

    # Block terms
    A00 = get_a_block_form(0, 0)
    A01 = get_a_block_form(0, 1)
    A10 = get_a_block_form(1, 0)
    A11 = get_a_block_form(1, 1)
    
    B00 = get_b_block_form(0, 0)
    B01 = get_b_block_form(0, 1)
    B10 = get_b_block_form(1, 0)
    B11 = get_b_block_form(1, 1)

    # 7 Block Terms (L, R)
    # L and R are 2x2 matrices of linear forms.
    
    # Term 0: (A00+A11)(B00+B11)
    L0 = add_2x2_forms(A00, A11)
    R0 = add_2x2_forms(B00, B11)
    
    # Term 1: (A10+A11)B00
    L1 = add_2x2_forms(A10, A11)
    R1 = B00
    
    # Term 2: A00(B01-B11)
    L2 = A00
    R2 = sub_2x2_forms(B01, B11)
    
    # Term 3: A11(B10-B00)
    L3 = A11
    R3 = sub_2x2_forms(B10, B00)
    
    # Term 4: (A00+A01)B11
    L4 = add_2x2_forms(A00, A01)
    R4 = B11
    
    # Term 5: (A10-A00)(B00+B01)
    L5 = sub_2x2_forms(A10, A00)
    R5 = add_2x2_forms(B00, B01)
    
    # Term 6: (A01-A11)(B10+B11)
    L6 = sub_2x2_forms(A01, A11)
    R6 = add_2x2_forms(B10, B11)

    block_terms = [
        (L0, R0),
        (L1, R1),
        (L2, R2),
        (L3, R3),
        (L4, R4),
        (L5, R5),
        (L6, R6)
    ]

    # Block contributions to C
    # C = [[C00, C01], [C10, C11]]
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    block_contribs = [
        [(0, 0, 1), (1, 1, 1)], # M1
        [(1, 0, 1), (1, 1, -1)], # M2
        [(0, 1, 1), (1, 1, 1)], # M3
        [(0, 0, 1), (1, 0, 1)], # M4
        [(0, 0, -1), (0, 1, 1)], # M5
        [(1, 1, 1)], # M6
        [(0, 0, 1)] # M7
    ]

    # Scalar Strassen contributions within a 2x2 multiplication
    # P = [[P00, P01], [P10, P11]]
    # P00 = S1 + S4 - S5 + S7
    # P01 = S3 + S5
    # P10 = S2 + S4
    # P11 = S1 - S2 + S3 + S6
    
    scalar_contribs = [
        [(0, 0, 1), (1, 1, 1)], # S1
        [(1, 0, 1), (1, 1, -1)], # S2
        [(0, 1, 1), (1, 1, 1)], # S3
        [(0, 0, 1), (1, 0, 1)], # S4
        [(0, 0, -1), (0, 1, 1)], # S5
        [(1, 1, 1)], # S6
        [(0, 0, 1)] # S7
    ]

    # Scalar Strassen formulas for 2x2 matrices L, R
    # S1 = (L00+L11)(R00+R11)
    # S2 = (L10+L11)R00
    # S3 = L00(R01-R11)
    # S4 = L11(R10-R00)
    # S5 = (L00+L01)R11
    # S6 = (L10-L00)(R00+R01)
    # S7 = (L01-L11)(R10+R11)
    
    for t in range(7):
        L, R = block_terms[t]
        b_contrib = block_contribs[t]
        
        # Extract L00, L01, L10, L11, R00, R01, R10, R11
        # These are 2x2 matrices of linear forms?
        # No, L is a 2x2 matrix of linear forms.
        # L[0][0] is a linear form (list of (idx, coeff)).
        
        L00, L01 = L[0]
        L10, L11 = L[1]
        R00, R01 = R[0]
        R10, R11 = R[1]
        
        # Define the 7 scalar products
        # Each is (FormX, FormY)
        
        s1_x = add_form(L00, L11)
        s1_y = add_form(R00, R11)
        
        s2_x = add_form(L10, L11)
        s2_y = R00
        
        s3_x = L00
        s3_y = sub_form(R01, R11)
        
        s4_x = L11
        s4_y = sub_form(R10, R00)
        
        s5_x = add_form(L00, L01)
        s5_y = R11
        
        s6_x = sub_form(L10, L00)
        s6_y = add_form(R00, R01)
        
        s7_x = sub_form(L01, L11)
        s7_y = add_form(R10, R11)
        
        scalar_products = [
            (s1_x, s1_y),
            (s2_x, s2_y),
            (s3_x, s3_y),
            (s4_x, s4_y),
            (s5_x, s5_y),
            (s6_x, s6_y),
            (s7_x, s7_y)
        ]
        
        for k in range(7):
            rank_idx = t * 7 + k
            x_form, y_form = scalar_products[k]
            s_contrib = scalar_contribs[k]
            
            # Update U
            for idx, coeff in x_form:
                U[idx, rank_idx] += coeff
                
            # Update V
            for idx, coeff in y_form:
                V[idx, rank_idx] += coeff
                
            # Update W
            # For each contribution in s_contrib (pr, pc, s_coeff)
            # and each contribution in b_contrib (br, bc, b_coeff)
            # The global C index is (2*br + pr, 2*bc + pc)
            for pr, pc, s_coeff in s_contrib:
                for br, bc, b_coeff in b_contrib:
                    out_row = 2 * br + pr
                    out_col = 2 * bc + pc
                    total_coeff = s_coeff * b_coeff
                    W[c_idx(out_row, out_col), rank_idx] += total_coeff

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
