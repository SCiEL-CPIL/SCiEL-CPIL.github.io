import time

def construct():
    # Hybrid Strassen (2x2) + Standard (rest) decomposition
    # Rank 26
    
    n, m, p = 3, 3, 3
    
    # We will build U, V, W lists
    # Each term is a rank-1 tensor: u * v * w
    # U shape (9, rank), V shape (9, rank), W shape (9, rank)
    
    terms = []
    
    # Helper to add a term
    # u_indices: dict { (i,j): coeff } for A
    # v_indices: dict { (j,k): coeff } for B
    # w_indices: dict { (k,i): coeff } for C (output index k*n+i? No, W[k*n+i] corresponds to C[i,k]? 
    # Task says: W[k*n+i, r]. 
    # Standard matrix mult: C[i,k] = sum_j A[i,j] B[j,k].
    # Tensor T[i,j,k,l] = delta_{i,j} delta_{k,l}? No.
    # T_{i,j,k} = A[i,j] B[j,k] -> C[i,k].
    # The output factor W has indices [k*n + i, r].
    # So W[k*n+i] corresponds to C[i,k].
    # Let's verify the convention.
    # "output factor follows the official trace convention W[k*n+i,r]"
    # Usually C[i,k] is stored. If W index is k*n+i, then W[row, col] = C[col, row]?
    # Let's assume W[k*n+i] contributes to C[i,k].
    
    def add_term(u_dict, v_dict, w_dict):
        # u_dict: {(i,j): coeff}
        # v_dict: {(j,k): coeff}
        # w_dict: {(i,k): coeff} -> stored at k*n+i
        terms.append((u_dict, v_dict, w_dict))

    # 1. Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A indices: (0,0), (0,1), (1,0), (1,1)
    # B indices: (0,0), (0,1), (1,0), (1,1)
    # C indices: (0,0), (0,1), (1,0), (1,1)
    
    # M1 = (A00+A11)(B00+B11)
    u1 = {(0,0): 1, (1,1): 1}
    v1 = {(0,0): 1, (1,1): 1}
    # Contribution to C00, C01, C10, C11
    # C00 += M1
    # C11 += M1
    # Wait, Strassen formulas:
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to add these to the terms.
    # Since M1 is a product of sums, it expands to 4 terms?
    # No, in tensor decomposition, a "term" is a product of linear forms.
    # (A00+A11) is a linear form in A.
    # (B00+B11) is a linear form in B.
    # So M1 is a single rank-1 term in the decomposition of the bilinear form?
    # Yes, $u = (1, 0, 1, 0)^T$ (flattened A), $v = (1, 0, 1, 0)^T$ (flattened B).
    # The output is a linear combination of C entries.
    
    # Let's define the linear forms for Strassen.
    # A_vec indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B_vec indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # M1: A: [1, 0, 1, 0], B: [1, 0, 1, 0]
    # Output: C00: 1, C11: 1
    # Wait, C00 and C11 both get M1.
    # So w_dict: {(0,0): 1, (1,1): 1}
    
    add_term({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1})
    
    # M2: (A10+A11) B00
    # A: [0, 0, 1, 1], B: [1, 0, 0, 0]
    # C10: 1, C11: -1
    add_term({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1})
    
    # M3: A00 (B10-B11)
    # A: [1, 0, 0, 0], B: [0, -1, 1, 0]
    # C01: 1, C11: 1
    add_term({(0,0): 1}, {(1,0): 1, (1,1): -1}, {(0,1): 1, (1,1): 1})
    
    # M4: A11 (B01-B00)
    # A: [0, 0, 0, 1], B: [-1, 1, 0, 0]
    # C00: -1, C10: 1
    add_term({(1,1): 1}, {(0,0): -1, (0,1): 1}, {(0,0): -1, (1,0): 1})
    
    # M5: (A00+A01) B11
    # A: [1, 1, 0, 0], B: [0, 0, 0, 1]
    # C00: 1, C01: 1
    add_term({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): 1, (0,1): 1})
    
    # M6: (A10-A00) (B00+B01)
    # A: [-1, 0, 1, 0], B: [1, 1, 0, 0]
    # C11: 1
    add_term({(0,0): -1, (1,0): 1}, {(0,0): 1, (0,1): 1}, {(1,1): 1})
    
    # M7: (A01-A11) (B10+B11)
    # A: [0, 1, 0, -1], B: [0, 0, 1, 1]
    # C00: -1
    add_term({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): -1})
    
    # 2. Cross terms
    
    # a e^T: A[0:2, 2] * B[2, 0:2]
    # A02 B20 -> C00
    add_term({(0,2): 1}, {(2,0): 1}, {(0,0): 1})
    # A02 B21 -> C01
    add_term({(0,2): 1}, {(2,1): 1}, {(0,1): 1})
    # A12 B20 -> C10
    add_term({(1,2): 1}, {(2,0): 1}, {(1,0): 1})
    # A12 B21 -> C11
    add_term({(1,2): 1}, {(2,1): 1}, {(1,1): 1})
    
    # A_TL d: A[0:2, 0:2] * B[0:2, 2]
    # A00 B02 -> C02
    add_term({(0,0): 1}, {(0,2): 1}, {(0,2): 1})
    # A01 B12 -> C02
    add_term({(0,1): 1}, {(1,2): 1}, {(0,2): 1})
    # A10 B02 -> C12
    add_term({(1,0): 1}, {(0,2): 1}, {(1,2): 1})
    # A11 B12 -> C12
    add_term({(1,1): 1}, {(1,2): 1}, {(1,2): 1})
    
    # b^T B_TL: A[2, 0:2] * B[0:2, 0:2]
    # A20 B00 -> C20
    add_term({(2,0): 1}, {(0,0): 1}, {(2,0): 1})
    # A20 B01 -> C21
    add_term({(2,0): 1}, {(0,1): 1}, {(2,1): 1})
    # A21 B10 -> C20
    add_term({(2,1): 1}, {(1,0): 1}, {(2,0): 1})
    # A21 B11 -> C21
    add_term({(2,1): 1}, {(1,1): 1}, {(2,1): 1})
    
    # a f: A[0:2, 2] * B[2, 2]
    # A02 B22 -> C02
    add_term({(0,2): 1}, {(2,2): 1}, {(0,2): 1})
    # A12 B22 -> C12
    add_term({(1,2): 1}, {(2,2): 1}, {(1,2): 1})
    
    # c e^T: A[2, 2] * B[2, 0:2]
    # A22 B20 -> C20
    add_term({(2,2): 1}, {(2,0): 1}, {(2,0): 1})
    # A22 B21 -> C21
    add_term({(2,2): 1}, {(2,1): 1}, {(2,1): 1})
    
    # b^T d: A[2, 0:2] * B[0:2, 2]
    # A20 B02 -> C22
    add_term({(2,0): 1}, {(0,2): 1}, {(2,2): 1})
    # A21 B12 -> C22
    add_term({(2,1): 1}, {(1,2): 1}, {(2,2): 1})
    
    # c f: A[2, 2] * B[2, 2]
    # A22 B22 -> C22
    add_term({(2,2): 1}, {(2,2): 1}, {(2,2): 1})
    
    rank = len(terms)
    
    U = [[0.0] * rank for _ in range(9)]
    V = [[0.0] * rank for _ in range(9)]
    W = [[0.0] * rank for _ in range(9)]
    
    for r, (u_dict, v_dict, w_dict) in enumerate(terms):
        for (i, j), val in u_dict.items():
            U[i * 3 + j][r] = val
        for (j, k), val in v_dict.items():
            V[j * 3 + k][r] = val
        for (i, k), val in w_dict.items():
            # W index is k*n + i
            W[k * 3 + i][r] = val
            
    return {"U": U, "V": V, "W": W}
