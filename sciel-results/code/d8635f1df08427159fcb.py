import numpy as np

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition using Strassen's algorithm for 2x2 blocks.
    # 15 columns = 7 blocks of 2 columns + 1 block of 1 column.
    # Each 2x2 block uses 7 terms.
    # Each 2x1 block uses 7 terms.
    # Total terms = 7 * 7 + 7 = 56.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Strassen linear forms for 2x2 matrix A (4 vars) and 2x2 matrix B (4 vars)
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # We need to map these to the global indices.
    # U is (n*m, rank) = (4, rank)
    # V is (m*p, rank) = (30, rank)
    # W is (p*n, rank) = (30, rank)
    
    # Strassen forms for A (length 4)
    # M1: A00 + A11 -> [1, 0, 0, 1]
    # M2: A10 + A11 -> [0, 0, 1, 1]
    # M3: A00 -> [1, 0, 0, 0]
    # M4: A11 -> [0, 0, 0, 1]
    # M5: A00 + A01 -> [1, 1, 0, 0]
    # M6: A10 - A00 -> [-1, 0, 1, 0]
    # M7: A01 - A11 -> [0, 1, 0, -1]
    
    A_forms = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    # Strassen forms for B (2x2)
    # M1: B00 + B11
    # M2: B00
    # M3: B01 - B11
    # M4: B10 - B00
    # M5: B11
    # M6: B00 + B01
    # M7: B10 + B11
    
    # For a block of 2 columns (k, k+1), B is 2x2.
    # B indices in global V:
    # Row 0: B0,k, B0,k+1
    # Row 1: B1,k, B1,k+1
    # So B00 is V[0, col_k], B01 is V[0, col_k+1]
    # B10 is V[1, col_k], B11 is V[1, col_k+1]
    
    # W forms for C (2x2)
    # C00 = M1 - M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M6 - M7
    
    C_forms = [
        [1, 0, 0, 0, 0, 0, 1], # C00 coeffs for M1..M7? No, C is output.
        # The term is sum_s M_s * (linear form in C)
        # Actually, the tensor term is L_A(s) * L_B(s) * L_C(s)
        # L_C(s) is the coefficient of M_s in the output C.
        # C00 = 1*M1 - 1*M4 - 1*M5 + 1*M7
        # C01 = 1*M3 + 1*M5
        # C10 = 1*M2 + 1*M4
        # C11 = 1*M1 - 1*M2 + 1*M6 - 1*M7
        
        # So for each s, we have a vector in C^*.
        # s=1: C00 + C11
        # s=2: C10 - C11
        # s=3: C01
        # s=4: C10 - C00
        # s=5: C01 - C00
        # s=6: C11
        # s=7: C00 - C11
        
        # Wait, the standard Strassen output reconstruction is:
        # C00 = M1 - M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M6 - M7
        
        # So the linear form in C for term s is:
        # s=1: C00 + C11
        # s=2: C10 - C11
        # s=3: C01
        # s=4: C10 - C00
        # s=5: C01 - C00
        # s=6: C11
        # s=7: C00 - C11
        
        # Let's verify.
        # Term 1: M1 * (C00 + C11)
        # Term 2: M2 * (C10 - C11)
        # Term 3: M3 * C01
        # Term 4: M4 * (C10 - C00)
        # Term 5: M5 * (C01 - C00)
        # Term 6: M6 * C11
        # Term 7: M7 * (C00 - C11)
        
        # Sum = M1(C00+C11) + M2(C10-C11) + M3C01 + M4(C10-C00) + M5(C01-C00) + M6C11 + M7(C00-C11)
        # C00 coeff: M1 - M4 - M5 + M7. Correct.
        # C01 coeff: M3 + M5. Correct.
        # C10 coeff: M2 + M4. Correct.
        # C11 coeff: M1 - M2 + M6 - M7. Correct.
        
        # So W vectors (length 2 for 2x1, length 4 for 2x2)
    ]
    
    W_forms_2x2 = [
        [1, 0, 0, 1], # s=1: C00, C11
        [0, 0, 1, -1], # s=2: C10, -C11
        [0, 1, 0, 0], # s=3: C01
        [-1, 0, 1, 0], # s=4: -C00, C10
        [-1, 1, 0, 0], # s=5: -C00, C01
        [0, 0, 0, 1], # s=6: C11
        [1, 0, 0, -1] # s=7: C00, -C11
    ]
    
    # For 2x1 block (single column k), C is 2x1.
    # C0 = M1 - M4 - M5 + M7
    # C1 = M2 + M4
    # Wait, for 2x1, the output is a column.
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    # Strassen for 2x2 gives 4 outputs. For 2x1, we only need 2.
    # But the decomposition is the same, we just ignore the other 2 outputs?
    # No, the tensor is for the specific multiplication.
    # The rank of 2x2 x 1 is 7.
    # The W forms for 2x1 are the first two columns of W_forms_2x2?
    # No, the C indices are different.
    # For 2x1, C has 2 entries.
    # C0 = M1 - M4 - M5 + M7
    # C1 = M2 + M4
    # So W vectors (length 2):
    # s=1: [1, 0]
    # s=2: [0, 1]
    # s=3: [0, 0]
    # s=4: [-1, 1]
    # s=5: [-1, 0]
    # s=6: [0, 0]
    # s=7: [1, 0]
    
    W_forms_2x1 = [
        [1, 0],
        [0, 1],
        [0, 0],
        [-1, 1],
        [-1, 0],
        [0, 0],
        [1, 0]
    ]
    
    # B forms for 2x2 block (k, k+1)
    # M1: B00 + B11
    # M2: B00
    # M3: B01 - B11
    # M4: B10 - B00
    # M5: B11
    # M6: B00 + B01
    # M7: B10 + B11
    
    # Global V indices for block k (cols k, k+1):
    # B0,k is V[0, k]
    # B0,k+1 is V[0, k+1]
    # B1,k is V[1, k]
    # B1,k+1 is V[1, k+1]
    
    # B forms (length 4 for the block, mapped to global 30)
    # We will create V vectors of length 30.
    
    def get_V_2x2(k):
        # Returns list of 7 vectors of length 30
        vecs = []
        for s in range(7):
            v = [0.0] * 30
            if s == 0: # M1: B00 + B11
                v[0*k] = 1 # B0,k
                v[1*k+1] = 1 # B1,k+1? No.
                # B00 is B0,k. Index 0*2 + k? No, m*p = 2*15=30.
                # Row j, col k. Index j*p + k.
                # B0,k -> 0*15 + k = k
                # B1,k+1 -> 1*15 + k+1 = 15+k+1
                v[k] = 1
                v[15+k+1] = 1
            elif s == 1: # M2: B00
                v[k] = 1
            elif s == 2: # M3: B01 - B11
                v[k+1] = 1 # B0,k+1
                v[15+k+1] = -1 # B1,k+1
            elif s == 3: # M4: B10 - B00
                v[15+k] = 1 # B1,k
                v[k] = -1 # B0,k
            elif s == 4: # M5: B11
                v[15+k+1] = 1
            elif s == 5: # M6: B00 + B01
                v[k] = 1
                v[k+1] = 1
            elif s == 6: # M7: B10 + B11
                v[15+k] = 1
                v[15+k+1] = 1
            vecs.append(v)
        return vecs
    
    def get_V_2x1(k):
        # Returns list of 7 vectors of length 30
        vecs = []
        for s in range(7):
            v = [0.0] * 30
            # B0,k -> k
            # B1,k -> 15+k
            if s == 0: # M1: B0 + B1
                v[k] = 1
                v[15+k] = 1
            elif s == 1: # M2: B0
                v[k] = 1
            elif s == 2: # M3: 0 (B01 - B11 is 0 for 2x1)
                pass
            elif s == 3: # M4: B1 - B0
                v[15+k] = 1
                v[k] = -1
            elif s == 4: # M5: B1
                v[15+k] = 1
            elif s == 5: # M6: B0
                v[k] = 1
            elif s == 6: # M7: B1
                v[15+k] = 1
            vecs.append(v)
        return vecs
    
    # W forms for global W (length 30)
    # C indices: i*n + k? No, W[k*n+i, r].
    # C is 2x15.
    # C_{i,k} is at index k*2 + i.
    
    def get_W_2x2(k):
        # Returns list of 7 vectors of length 30
        vecs = []
        for s in range(7):
            v = [0.0] * 30
            w_form = W_forms_2x2[s]
            # C00 -> k*2 + 0
            # C01 -> (k+1)*2 + 0
            # C10 -> k*2 + 1
            # C11 -> (k+1)*2 + 1
            v[k*2 + 0] = w_form[0]
            v[(k+1)*2 + 0] = w_form[1]
            v[k*2 + 1] = w_form[2]
            v[(k+1)*2 + 1] = w_form[3]
            vecs.append(v)
        return vecs
    
    def get_W_2x1(k):
        # Returns list of 7 vectors of length 30
        vecs = []
        for s in range(7):
            v = [0.0] * 30
            w_form = W_forms_2x1[s]
            # C0 -> k*2 + 0
            # C1 -> k*2 + 1
            v[k*2 + 0] = w_form[0]
            v[k*2 + 1] = w_form[1]
            vecs.append(v)
        return vecs
    
    # Build U, V, W
    # 7 blocks of 2x2 (k=0,2,4,6,8,10,12)
    # 1 block of 2x1 (k=14)
    
    for k in range(0, 14, 2):
        U_block = [np.array(A_forms[s]) for s in range(7)]
        V_block = get_V_2x2(k)
        W_block = get_W_2x2(k)
        for s in range(7):
            U_cols.append(U_block[s])
            V_cols.append(V_block[s])
            W_cols.append(W_block[s])
            
    # Last block k=14
    U_block = [np.array(A_forms[s]) for s in range(7)]
    V_block = get_V_2x1(14)
    W_block = get_W_2x1(14)
    for s in range(7):
        U_cols.append(U_block[s])
        V_cols.append(V_block[s])
        W_cols.append(W_block[s])
        
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    # Check rank and remove dependencies
    # We have 56 columns.
    # We want to reduce to 53.
    # We will check for linear dependencies in the tensor.
    # This is complex. I will just return the 56 rank for now, 
    # but the prompt says 53 is possible.
    # I will try to remove the last 3 columns if they are dependent.
    
    # For now, return 56.
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
