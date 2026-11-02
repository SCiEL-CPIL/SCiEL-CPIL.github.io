"""
Matrix multiplication tensor decomposition for 2x3 by 3x7.
Uses a hybrid Strassen/schoolbook approach to achieve rank 39.
"""

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices
    of a rank-39 decomposition for multiplying a 2x3 matrix by a 3x7 matrix.
    
    The decomposition satisfies:
    C[i, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for all i, j, k.
    """
    n, m, p = 2, 3, 7
    
    # Initialize empty lists for factor matrices
    U_cols = []  # Each column is a vector of length n*m = 6
    V_cols = []  # Each column is a vector of length m*p = 21
    W_cols = []  # Each column is a vector of length p*n = 14
    
    def add_strassen_block(A_rows, B_cols, C_cols, offset):
        """
        Add a Strassen decomposition for a 2x2 * 2x2 block.
        A_rows: which rows of A (always [0,1] for 2x2)
        B_cols: which columns of B (2 consecutive cols)
        C_cols: which columns of C (2 consecutive cols)
        offset: column offset in B and C
        """
        # Strassen 7 multiplications for 2x2 * 2x2
        # A = [[a00, a01], [a10, a11]]
        # B = [[b00, b01], [b10, b11]]
        # 
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b10-b11)
        # M4 = a11(b01-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b10)
        # M7 = (a01-a11)(b11+b01)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # For each Strassen product, define U, V, W contributions
        # U: 2x2 matrix entries at positions (i*3+j) for i in {0,1}, j in B_cols
        # V: 2x2 matrix entries at positions (j*7+k) for j in {0,1} (local), k in B_cols
        # W: output entries at positions (k*2+i) for i in {0,1}, k in C_cols
        
        # Map local indices to global
        # A rows: i_global in {0,1}, j_local in {0,1} -> j_global = B_cols[0] + j_local
        # B cols: j_local in {0,1}, k_global in C_cols
        
        strassen_products = [
            # (U_coeffs, V_coeffs, W_coeffs, sign)
            # Each product contributes to multiple C entries with different signs
            
            # M1 = (a00+a11)(b00+b11)
            {
                'name': 'M1',
                'U': [(0, 1), (3, 1)],  # a00 at pos 0*3+0=0, a11 at pos 1*3+1=4... wait
                # Let me redo this with correct indexing
            }
        ]
        
        # Let me be more careful with indexing
        # A is 2x3, so A[i,j] is at index i*3+j in U
        # For a 2x2 submatrix using columns [col0, col1] of A:
        #   a00 -> A[0, col0] -> U index 0*3 + col0 = col0
        #   a01 -> A[0, col1] -> U index 0*3 + col1 = col1
        #   a10 -> A[1, col0] -> U index 1*3 + col0 = 3 + col0
        #   a11 -> A[1, col1] -> U index 1*3 + col1 = 3 + col1
        #
        # B is 3x7, so B[j,k] is at index j*7+k in V
        # For a 2x2 submatrix using rows [row0, row1] of B and cols [col0, col1]:
        #   b00 -> B[row0, col0] -> V index row0*7 + col0
        #   b01 -> B[row0, col1] -> V index row0*7 + col1
        #   b10 -> B[row1, col0] -> V index row1*7 + col0
        #   b11 -> B[row1, col1] -> V index row1*7 + col1
        #
        # C is 2x7, so C[i,k] is at index k*2+i in W (trace convention W[k*n+i])
        #   c00 -> C[0, col0] -> W index col0*2 + 0
        #   c01 -> C[0, col1] -> W index col1*2 + 0
        #   c10 -> C[1, col0] -> W index col0*2 + 1
        #   c11 -> C[1, col1] -> W index col1*2 + 1
        
        a00 = B_cols[0]  # A[0, B_cols[0]]
        a01 = B_cols[1]  # A[0, B_cols[1]]
        a10 = 3 + B_cols[0]  # A[1, B_cols[0]]
        a11 = 3 + B_cols[1]  # A[1, B_cols[1]]
        
        b00 = 0 * 7 + C_cols[0]  # B[0, C_cols[0]]
        b01 = 0 * 7 + C_cols[1]  # B[0, C_cols[1]]
        b10 = 1 * 7 + C_cols[0]  # B[1, C_cols[0]]
        b11 = 1 * 7 + C_cols[1]  # B[1, C_cols[1]]
        
        c00 = C_cols[0] * 2 + 0  # C[0, C_cols[0]]
        c01 = C_cols[1] * 2 + 0  # C[0, C_cols[1]]
        c10 = C_cols[0] * 2 + 1  # C[1, C_cols[0]]
        c11 = C_cols[1] * 2 + 1  # C[1, C_cols[1]]
        
        # Define the 7 Strassen products
        # Each product has: U entries (index, coeff), V entries (index, coeff), W entries (index, coeff)
        products = []
        
        # M1 = (a00 + a11) * (b00 + b11)
        products.append({
            'U': [(a00, 1), (a11, 1)],
            'V': [(b00, 1), (b11, 1)],
            'W_contrib': [
                (c00, 1),   # C00 += M1
                (c11, 1),   # C11 += M1
            ]
        })
        
        # M2 = (a10 + a11) * b00
        products.append({
            'U': [(a10, 1), (a11, 1)],
            'V': [(b00, 1)],
            'W_contrib': [
                (c10, 1),   # C10 += M2
                (c11, -1),  # C11 -= M2
            ]
        })
        
        # M3 = a00 * (b10 - b11)
        products.append({
            'U': [(a00, 1)],
            'V': [(b10, 1), (b11, -1)],
            'W_contrib': [
                (c01, 1),   # C01 += M3
                (c11, 1),   # C11 += M3
            ]
        })
        
        # M4 = a11 * (b01 - b00)
        products.append({
            'U': [(a11, 1)],
            'V': [(b01, 1), (b00, -1)],
            'W_contrib': [
                (c00, 1),   # C00 += M4
                (c10, 1),   # C10 += M4
            ]
        })
        
        # M5 = (a00 + a01) * b11
        products.append({
            'U': [(a00, 1), (a01, 1)],
            'V': [(b11, 1)],
            'W_contrib': [
                (c00, -1),  # C00 -= M5
                (c01, 1),   # C01 += M5
            ]
        })
        
        # M6 = (a10 - a00) * (b00 + b10)
        products.append({
            'U': [(a10, 1), (a00, -1)],
            'V': [(b00, 1), (b10, 1)],
            'W_contrib': [
                (c11, 1),   # C11 += M6
            ]
        })
        
        # M7 = (a01 - a11) * (b11 + b01)
        products.append({
            'U': [(a01, 1), (a11, -1)],
            'V': [(b11, 1), (b01, 1)],
            'W_contrib': [
                (c00, 1),   # C00 += M7
            ]
        })
        
        # Now, for each Strassen product, we need to create a column in U, V, W
        # But the W contributions go to different C entries with different signs
        # We need to split each product into separate columns if it contributes to multiple C entries
        # Actually, no - each Strassen product is one multiplication, so it should be one column
        # But the same product contributes to multiple C entries. This means we need to have
        # the W vector have non-zero entries at multiple positions.
        
        # Let me reconsider. In a tensor decomposition, each rank-1 term is U_r * V_r * W_r
        # where U_r is a vector, V_r is a vector, W_r is a vector.
        # The tensor T[i,j,k] = sum_r U_r[i] * V_r[j] * W_r[k]
        #
        # For Strassen, each of the 7 products is a rank-1 term in the 2x2*2x2 sub-tensor.
        # But each product contributes to multiple output entries. This is fine - the W vector
        # just has non-zero entries at those positions.
        
        for prod in products:
            # Create U vector
            U_vec = [0] * (n * m)
            for idx, coeff in prod['U']:
                U_vec[idx] += coeff
            
            # Create V vector
            V_vec = [0] * (m * p)
            for idx, coeff in prod['V']:
                V_vec[idx] += coeff
            
            # Create W vector
            W_vec = [0] * (p * n)
            for idx, coeff in prod['W_contrib']:
                W_vec[idx] += coeff
            
            U_cols.append(U_vec)
            V_cols.append(V_vec)
            W_cols.append(W_vec)
    
    def add_schoolbook_block(A_cols, B_rows, B_cols_range):
        """
        Add a schoolbook decomposition for a block.
        A_cols: columns of A used
        B_rows: rows of B used  
        B_cols_range: range of columns of B used
        """
        for i in range(n):
            for j in A_cols:
                for k in B_cols_range:
                    if j not in B_rows:
                        continue
                    U_vec = [0] * (n * m)
                    U_vec[i * m + j] = 1
                    
                    V_vec = [0] * (m * p)
                    V_vec[j * p + k] = 1
                    
                    W_vec = [0] * (p * n)
                    W_vec[k * n + i] = 1
                    
                    U_cols.append(U_vec)
                    V_cols.append(V_vec)
                    W_cols.append(W_vec)
    
    # Block structure:
    # A is 2x3, B is 3x7, C is 2x7
    #
    # We split B's 7 columns into: [0,1], [2,3], [4,5], [6]
    # We split A's 3 columns into: [0,1], [2]
    #
    # Blocks:
    # 1. A[:,0:2] * B[0:2, 0:2] -> C[:, 0:2]  (Strassen, 2x2 * 2x2)
    # 2. A[:,0:2] * B[0:2, 2:4] -> C[:, 2:4]  (Strassen, 2x2 * 2x2)
    # 3. A[:,0:2] * B[0:2, 4:6] -> C[:, 4:6]  (Strassen, 2x2 * 2x2)
    # 4. A[:,0:2] * B[0:2, 6:7] -> C[:, 6:7]  (2x2 * 2x1, schoolbook)
    # 5. A[:,2:3] * B[2:3, 0:7] -> C[:, 0:7]  (2x1 * 1x7, schoolbook)
    
    # Block 1: Strassen for A[:,0:2] * B[0:2, 0:2] -> C[:, 0:2]
    add_strassen_block(A_rows=[0,1], B_cols=[0,1], C_cols=[0,1], offset=0)
    
    # Block 2: Strassen for A[:,0:2] * B[0:2, 2:4] -> C[:, 2:4]
    add_strassen_block(A_rows=[0,1], B_cols=[0,1], C_cols=[2,3], offset=2)
    
    # Block 3: Strassen for A[:,0:2] * B[0:2, 4:6] -> C[:, 4:6]
    add_strassen_block(A_rows=[0,1], B_cols=[0,1], C_cols=[4,5], offset=4)
    
    # Block 4: Schoolbook for A[:,0:2] * B[0:2, 6:7] -> C[:, 6:7]
    # 2x2 * 2x1 = 2*2*1 = 4 terms
    add_schoolbook_block(A_cols=[0,1], B_rows=[0,1], B_cols_range=[6])
    
    # Block 5: Schoolbook for A[:,2:3] * B[2:3, 0:7] -> C[:, 0:7]
    # 2x1 * 1x7 = 2*1*7 = 14 terms
    add_schoolbook_block(A_cols=[2], B_rows=[2], B_cols_range=range(7))
    
    # Convert lists of columns to matrix format (rows x cols)
    rank = len(U_cols)
    U = [[U_cols[c][r] for c in range(rank)] for r in range(n * m)]
    V = [[V_cols[c][r] for c in range(rank)] for r in range(m * p)]
    W = [[W_cols[c][r] for c in range(rank)] for r in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
