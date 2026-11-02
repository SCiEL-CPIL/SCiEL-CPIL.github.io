import numpy as np

def construct():
    n, m, p = 4, 5, 6
    
    # We will build U, V, W by concatenating blocks.
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # Helper to add a block to the global matrices
    # We'll collect lists of (u_vec, v_vec, w_vec) for each rank-1 term
    # But for Strassen, we have structured blocks.
    
    # Let's define a function to get the decomposition for a sub-problem
    # defined by row/col ranges.
    
    # This approach is complex to manage indices.
    # Alternative: Construct the full tensor T and then find a low-rank approximation?
    # No, we need exact integer/half-integer.
    
    # Let's stick to the explicit construction.
    
    # We'll create empty lists for columns of U, V, W
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_idx, v_idx, w_idx, val=1):
        """Add a rank-1 term with 1 at specific indices."""
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        u_vec[u_idx] = val
        v_vec[v_idx] = val
        w_vec[w_idx] = val
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    def add_scaled_term(u_vec, v_vec, w_vec, scale=1):
        """Add a rank-1 term with given vectors scaled."""
        U_cols.append([x * scale for x in u_vec])
        V_cols.append([x * scale for x in v_vec])
        W_cols.append([x * scale for x in w_vec])

    # 1. A[0:4, 0] * B[0, 0:6]
    # A rows 0-3, col 0.
    # B row 0, cols 0-5.
    for i in range(4):
        for k in range(6):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            add_term(u_idx, v_idx, w_idx)
            
    # 2. A[0:4, 1:5] * B[1:5, 0:4]  (4x4 * 4x4)
    # This is a 4x4 multiplication.
    # We need a function to expand a 4x4 Strassen decomposition.
    
    def get_4x4_decomp(a_row_offset, b_col_offset, c_row_offset, c_col_offset):
        """
        Returns list of (u_vec, v_vec, w_vec) for 4x4x4 multiplication.
        a_row_offset: starting row in A
        b_col_offset: starting col in B (which is row in B)
        c_row_offset: starting row in C
        c_col_offset: starting col in C
        """
        terms = []
        # Strassen 4x4 is 2x2 blocks of 2x2.
        # A = [[A00, A01], [A10, A11]] each 2x2.
        # B = [[B00, B01], [B10, B11]] each 2x2.
        
        # We need the 2x2 Strassen decomposition for each of the 7 products.
        # Product 1: (A00+A11) * (B00+B11)
        # Product 2: (A10+A11) * B00
        # ...
        
        # This is getting very nested. 
        # Let's write a generic recursive Strassen.
        
        pass

    # Given the complexity, let's use a simpler but effective method:
    # Just use the 1+4 split with 4x4x4 and 4x4x2.
    # For 4x4x4, we can hardcode the 49 terms? No, too long.
    # We can use numpy to generate the tensor and then use a solver?
    # No, we need integer/half-integer.
    
    # Let's use the 2+3 split (rank 110) as it is easier to implement correctly?
    # 105 is better.
    
    # Let's implement the 4x4 Strassen explicitly.
    
    def strassen_2x2_decomp():
        """
        Returns U, V, W for 2x2x2 Strassen.
        U: 4x7, V: 4x7, W: 4x7
        Indices:
        A: 00, 01, 10, 11 -> 0, 1, 2, 3
        B: 00, 01, 10, 11 -> 0, 1, 2, 3
        C: 00, 01, 10, 11 -> 0, 1, 2, 3
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1 # Wait, W needs to map to C
        # C00 gets M1. C11 gets M1.
        # So W[0, 0] = 1, W[3, 0] = 1.
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1 # C10 += M2, C11 -= M2
        
        # M3 = A00(B10-B11)
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[2, 2] = 1 # C01 += M3, C10 += M3
        
        # M4 = A11(B01-B00)
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1 # C00 += M4, C10 += M4
        
        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1 # C00 -= M5, C01 += M5
        
        # M6 = (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1 # C11 += M6
        
        # M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1 # C00 += M7
        
        return U, V, W

    # Now we need to embed this 2x2 decomp into larger matrices.
    
    return {"U": [], "V": [], "W": []} # Placeholder
