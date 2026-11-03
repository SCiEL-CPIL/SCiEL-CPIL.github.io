import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # U: n*m x R
    # V: m*p x R
    # W: p*n x R
    
    # We will build the decomposition term by term.
    # Each term is a rank-1 tensor (u, v, w).
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    def add_term(u, v, w):
        U_terms.append(u)
        V_terms.append(v)
        W_terms.append(w)
        
    # Indices:
    # i: 0..2 (3)
    # j: 0..3 (4)
    # k: 0..5 (6)
    
    # Block definitions
    # I1: i in [0, 1], I2: i in [2]
    # J1: j in [0, 1], J2: j in [2, 3]
    # K1: k in [0, 1], K2: k in [2, 3], K3: k in [4, 5]
    
    # Helper to create a zero vector of appropriate size
    def zero_u(): return [0.0] * (n * m)
    def zero_v(): return [0.0] * (m * p)
    def zero_w(): return [0.0] * (p * n)
    
    def set_u(u, i, j, val):
        u[i * m + j] = val
    def set_v(v, j, k, val):
        v[j * p + k] = val
    def set_w(w, k, i, val):
        w[k * n + i] = val
        
    # Strassen coefficients for 2x2x2
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map these to our indices.
    
    # Top Row (i in I1 = [0,1])
    # Output blocks K1, K2, K3
    
    # Block 1: C[0:2, 0:2] (i in 0,1; k in 0,1)
    # A[0:2, 0:2] (A11) * B[0:2, 0:2] (B11) + A[0:2, 2:4] (A12) * B[2:4, 0:2] (B21)
    
    # This is not a single 2x2x2 multiplication.
    # It is sum of two.
    # We can use Strassen for A11*B11 and Strassen for A12*B21.
    # Rank 7 + 7 = 14.
    
    # Let's do this for all 3 top blocks.
    # 3 blocks * 14 = 42.
    
    # Bottom Row (i in I2 = [2])
    # Output blocks K1, K2, K3
    # C[2, 0:2] = A[2, 0:2] (A21) * B[0:2, 0:2] (B11) + A[2, 2:4] (A22) * B[2:4, 0:2] (B21)
    # A21 is 1x2, B11 is 2x2.
    # Rank 4 for A21*B11.
    # Rank 4 for A22*B21.
    # Total 8 per block.
    # 3 blocks * 8 = 24.
    
    # Total rank = 42 + 24 = 66.
    
    # This is better than 68.
    
    # Let's implement this.
    
    # Strassen implementation for 2x2x2
    def add_strassen(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        # a_rows, a_cols: start indices for A block (2x2)
        # b_rows, b_cols: start indices for B block (2x2)
        # c_rows, c_cols: start indices for C block (2x2)
        
        # A11, A12, A21, A22
        # B11, B12, B21, B22
        
        # M1 = (A11 + A22)(B11 + B22)
        # ...
        
        pass # Placeholder

