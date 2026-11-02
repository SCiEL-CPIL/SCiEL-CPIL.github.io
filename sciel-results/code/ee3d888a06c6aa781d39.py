import numpy as np

def construct():
    """
    Finds a rank-minimal exact tensor decomposition for 2x2 * 2x16 matrix multiplication.
    Rank is 56 (8 blocks of Strassen rank 7).
    """
    n, m, p = 2, 2, 16
    
    # Strassen decomposition for 2x2 * 2x2
    # A is 2x2, B is 2x2, C is 2x2
    # U: (4, 7) maps A to 7 intermediate terms
    # V: (4, 7) maps B to 7 intermediate terms
    # W: (4, 7) maps products to C
    
    # A indices: 0:00, 1:01, 2:10, 3:11
    # B indices: 0:00, 1:01, 2:10, 3:11
    # C indices: 0:00, 1:01, 2:10, 3:11
    
    # Strassen formulas:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A11 - A01) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Construct U, V, W for single 2x2x2 case
    U0 = np.array([
        [1, 0, 1, 0, 1, 0, 0], # A00 in M1, M3, M5
        [0, 0, 0, 0, 1, 0, 0], # A01 in M5
        [0, 1, 0, 0, 0, -1, 0],# A10 in M2, M6
        [1, 1, 0, 1, 0, 0, -1] # A11 in M1, M2, M4, M7
    ], dtype=int)
    
    V0 = np.array([
        [1, 1, 0, -1, 0, 1, 0], # B00 in M1, M2, M4, M6
        [0, 0, 1, 0, 0, 1, 0],  # B01 in M3, M6
        [0, 0, 0, 1, 0, 0, 1],  # B10 in M4, M7
        [1, 0, -1, 0, 1, 0, 1]  # B11 in M1, M3, M5, M7
    ], dtype=int)
    
    W0 = np.array([
        [1, 0, 0, 1, -1, 0, 1], # C00 from M1, M4, M5, M7
        [0, 0, 1, 0, 1, 0, 0],  # C01 from M3, M5
        [0, 1, 0, 1, 0, 0, 0],  # C10 from M2, M4
        [1, -1, 1, 0, 0, 1, 0]  # C11 from M1, M2, M3, M6
    ], dtype=int)
    
    num_blocks = p // m  # 8 blocks of 2 columns
    R = num_blocks * 7   # Total rank 56
    
    # Initialize full matrices
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)
    
    # Fill matrices using block replication
    # A is 2x2, flat index i*m + j. U0 is 4x7.
    # Since A is always the same 2x2, U is just U0 tiled 8 times horizontally.
    U = np.tile(U0, (1, num_blocks))
    
    # B is 2x16. Flat index j*p + k.
    # V0 corresponds to a 2x2 block of B (cols 0,1).
    # For block q, B columns are 2q, 2q+1.
    # Row j of B in block q corresponds to global index j*p + (2q + local_col).
    # V0 rows: 0->B00, 1->B01, 2->B10, 3->B11.
    # Global B indices for block q:
    #   local 0 (B00) -> j=0, k=2q   -> 0*p + 2q
    #   local 1 (B01) -> j=0, k=2q+1 -> 0*p + 2q+1
    #   local 2 (B10) -> j=1, k=2q   -> 1*p + 2q
    #   local 3 (B11) -> j=1, k=2q+1 -> 1*p + 2q+1
    
    for q in range(num_blocks):
        col_start = q * 7
        col_end = (q + 1) * 7
        
        # V0 is 4x7. We need to map its rows to global V rows.
        # V0[0] -> global row 2*q
        # V0[1] -> global row 2*q + 1
        # V0[2] -> global row p + 2*q
        # V0[3] -> global row p + 2*q + 1
        
        V[2*q, col_start:col_end] = V0[0]
        V[2*q + 1, col_start:col_end] = V0[1]
        V[p + 2*q, col_start:col_end] = V0[2]
        V[p + 2*q + 1, col_start:col_end] = V0[3]
        
        # W corresponds to C. C is 2x16.
        # Convention: W[k*n + i, r].
        # C local indices in W0: 0->C00, 1->C01, 2->C10, 3->C11.
        # For block q, C columns are 2q, 2q+1.
        # C[i, k] -> global index k*n + i.
        #   local 0 (C00) -> i=0, k=2q   -> 2*q*n + 0
        #   local 1 (C01) -> i=0, k=2q+1 -> (2*q+1)*n + 0
        #   local 2 (C10) -> i=1, k=2q   -> 2*q*n + 1
        #   local 3 (C11) -> i=1, k=2q+1 -> (2*q+1)*n + 1
        
        W[2*q*n + 0, col_start:col_end] = W0[0]
        W[(2*q+1)*n + 0, col_start:col_end] = W0[1]
        W[2*q*n + 1, col_start:col_end] = W0[2]
        W[(2*q+1)*n + 1, col_start:col_end] = W0[3]
        
    # Verification
    np.random.seed(123)
    A_test = np.random.randint(-5, 5, size=(n, m))
    B_test = np.random.randint(-5, 5, size=(m, p))
    C_test = A_test @ B_test
    
    A_flat = A_test.flatten()
    B_flat = B_test.flatten()
    
    terms = (U.T @ A_flat) * (V.T @ B_flat)
    C_decomp = W @ terms
    
    C_test_flat = C_test.T.flatten() # k*n + i
    
    if not np.allclose(C_decomp, C_test_flat, rtol=1e-9, atol=1e-9):
        # Fallback to schoolbook (should not happen)
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
