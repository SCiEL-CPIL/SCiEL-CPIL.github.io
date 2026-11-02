import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (16, rank)
    # W: (p*n, rank) = (8, rank)
    
    # Strassen factors for 2x2x2
    # Local indices:
    # A: 0:(0,0) 1:(0,1) 2:(1,0) 3:(1,1)
    # B: 0:(0,0) 1:(0,1) 2:(1,0) 3:(1,1)
    # C: 0:(0,0) 1:(0,1) 2:(1,0) 3:(1,1)
    
    strassen_factors = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6
        ({2: 1, 0: -1}, {0: 1, 2: 1}, {3: 1}),
        # M7
        ({1: 1, 3: -1}, {1: 1, 3: 1}, {0: 1}),
    ]
    
    # Blocks:
    # 1. A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # 2. A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # 3. A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # 4. A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    
    # Global index mapping
    # A[i, j] -> i*4 + j
    # B[j, k] -> j*4 + k
    # C[k, i] -> k*2 + i  (Trace convention: W[k*n+i])
    
    def get_global_A(i, j):
        return i * 4 + j
        
    def get_global_B(j, k):
        return j * 4 + k
        
    def get_global_C(k, i):
        return k * 2 + i
        
    blocks = [
        # (A_row_offset, A_col_offset, B_row_offset, B_col_offset, C_row_offset, C_col_offset)
        # Note: C is 2x4. C_row is i (0..1), C_col is k (0..3).
        # In W, index is k*2 + i.
        # So C_row_offset applies to i, C_col_offset applies to k.
        
        # Block 1: A cols 0-1, B rows 0-1, B cols 0-1 -> C cols 0-1
        (0, 0, 0, 0, 0, 0),
        # Block 2: A cols 0-1, B rows 0-1, B cols 2-3 -> C cols 2-3
        (0, 0, 0, 2, 0, 2),
        # Block 3: A cols 2-3, B rows 2-3, B cols 0-1 -> C cols 0-1
        (0, 2, 2, 0, 0, 0),
        # Block 4: A cols 2-3, B rows 2-3, B cols 2-3 -> C cols 2-3
        (0, 2, 2, 2, 0, 2),
    ]
    
    rank = 28
    U = [[0] * rank for _ in range(8)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(8)]
    
    col = 0
    for a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off in blocks:
        for A_coeffs, B_coeffs, C_coeffs in strassen_factors:
            # Fill U
            for local_i, coeff in A_coeffs.items():
                # local_i is 0..3 for 2x2 A
                # local_i = 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
                li, lj = divmod(local_i, 2) # Wait, 0:(0,0) 1:(0,1) 2:(1,0) 3:(1,1)
                # divmod(0,2) -> 0,0. divmod(1,2) -> 0,1. divmod(2,2) -> 1,0. divmod(3,2) -> 1,1.
                # This matches row, col.
                gi = a_row_off + li
                gj = a_col_off + lj
                U[get_global_A(gi, gj)][col] += coeff
                
            # Fill V
            for local_j, coeff in B_coeffs.items():
                lj, lk = divmod(local_j, 2)
                gj = b_row_off + lj
                gk = b_col_off + lk
                V[get_global_B(gj, gk)][col] += coeff
                
            # Fill W
            for local_k, coeff in C_coeffs.items():
                # local_k is 0..3 for 2x2 C
                # local_k = 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
                # C is 2x4. local_k maps to (i, k) of the 2x2 block.
                # li, lk = divmod(local_k, 2)
                li, lk = divmod(local_k, 2)
                gi = c_row_off + li
                gk = c_col_off + lk
                W[get_global_C(gk, gi)][col] += coeff
                
            col += 1
            
    # Verify
    A = np.random.rand(2, 4)
    B = np.random.rand(4, 4)
    C_expected = A @ B
    
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # C[i, k] = sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*2+i, r] ?
    # No, tensor decomposition:
    # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
    # Here T is the multiplication tensor.
    # A[i, j] B[j, k] -> C[i, k]
    # U index: i*4 + j
    # V index: j*4 + k
    # W index: k*2 + i
    
    C_computed = np.zeros((2, 4))
    for i in range(2):
        for k in range(4):
            s = 0
            for j in range(4):
                for r in range(28):
                    s += U[i*4 + j][r] * V[j*4 + k][r] * W[k*2 + i][r]
            C_computed[i, k] = s
            
    if not np.allclose(C_computed, C_expected):
        # Fallback to schoolbook if Strassen fails
        rank = 32
        U = [[0] * rank for _ in range(8)]
        V = [[0] * rank for _ in range(16)]
        W = [[0] * rank for _ in range(8)]
        col = 0
        for i in range(2):
            for j in range(4):
                for k in range(4):
                    U[i*4 + j][col] = 1
                    V[j*4 + k][col] = 1
                    W[k*2 + i][col] = 1
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
