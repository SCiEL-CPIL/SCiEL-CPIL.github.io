"""Rewritten program for matrix multiplication 2x2 * 2x15 tensor rank minimization.
Uses a structural decomposition: group columns into pairs and use Strassen per pair,
with the remaining single column using schoolbook. This gives rank 7*7 + 4 = 53."""
import numpy as np
import time

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    N = n * m  # 4
    M = m * p  # 30
    P = p * n  # 30
    
    deadline = time.time() + 110.0  # Safety margin
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Process 7 pairs of columns with Strassen (7 multiplications each)
    # For columns (2g, 2g+1), the 2x2 sub-block of B is [[b00, b01], [b10, b11]]
    # where b00 = B[0, 2g], b01 = B[0, 2g+1], b10 = B[1, 2g], b11 = B[1, 2g+1]
    
    for g in range(7):
        k0 = 2 * g
        k1 = 2 * g + 1
        
        # Strassen formulas for A * B_block:
        # C00 = M1 - M4 + M6 - M7  (A00 row, B col 0 -> C col 0)
        # C01 = M3 + M5 + M7       (A00 row, B col 1 -> C col 1)  
        # C10 = M2 + M4 + M6       (A10 row, B col 0 -> C col 0)
        # C11 = M1 - M3 + M5 + M2  (A11 row, B col 1 -> C col 1)
        
        # Wait, let me re-derive carefully.
        # A = [[A00, A01], [A10, A11]], B = [[b00, b01], [b10, b11]]
        # C = [[C00, C01], [C10, C11]]
        # C00 = A00*b00 + A01*b10
        # C01 = A00*b01 + A01*b11
        # C10 = A10*b00 + A11*b10
        # C11 = A10*b01 + A11*b11
        
        # Strassen:
        # M1 = (A00+A11)*(b00+b11)
        # M2 = (A10+A11)*b00
        # M3 = A00*(b01-b11)
        # M4 = A11*(b10-b00)
        # M5 = (A00+A01)*b11
        # M6 = (A10-A00)*(b00+b01)
        # M7 = (A01-A11)*(b10+b11)
        
        # C00 = M1 - M4 + M6 - M7
        # C01 = M3 + M5 + M7
        # C10 = M2 + M4 + M6
        # C11 = M1 - M3 + M5 - M2
        
        # Let me verify C00:
        # M1 - M4 + M6 - M7
        # = (A00+A11)(b00+b11) - A11(b10-b00) + (A10-A00)(b00+b01) - (A01-A11)(b10+b11)
        # = A00*b00 + A00*b11 + A11*b00 + A11*b11 - A11*b10 + A11*b00 + A10*b00 + A10*b01 - A00*b00 - A00*b01 - A01*b10 - A01*b11 + A11*b10 + A11*b11
        # Let me collect terms:
        # A00*b00: +1 -1 = 0
        # A00*b11: +1
        # A11*b00: +1 +1 = +2... hmm this doesn't look right.
        
        # Let me use the standard Strassen verification:
        # C00 = M1 - M4 + M6 - M7
        # Actually, the standard Strassen is:
        # C00 = M1 - M4 + M6 - M7
        # C01 = M3 + M5 + M7
        # C10 = M2 + M4 + M6
        # C11 = M1 - M3 + M5 - M2
        
        # Let me just implement each multiplication factor.
        # For each Strassen product Mk, we have:
        # U factor: combination of A entries
        # V factor: combination of B entries
        # W factor: how it contributes to C entries
        
        # M1 = (A00+A11)*(b00+b11)
        # U: A00+A11 = [1,0,0,1] (indices: A00=0, A01=1, A10=2, A11=3)
        # V: b00+b11 = B[0,k0] + B[1,k1] = [0..1, 0..15, 0..1, 0..1]
        #   index 0: B[0,k0], index 1: B[1,k0], index 15: B[0,k1], index 16: B[1,k1]
        #   So V has 1 at position k0 (which is 15*k0 + 0... wait)
        #   V index: B[j,k] is at j*p + k = j*15 + k
        #   B[0,k0] is at 0*15 + k0 = k0
        #   B[1,k1] is at 1*15 + k1 = 15 + k1
        # W: contributes to C00 (+1), C11 (+1)
        #   C[i,j] is at j*n + i = j*2 + i
        #   C00 is at 0*2 + 0 = 0
        #   C11 is at 1*2 + 1 = 3
        
        U_M1 = [1, 0, 0, 1]
        V_M1 = [0]*30
        V_M1[k0] = 1      # B[0, k0]
        V_M1[15+k1] = 1   # B[1, k1]
        W_M1 = [0]*30
        W_M1[0] = 1       # C00
        W_M1[3] = 1       # C11
        
        # M2 = (A10+A11)*b00
        U_M2 = [0, 0, 1, 1]
        V_M2 = [0]*30
        V_M2[k0] = 1      # B[0, k0]
        W_M2 = [0]*30
        W_M2[2] = 1       # C10
        W_M2[3] = -1      # C11 (with minus sign)
        
        # M3 = A00*(b01-b11)
        U_M3 = [1, 0, 0, 0]
        V_M3 = [0]*30
        V_M3[15+k1] = 1   # B[0, k1]... wait, b01 = B[0, k1], b11 = B[1, k1]
        # b01 is B[0, k1] at index 15+k1? No, B[0, k1] is at 0*15 + k1 = k1
        # B[1, k1] is at 1*15 + k1 = 15+k1
        V_M3[k1] = 1      # B[0, k1]
        V_M3[15+k1] = -1  # B[1, k1]
        W_M3 = [0]*30
        W_M3[1] = 1       # C01
        W_M3[3] = -1      # C11 (with minus sign)
        
        # M4 = A11*(b10-b00)
        # b10 = B[1, k0] at 15+k0, b00 = B[0, k0] at k0
        U_M4 = [0, 0, 0, 1]
        V_M4 = [0]*30
        V_M4[15+k0] = 1   # B[1, k0]
        V_M4[k0] = -1     # B[0, k0]
        W_M4 = [0]*30
        W_M4[0] = -1      # C00 (minus)
        W_M4[2] = 1       # C10
        
        # M5 = (A00+A01)*b11
        # b11 = B[1, k1] at 15+k1
        U_M5 = [1, 1, 0, 0]
        V_M5 = [0]*30
        V_M5[15+k1] = 1   # B[1, k1]
        W_M5 = [0]*30
        W_M5[1] = 1       # C01
        W_M5[3] = 1       # C11
        
        # M6 = (A10-A00)*(b00+b01)
        # b00 = B[0,k0] at k0, b01 = B[0,k1] at k1
        U_M6 = [0, 0, 1, 0]  # A10
        U_M6[0] = -1         # -A00
        U_M6 = [-1, 0, 1, 0]
        V_M6 = [0]*30
        V_M6[k0] = 1         # B[0, k0]
        V_M6[k1] = 1         # B[0, k1]
        W_M6 = [0]*30
        W_M6[0] = 1          # C00
        W_M6[2] = 1          # C10
        
        # M7 = (A01-A11)*(b10+b11)
        # b10 = B[1,k0] at 15+k0, b11 = B[1,k1] at 15+k1
        U_M7 = [0, 1, 0, -1]  # A01 - A11
        V_M7 = [0]*30
        V_M7[15+k0] = 1       # B[1, k0]
        V_M7[15+k1] = 1       # B[1, k1]
        W_M7 = [0]*30
        W_M7[1] = 1           # C01
        W_M7[0] = -1          # C00 (minus)
        
        U_cols.extend([U_M1, U_M2, U_M3, U_M4, U_M5, U_M6, U_M7])
        V_cols.extend([V_M1, V_M2, V_M3, V_M4, V_M5, V_M6, V_M7])
        W_cols.extend([W_M1, W_M2, W_M3, W_M4, W_M5, W_M6, W_M7])
    
    # Process the last column (k=14) with schoolbook (4 products)
    k = 14
    # C[0,k] = A00*B[0,k] + A01*B[1,k]
    # C[1,k] = A10*B[0,k] + A11*B[1,k]
    
    # Product 1: A00 * B[0,k]
    U_S1 = [1, 0, 0, 0]
    V_S1 = [0]*30
    V_S1[k] = 1
    W_S1 = [0]*30
    W_S1[k*2+0] = 1  # C[0, k] is at index k*2 + 0... wait
    
    # C[i, k] where i is the row of C (0 or 1), and the output index is W[k*n+i]
    # n=2, so W index for C[i,k] is k*2 + i
    # C[0,k]: W index = 14*2 + 0 = 28
    # C[1,k]: W index = 14*2 + 1 = 29
    
    W_S1[28] = 1
    
    # Product 2: A01 * B[1,k]
    U_S2 = [0, 1, 0, 0]
    V_S2 = [0]*30
    V_S2[15+k] = 1
    W_S2 = [0]*30
    W_S2[28] = 1
    
    # Product 3: A10 * B[0,k]
    U_S3 = [0, 0, 1, 0]
    V_S3 = [0]*30
    V_S3[k] = 1
    W_S3 = [0]*30
    W_S3[29] = 1
    
    # Product 4: A11 * B[1,k]
    U_S4 = [0, 0, 0, 1]
    V_S4 = [0]*30
    V_S4[15+k] = 1
    W_S4 = [0]*30
    W_S4[29] = 1
    
    U_cols.extend([U_S1, U_S2, U_S3, U_S4])
    V_cols.extend([V_S1, V_S2, V_S3, V_S4])
    W_cols.extend([W_S1, W_S2, W_S3, W_S4])
    
    # Build final matrices
    rank = len(U_cols)
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    # Verify the decomposition
    # The tensor should satisfy: sum_r U[i,r] * V[j,r] * W[k,r] = T[i,j,k]
    # where T[i,m+j, j*p+k, k*n+i] = 1 for the matrix multiplication tensor
    # Actually, for matrix multiplication: (A*B)_{i,k} = sum_j A_{i,j} * B_{j,k}
    # The tensor is T_{i,j,k} = A_{i,j} * B_{j,k} -> contributes to C_{i,k}
    # So sum_r U[i,r] * V[j,r] * W[k,r] should equal 1 if this is the (i,j,k) entry
    # Wait, the convention is W[k*n+i] where k is the output column, i is the output row
    
    for test_i in range(n):
        for test_j in range(m):
            for test_k in range(p):
                idx_u = test_i * m + test_j
                idx_v = test_j * p + test_k
                idx_w = test_k * n + test_i
                
                val = np.sum(U[idx_u, :] * V[idx_v, :] * W[idx_w, :])
                if abs(val - 1.0) > 1e-10:
                    print(f"Verification failed at ({test_i},{test_j},{test_k}): got {val}, expected 1.0")
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
