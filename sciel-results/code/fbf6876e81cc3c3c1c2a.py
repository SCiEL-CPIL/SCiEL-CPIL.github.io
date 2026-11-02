import numpy as np
import time

def construct():
    n, m, p = 2, 2, 12
    start_time = time.time()
    deadline = 100  # seconds, safety margin before 120s limit

    # Rank 42 Strassen decomposition
    def get_strassen_42():
        rank = 42
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Strassen coefficients for 2x2x2
        # M1: (A00+A11)(B00+B11)
        # M2: (A10+A11)B00
        # M3: A00(B10-B11)
        # M4: A11(B01-B00)
        # M5: (A00+A01)B11
        # M6: (A10-A00)(B00+B10)
        # M7: (A01-A11)(B01+B11)
        
        # Indices: U[i*m+j], V[j*p+k], W[k*n+i]
        # For block b (0..5), k_offset = 2*b
        
        strassen_coeffs = [
            # U (i, j), V (j, k), W (k, i)
            # M1
            ([[1, 0], [0, 1]], [[1, 1], [0, 0]], [[1, 0], [1, 0]]), # A00+A11, B00+B11 -> C00, C10? No.
            # Let's use the standard Strassen mapping.
            # C00 = M1 - M2 + M3 + M4
            # C01 = M5 - M2 + M4 + M7
            # C10 = M1 - M3 - M4 + M6
            # C11 = M2 + M3 - M5 + M7
            
            # We need to construct U, V, W such that sum U V W = 1.
            # This is equivalent to the decomposition.
            
            # Standard Strassen:
            # M1 = (A00+A11)(B00+B11)
            # U: A00=1, A11=1. V: B00=1, B11=1.
            # M2 = (A10+A11)B00
            # U: A10=1, A11=1. V: B00=1.
            # M3 = A00(B10-B11)
            # U: A00=1. V: B10=1, B11=-1.
            # M4 = A11(B01-B00)
            # U: A11=1. V: B01=1, B00=-1.
            # M5 = (A00+A01)B11
            # U: A00=1, A01=1. V: B11=1.
            # M6 = (A10-A00)(B00+B10)
            # U: A10=1, A00=-1. V: B00=1, B10=1.
            # M7 = (A01-A11)(B01+B11)
            # U: A01=1, A11=-1. V: B01=1, B11=1.
            
            # Reconstruction:
            # C00 = M1 - M2 + M3 + M4
            # C01 = M5 - M2 + M4 + M7
            # C10 = M1 - M3 - M4 + M6
            # C11 = M2 + M3 - M5 + M7
            
            # This corresponds to W coefficients.
            # W is indexed by k, i.
            # For a block, k=0,1. i=0,1.
            # W[0, 0] (k=0, i=0) coeff for M1..M7: 1, -1, 1, 1, 0, 0, 0
            # W[1, 0] (k=1, i=0) coeff for M1..M7: 0, -1, 0, 1, 1, 0, 1
            # W[0, 1] (k=0, i=1) coeff for M1..M7: 1, 0, -1, -1, 0, 1, 0
            # W[1, 1] (k=1, i=1) coeff for M1..M7: 0, 1, 1, 0, -1, 0, 1
        ]
        
        # Define U, V, W for one Strassen block (rank 7)
        # U is 2x2 (i, j)
        # V is 2x2 (j, k)
        # W is 2x2 (k, i)
        
        U_block = [
            [1, 0, 1, 0, 1, -1, 0], # i=0: A00, A01
            [0, 1, 0, 1, 0, 1, -1]  # i=1: A10, A11
        ]
        # Wait, U indices are i, j.
        # M1: A00, A11. U[0,0]=1, U[1,1]=1.
        # M2: A10, A11. U[1,0]=1, U[1,1]=1.
        # M3: A00. U[0,0]=1.
        # M4: A11. U[1,1]=1.
        # M5: A00, A01. U[0,0]=1, U[0,1]=1.
        # M6: A10, A00. U[1,0]=1, U[0,0]=-1.
        # M7: A01, A11. U[0,1]=1, U[1,1]=-1.
        
        U_block = np.zeros((4, 7))
        # Flatten U: i*m+j
        # i=0, j=0 -> 0
        # i=0, j=1 -> 1
        # i=1, j=0 -> 2
        # i=1, j=1 -> 3
        
        U_block[0] = [1, 0, 1, 0, 1, -1, 0] # A00
        U_block[1] = [0, 1, 0, 0, 1, 0, 1]  # A01
        U_block[2] = [0, 1, 0, 0, 0, 1, 0]  # A10
        U_block[3] = [1, 0, 0, 1, 0, 0, -1] # A11
        
        V_block = np.zeros((4, 7))
        # Flatten V: j*p+k (p=2 for block)
        # j=0, k=0 -> 0
        # j=0, k=1 -> 1
        # j=1, k=0 -> 2
        # j=1, k=1 -> 3
        
        V_block[0] = [1, 0, 0, -1, 0, 1, 0] # B00
        V_block[1] = [0, 1, 0, 1, 0, 0, 1]  # B01
        V_block[2] = [1, 0, 1, 0, 0, 1, 0]  # B10
        V_block[3] = [1, 0, -1, 0, 1, 0, 1] # B11
        
        W_block = np.zeros((4, 7))
        # Flatten W: k*n+i (n=2)
        # k=0, i=0 -> 0
        # k=0, i=1 -> 1
        # k=1, i=0 -> 2
        # k=1, i=1 -> 3
        
        W_block[0] = [1, 0, 0, 0, 0, 0, 0] # C00 (k=0, i=0) -> M1-M2+M3+M4
        # Wait, W index is k, i.
        # C00 is i=0, k=0. Index 0*2+0=0.
        # C01 is i=0, k=1. Index 1*2+0=2.
        # C10 is i=1, k=0. Index 0*2+1=1.
        # C11 is i=1, k=1. Index 1*2+1=3.
        
        # Coeffs for M1..M7
        # C00: 1, -1, 1, 1, 0, 0, 0
        # C01: 0, -1, 0, 1, 1, 0, 1
        # C10: 1, 0, -1, -1, 0, 1, 0
        # C11: 0, 1, 1, 0, -1, 0, 1
        
        W_block[0] = [1, -1, 1, 1, 0, 0, 0]
        W_block[1] = [1, 0, -1, -1, 0, 1, 0]
        W_block[2] = [0, -1, 0, 1, 1, 0, 1]
        W_block[3] = [0, 1, 1, 0, -1, 0, 1]
        
        # Construct full 2x2x12
        full_rank = 42
        full_U = [[0] * full_rank for _ in range(4)]
        full_V = [[0] * full_rank for _ in range(24)]
        full_W = [[0] * full_rank for _ in range(24)]
        
        for b in range(6):
            offset = b * 7
            k_off = b * 2
            for s in range(7):
                r = offset + s
                # U
                for i in range(2):
                    for j in range(2):
                        full_U[i*2+j][r] = float(U_block[i*2+j, s])
                # V
                for j in range(2):
                    for k in range(2):
                        full_V[j*12 + (k_off + k)][r] = float(V_block[j*2+k, s])
                # W
                for k in range(2):
                    for i in range(2):
                        full_W[(k_off + k)*2 + i][r] = float(W_block[k*2+i, s])
                        
        return {"U": full_U, "V": full_V, "W": full_W}

    return get_strassen_42()
