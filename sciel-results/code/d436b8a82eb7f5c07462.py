import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    deadline = time.time() + 100  # Safety margin before 120s limit
    
    # We'll build the decomposition using block structure:
    # j split: {0,1} and {2}
    # k split: {0,1}, {2,3}, {4,5}, {6,7}, {8}
    # 
    # Blocks:
    # 1-4: (j in {0,1}, k in {0,1},{2,3},{4,5},{6,7}) -> 2x2x2 Strassen, rank 7 each
    # 5:   (j in {0,1}, k in {8}) -> 2x2x1 schoolbook, rank 4
    # 6:   (j in {2}, k in {0..8}) -> 2x1x9 schoolbook, rank 2
    # Total rank: 4*7 + 4 + 2 = 34
    
    U_list = []
    V_list = []
    W_list = []
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec):
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)
    
    # Strassen decomposition for 2x2x2
    # A is 2x2 (indices i,j with i in {0,1}, j in {0,1})
    # B is 2x2 (indices j,k with j in {0,1}, k in {0,1})
    # C is 2x2 (indices i,k with i in {0,1}, k in {0,1})
    #
    # The 7 Strassen products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B11-B01)
    # M5 = (A00+A01)(B11+B00)
    # M6 = (A10-A00)(B01+B11)
    # M7 = (A01-A11)(B00+B10)
    #
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M5
    # C11 = M1-M2+M3+M6
    
    def strassen_2x2x2(j_offset, k_offset):
        """Add 7 Strassen terms for the 2x2x2 block at j_offset, k_offset."""
        # j indices: j_offset, j_offset+1
        # k indices: k_offset, k_offset+1
        
        # M1 = (A00+A11)(B00+B11)
        u = [0]*6
        u[0*3 + j_offset] = 1      # A[0, j_offset]
        u[1*3 + j_offset+1] = 1    # A[1, j_offset+1]
        v = [0]*27
        v[j_offset*9 + k_offset] = 1      # B[j_offset, k_offset]
        v[(j_offset+1)*9 + k_offset+1] = 1  # B[j_offset+1, k_offset+1]
        w = [0]*18
        w[k_offset*2 + 0] = 1      # C[0, k_offset]
        w[(k_offset+1)*2 + 1] = 1  # C[1, k_offset+1]
        add_term(u, v, w)
        
        # M2 = (A10+A11)B00
        u = [0]*6
        u[1*3 + j_offset] = 1      # A[1, j_offset]
        u[1*3 + j_offset+1] = 1    # A[1, j_offset+1]
        v = [0]*27
        v[j_offset*9 + k_offset] = 1  # B[j_offset, k_offset]
        w = [0]*18
        w[k_offset*2 + 1] = 1      # C[1, k_offset]
        add_term(u, v, w)
        
        # M3 = A00(B10-B00)
        u = [0]*6
        u[0*3 + j_offset] = 1      # A[0, j_offset]
        v = [0]*27
        v[(j_offset+1)*9 + k_offset] = 1   # B[j_offset+1, k_offset]
        v[j_offset*9 + k_offset] = -1      # B[j_offset, k_offset]
        w = [0]*18
        w[k_offset*2 + 0] = 1      # C[0, k_offset]
        add_term(u, v, w)
        
        # M4 = A11(B11-B01)
        u = [0]*6
        u[1*3 + j_offset+1] = 1    # A[1, j_offset+1]
        v = [0]*27
        v[(j_offset+1)*9 + k_offset+1] = 1  # B[j_offset+1, k_offset+1]
        v[j_offset*9 + k_offset+1] = -1     # B[j_offset, k_offset+1]
        w = [0]*18
        w[(k_offset+1)*2 + 1] = 1  # C[1, k_offset+1]
        add_term(u, v, w)
        
        # M5 = (A00+A01)(B11+B00)
        u = [0]*6
        u[0*3 + j_offset] = 1      # A[0, j_offset]
        u[0*3 + j_offset+1] = 1    # A[0, j_offset+1]
        v = [0]*27
        v[(j_offset+1)*9 + k_offset+1] = 1  # B[j_offset+1, k_offset+1]
        v[j_offset*9 + k_offset] = 1        # B[j_offset, k_offset]
        w = [0]*18
        w[k_offset*2 + 0] = -1     # C[0, k_offset]
        w[k_offset*2 + 1] = 1      # C[1, k_offset]
        w[(k_offset+1)*2 + 0] = 1  # C[0, k_offset+1]
        add_term(u, v, w)
        
        # M6 = (A10-A00)(B01+B11)
        u = [0]*6
        u[1*3 + j_offset] = 1      # A[1, j_offset]
        u[0*3 + j_offset] = -1     # A[0, j_offset]
        v = [0]*27
        v[j_offset*9 + k_offset+1] = 1     # B[j_offset, k_offset+1]
        v[(j_offset+1)*9 + k_offset+1] = 1 # B[j_offset+1, k_offset+1]
        w = [0]*18
        w[(k_offset+1)*2 + 1] = 1  # C[1, k_offset+1]
        add_term(u, v, w)
        
        # M7 = (A01-A11)(B00+B10)
        u = [0]*6
        u[0*3 + j_offset+1] = 1    # A[0, j_offset+1]
        u[1*3 + j_offset+1] = -1   # A[1, j_offset+1]
        v = [0]*27
        v[j_offset*9 + k_offset] = 1        # B[j_offset, k_offset]
        v[(j_offset+1)*9 + k_offset] = 1   # B[j_offset+1, k_offset]
        w = [0]*18
        w[k_offset*2 + 0] = 1      # C[0, k_offset]
        add_term(u, v, w)
    
    # Add Strassen blocks for k pairs: (0,1), (2,3), (4,5), (6,7)
    for k_offset in [0, 2, 4, 6]:
        strassen_2x2x2(0, k_offset)
    
    # 2x2x1 block: j in {0,1}, k=8
    # Schoolbook: 4 terms, one for each (i,j)
    for i in range(2):
        for j in range(2):
            u = [0]*6
            u[i*3 + j] = 1
            v = [0]*27
            v[j*9 + 8] = 1
            w = [0]*18
            w[8*2 + i] = 1
            add_term(u, v, w)
    
    # 2x1x9 block: j=2, k in {0..8}
    # Schoolbook: 2 terms, one for each i
    for i in range(2):
        u = [0]*6
        u[i*3 + 2] = 1
        v = [0]*27
        for k in range(9):
            v[2*9 + k] = 1
        w = [0]*18
        for k in range(9):
            w[k*2 + i] = 1
        add_term(u, v, w)
    
    # Verify the decomposition
    R = len(U_list)
    U = np.array(U_list).T  # (6, R)
    V = np.array(V_list).T  # (27, R)
    W = np.array(W_list).T  # (18, R)
    
    # Check: for all i,j,k: sum_r U[i*3+j, r] * V[j*9+k, r] * W[k*2+i, r] == 1
    valid = True
    for i in range(2):
        for j in range(3):
            for k in range(9):
                total = 0.0
                for r in range(R):
                    total += U[i*3+j, r] * V[j*9+k, r] * W[k*2+i, r]
                if abs(total - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        # All entries are integers, which is fine
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook if verification fails
    print("Verification failed, falling back to schoolbook")
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
