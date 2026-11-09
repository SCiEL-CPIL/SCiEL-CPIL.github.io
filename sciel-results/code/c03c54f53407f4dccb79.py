import numpy as np
import time

def construct():
    n, m, p = 5, 5, 5
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s limit
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_vals, v_vals, w_vals):
        U = np.zeros((n*m, 1), dtype=float)
        V = np.zeros((m*p, 1), dtype=float)
        W = np.zeros((p*n, 1), dtype=float)
        for (i, j), val in u_vals.items():
            U[i*m+j, 0] = val
        for (j, k), val in v_vals.items():
            V[j*p+k, 0] = val
        for (k, i), val in w_vals.items():
            W[k*n+i, 0] = val
        U_list.append(U.flatten())
        V_list.append(V.flatten())
        W_list.append(W.flatten())
    
    def strassen_2x2(i0, i1, j0, j1, k0, k1):
        """Strassen for 2x2*2x2: rows i0,i1; A cols j0,j1; B cols k0,k1"""
        add_term({(i0,j0):1, (i1,j1):1}, {(j0,k0):1, (j1,k1):1}, {(k0,i0):1, (k1,i1):1})
        add_term({(i1,j0):1, (i1,j1):1}, {(j0,k0):1}, {(k0,i1):1, (k1,i1):-1})
        add_term({(i0,j0):1}, {(j0,k1):1, (j1,k1):-1}, {(k1,i0):1, (k1,i1):1})
        add_term({(i1,j1):1}, {(j1,k0):1, (j0,k0):-1}, {(k0,i0):1, (k0,i1):1})
        add_term({(i0,j0):1, (i0,j1):1}, {(j1,k1):1}, {(k0,i0):-1, (k1,i0):1})
        add_term({(i1,j0):1, (i0,j0):-1}, {(j0,k0):1, (j0,k1):1}, {(k1,i1):1})
        add_term({(i0,j1):1, (i1,j1):-1}, {(j1,k0):1, (j1,k1):1}, {(k0,i0):1})
    
    def strassen_3x3(i0, j0, k0):
        """Strassen for 3x3*3x3: rows i0..i0+2; A cols j0..j0+2; B cols k0..k0+2"""
        # Split 3x3 into 2+1 blocks
        # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
        # B = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x1, B21 is 1x2, B22 is 1x1
        
        i1, i2 = i0+1, i0+2
        j1, j2 = j0+1, j0+2
        k1, k2 = k0+1, k0+2
        
        # M1 = (A11+A22)(B11+B22) -> 2x2*2x2 (Strassen)
        # A11+A22: (i0,j0)=1,(i0,j1)=1,(i1,j0)=1,(i1,j1)=1,(i2,j2)=1
        # B11+B22: (j0,k0)=1,(j0,k1)=1,(j1,k0)=1,(j1,k1)=1,(j2,k2)=1
        # This is not a clean 2x2*2x2, so we need to handle it differently.
        
        # Actually, for 3x3, the standard approach is to use the 2+1 split with 7 multiplications
        # but the intermediate products are rectangular. Let me use a different decomposition.
        
        # Alternative: Use the known rank-14 decomposition for 3x3? No, let's just do schoolbook
        # for the 3x3 parts since Strassen for 3x3 is complex with rectangular blocks.
        
        # Schoolbook for 3x3*3x3 = 27 terms
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    add_term({(i0+i, j0+j): 1}, {(j0+j, k0+k): 1}, {(k0+k, i0+i): 1})
    
    def schoolbook_mul(i0, i1, j0, j1, k0, k1):
        """Schoolbook multiplication for (i1-i0)x(j1-j0) * (j1-j0)x(k1-k0)"""
        for i in range(i0, i1):
            for j in range(j0, j1):
                for k in range(k0, k1):
                    add_term({(i, j): 1}, {(j, k): 1}, {(k, i): 1})
    
    # 2+3 split:
    # A = [[A22, A23], [A32, A33]]
    #   A22: rows 0-1, cols 0-1 (2x2)
    #   A23: rows 0-1, cols 2-4 (2x3)
    #   A32: rows 2-4, cols 0-1 (3x2)
    #   A33: rows 2-4, cols 2-4 (3x3)
    # B = [[B22, B23], [B32, B33]]
    #   B22: rows 0-1, cols 0-1 (2x2)
    #   B23: rows 0-1, cols 2-4 (2x3)
    #   B32: rows 2-4, cols 0-1 (3x2)
    #   B33: rows 2-4, cols 2-4 (3x3)
    
    # C22 = A22*B22 + A23*B32
    # A22*B22: 2x2*2x2 -> Strassen (7 terms)
    strassen_2x2(0, 1, 0, 1, 0, 1)
    # A23*B32: 2x3*3x2 -> schoolbook (12 terms)
    schoolbook_mul(0, 2, 2, 5, 0, 2)
    
    # C23 = A22*B23 + A23*B33
    # A22*B23: 2x2*2x3 -> schoolbook (12 terms)
    schoolbook_mul(0, 2, 0, 2, 2, 5)
    # A23*B33: 2x3*3x3 -> schoolbook (18 terms)
    schoolbook_mul(0, 2, 2, 5, 2, 5)
    
    # C32 = A32*B22 + A33*B32
    # A32*B22: 3x2*2x2 -> schoolbook (12 terms)
    schoolbook_mul(2, 5, 0, 2, 0, 2)
    # A33*B32: 3x3*3x2 -> schoolbook (18 terms)
    schoolbook_mul(2, 5, 2, 5, 0, 2)
    
    # C33 = A32*B23 + A33*B33
    # A32*B23: 3x2*2x3 -> schoolbook (12 terms)
    schoolbook_mul(2, 5, 0, 2, 2, 5)
    # A33*B33: 3x3*3x3 -> schoolbook (27 terms)
    schoolbook_mul(2, 5, 2, 5, 2, 5)
    
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
