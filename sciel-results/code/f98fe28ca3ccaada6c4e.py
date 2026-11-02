import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    rank = 50
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_222(i0, j0, k0):
        nonlocal col
        if col >= rank:
            return False
        
        # A indices: (i0,j0), (i0,j0+1), (i0+1,j0), (i0+1,j0+1)
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        
        # B indices: (j0,k0), (j0,k0+1), (j0+1,k0), (j0+1,k0+1)
        e = j0 * p + k0
        f = j0 * p + (k0 + 1)
        g = (j0 + 1) * p + k0
        h = (j0 + 1) * p + (k0 + 1)
        
        # C indices: (k0,i0), (k0,i0+1), (k0+1,i0), (k0+1,i0+1)
        c00 = k0 * n + i0
        c01 = k0 * n + (i0 + 1)
        c10 = (k0 + 1) * n + i0
        c11 = (k0 + 1) * n + (i0 + 1)
        
        # Strassen 2x2x2: 7 multiplications
        # M1 = (A11+A22)(B11+B22)
        U[a][col] += 1
        U[d][col] += 1
        V[e][col] += 1
        V[h][col] += 1
        W[c00][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M2 = (A21+A22)B11
        U[c][col] += 1
        U[d][col] += 1
        V[e][col] += 1
        W[c10][col] += 1
        W[c11][col] -= 1
        col += 1
        
        # M3 = A11(B21-B11)
        U[a][col] += 1
        V[f][col] += 1
        V[h][col] -= 1
        W[c01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M4 = A22(B12-B21)
        U[d][col] += 1
        V[g][col] += 1
        V[e][col] -= 1
        W[c00][col] += 1
        W[c10][col] += 1
        col += 1
        
        # M5 = (A11+A12)B22
        U[a][col] += 1
        U[b][col] += 1
        V[h][col] += 1
        W[c00][col] -= 1
        W[c01][col] += 1
        col += 1
        
        # M6 = (A21-A11)(B11+B12)
        U[c][col] += 1
        U[a][col] -= 1
        V[e][col] += 1
        V[f][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M7 = (A12-A21)(B21+B22)
        U[b][col] += 1
        U[d][col] -= 1
        V[g][col] += 1
        V[h][col] += 1
        W[c00][col] += 1
        col += 1
        
        return True

    def add_schoolbook_221(i0, j0, k0):
        nonlocal col
        if col >= rank:
            return False
        
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        
        e = j0 * p + k0
        g = (j0 + 1) * p + k0
        
        c00 = k0 * n + i0
        c10 = k0 * n + (i0 + 1)
        
        U[a][col] += 1
        V[e][col] += 1
        W[c00][col] += 1
        col += 1
        
        U[b][col] += 1
        V[g][col] += 1
        W[c00][col] += 1
        col += 1
        
        U[c][col] += 1
        V[e][col] += 1
        W[c10][col] += 1
        col += 1
        
        U[d][col] += 1
        V[g][col] += 1
        W[c10][col] += 1
        col += 1
        
        return True

    # Layout: split inner dim (4) into j=0,1 and j=2,3
    # Split output (7) into k=0,1,2,3,4,5,6
    # For each (i-block, j-block):
    #   j-block 0 (j=0,1): k=0,1 -> Strassen; k=2,3 -> Strassen; k=4,5 -> Strassen; k=6 -> schoolbook 2x2x1
    #   j-block 1 (j=2,3): k=0,1 -> Strassen; k=2,3 -> Strassen; k=4,5 -> Strassen; k=6 -> schoolbook 2x2x1
    
    # That's 6 Strassen + 2 schoolbook = 42 + 8 = 50
    
    if not add_strassen_222(0, 0, 0):
        pass
    if not add_strassen_222(0, 0, 2):
        pass
    if not add_strassen_222(0, 0, 4):
        pass
    if not add_schoolbook_221(0, 0, 6):
        pass
    
    if not add_strassen_222(0, 2, 0):
        pass
    if not add_strassen_222(0, 2, 2):
        pass
    if not add_strassen_222(0, 2, 4):
        pass
    if not add_schoolbook_221(0, 2, 6):
        pass

    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    max_error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                recon = np.dot(U_arr[a_idx, :], V_arr[b_idx, :] * W_arr[c_idx, :])
                err = abs(recon - 1.0)
                if err > max_error:
                    max_error = err
    
    # Check zero entries: for each (a_idx, b_idx), the product should only contribute to the correct c_idx
    for i in range(n):
        for j in range(m):
            a_idx = i * m + j
            for k in range(p):
                b_idx = j * p + k
                c_idx = k * n + i
                # For other c_idx', should be 0
                for k2 in range(p):
                    for i2 in range(n):
                        c_idx2 = k2 * n + i2
                        if c_idx2 == c_idx:
                            continue
                        recon2 = np.dot(U_arr[a_idx, :], V_arr[b_idx, :] * W_arr[c_idx2, :])
                        err2 = abs(recon2)
                        if err2 > max_error:
                            max_error = err2
    
    if max_error > 1e-9:
        # Fallback to empty
        return {"U": [[0]*rank for _ in range(n*m)], 
                "V": [[0]*rank for _ in range(m*p)], 
                "W": [[0]*rank for _ in range(p*n)]}
    
    return {"U": U, "V": V, "W": W}
