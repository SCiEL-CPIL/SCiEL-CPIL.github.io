import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 2, 4, 5
    
    # Build the target tensor for verification
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # Strassen rank-7 decomposition for 2x2x2
    # Standard Strassen:
    # M1 = A11*B11
    # M2 = (A21+A22)*B11
    # M3 = A11*(B21+B22)
    # M4 = A22*B22
    # M5 = (A11+A22)*(B11+B22)
    # M6 = (A12-A22)*(B21+B22)
    # M7 = (A21-A11)*(B11-B21)
    # C11 = M1+M4-M5+M7, C12 = M3+M6, C21 = M2+M5, C22 = M4-M3+M6+M7
    
    # U[r][li*2+lj], V[r][lj*2+lk], W[r][lk*2+li]
    strassen_U = [
        [1,0, 0,0],  # M1: A11
        [0,0, 1,1],  # M2: A21+A22
        [1,0, 0,0],  # M3: A11
        [0,0, 0,1],  # M4: A22
        [1,0, 1,1],  # M5: A11+A22
        [0,1, 0,-1], # M6: A12-A22
        [-1,0, 1,0], # M7: A21-A11
    ]
    strassen_V = [
        [1,0, 0,0],  # M1: B11
        [1,0, 0,0],  # M2: B11
        [0,1, 1,1],  # M3: B21+B22
        [0,0, 0,1],  # M4: B22
        [1,0, 0,1],  # M5: B11+B22
        [0,1, 1,1],  # M6: B21+B22
        [1,0, -1,0], # M7: B11-B21
    ]
    strassen_W = [
        [1,0, 0,1],  # M1: C11+=1, C22+=1... wait
        # C11 = M1+M4-M5+M7, C12 = M3+M6, C21 = M2+M5, C22 = M4-M3+M6+M7
        # W[r][lk*2+li] where lk is local k (0,1), li is local i (0,1)
        # C[i_local, k_local] gets contributions
        # For M1: contributes to C11 (li=0,lk=0) with +1
        # For M4: contributes to C11 (+1), C22 (+1)
        # For M5: contributes to C11 (-1), C21 (+1)
        # For M7: contributes to C11 (+1)
        # For M3: contributes to C12 (+1)
        # For M6: contributes to C12 (+1), C22 (+1)
        # For M2: contributes to C21 (+1)
        [1,0, 0,0],  # M1: C[0,0] += 1
        [0,0, 1,0],  # M2: C[1,0] += 1
        [0,1, 0,0],  # M3: C[0,1] += 1
        [1,0, 0,1],  # M4: C[0,0] += 1, C[1,1] += 1
        [-1,0, 1,0], # M5: C[0,0] -= 1, C[1,0] += 1
        [0,1, 0,1],  # M6: C[0,1] += 1, C[1,1] += 1
        [1,0, 0,0],  # M7: C[0,0] += 1
    ]
    
    rank = 4 * 7 + 8
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j0 in [0, 2]:
        for k0 in [0, 2]:
            for r in range(7):
                for li in range(2):
                    for lj in range(2):
                        gi = li
                        gj = j0 + lj
                        U[gi * m + gj][col] = strassen_U[r][li * 2 + lj]
                for lj in range(2):
                    for lk in range(2):
                        gj = j0 + lj
                        gk = k0 + lk
                        V[gj * p + gk][col] = strassen_V[r][lj * 2 + lk]
                for lk in range(2):
                    for li in range(2):
                        gk = k0 + lk
                        gi = li
                        W[gk * n + gi][col] = strassen_W[r][lk * 2 + li]
                col += 1
    
    # Schoolbook for k=4 column
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
    
    # Verify the decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check: sum_r U[a,r]*V[b,r]*W[c,r] should equal T[a,b,c]
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some valid (i,j,k)
    max_err = 0.0
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                val = np.sum(U_arr[a,:] * V_arr[b,:] * W_arr[c,:])
                err = abs(val - T[a,b,c])
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-9:
        # Decomposition is incorrect, fall back to pure schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][col_fb] = 1
                    V_fb[j*p+k][col_fb] = 1
                    W_fb[k*n+i][col_fb] = 1
                    col_fb += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
