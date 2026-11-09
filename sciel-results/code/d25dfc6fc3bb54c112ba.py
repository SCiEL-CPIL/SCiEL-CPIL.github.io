import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Strassen decomposition for 2x2x2
    # u, v, w are vectors of length 4
    # Flattening order: i*m+j, j*p+k, k*n+i
    # For 2x2x2:
    # u: i*2+j -> 0,1,2,3
    # v: j*2+k -> 0,1,2,3
    # w: k*2+i -> 0,1,2,3
    
    strassen_u = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2 (A21+A22) -> j=1, i=0,1
        [1, 0, 0, 0], # M3 (A11)
        [0, 0, 0, 1], # M4 (A22)
        [1, 1, 0, 0], # M5 (A11+A12) -> i=0, j=0,1
        [0, -1, 1, 0],# M6 (A21-A11) -> i=1,j=0; i=0,j=0? No.
                      # A21 is i=1,j=0. A11 is i=0,j=0.
                      # u6: u[2]=1 (i=1,j=0), u[0]=-1 (i=0,j=0)
        [0, 1, 0, -1] # M7 (A12-A22) -> i=0,j=1; i=1,j=1
    ]
    
    strassen_v = [
        [1, 0, 0, 1], # M1 (B11+B22)
        [1, 0, 0, 0], # M2 (B11)
        [0, 1, 0, 0], # M3 (B12-B22) -> B12 is j=0,k=1. B22 is j=1,k=1.
                      # v3: v[1]=1, v[3]=-1
        [0, 0, 1, -1],# M4 (B21-B11) -> B21 is j=1,k=0. B11 is j=0,k=0.
                      # v4: v[2]=1, v[0]=-1
        [0, 0, 0, 1], # M5 (B22)
        [1, 1, 0, 0], # M6 (B11+B12)
        [0, 0, 1, 1]  # M7 (B21+B22)
    ]
    
    strassen_w = [
        [1, 0, 0, 1], # M1 (C11+C22)
        [0, 0, 1, 0], # M2 (C21)
        [0, 1, 0, 0], # M3 (C12)
        [1, 0, 0, 0], # M4 (C11+C21? No. C11 has M4. C21 has M4.)
                      # w4: w[0]=1 (C11), w[2]=1 (C21)
        [0, 1, 0, 0], # M5 (C12-C11? No. C12 has M5. C11 has -M5.)
                      # w5: w[1]=1 (C12), w[0]=-1 (C11)
        [0, 0, 0, 1], # M6 (C22)
        [1, 0, 0, 0]  # M7 (C11)
    ]
    # Correction on W indices for Strassen:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # w is indexed by k*n+i. n=2.
    # C11: i=0, k=0 -> idx 0
    # C12: i=0, k=1 -> idx 2
    # C21: i=1, k=0 -> idx 1
    # C22: i=1, k=1 -> idx 3
    
    strassen_w = [
        [1, 0, 0, 1], # M1: C11(1), C22(1)
        [0, 1, 0, 0], # M2: C21(1)
        [0, 0, 1, 0], # M3: C12(1)
        [1, 1, 0, 0], # M4: C11(1), C21(1)
        [-1, 0, 1, 0],# M5: C11(-1), C12(1)
        [0, 0, 0, 1], # M6: C22(1)
        [1, 0, 0, 0]  # M7: C11(1)
    ]
    
    # Find disjoint 2x2x2 blocks
    blocks = []
    used = set()
    for i in range(n - 1):
        for j in range(m - 1):
            for k in range(p - 1):
                # Check if block (i,j,k) to (i+1,j+1,k+1) is free
                is_free = True
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            if (i+di, j+dj, k+dk) in used:
                                is_free = False
                                break
                        if not is_free: break
                    if not is_free: break
                if is_free:
                    blocks.append((i, j, k))
                    for di in range(2):
                        for dj in range(2):
                            for dk in range(2):
                                used.add((i+di, j+dj, k+dk))
    
    # Initialize U, V, W
    U = [[0] * (len(blocks) * 7 + (n*m*p - len(blocks)*8)) for _ in range(n * m)]
    V = [[0] * (len(blocks) * 7 + (n*m*p - len(blocks)*8)) for _ in range(m * p)]
    W = [[0] * (len(blocks) * 7 + (n*m*p - len(blocks)*8)) for _ in range(p * n)]
    
    col = 0
    # Add Strassen blocks
    for (i0, j0, k0) in blocks:
        for r in range(7):
            u_vec = strassen_u[r]
            v_vec = strassen_v[r]
            w_vec = strassen_w[r]
            
            for li in range(2):
                for lj in range(2):
                    gi = i0 + li
                    gj = j0 + lj
                    U[gi * m + gj][col] = u_vec[li * 2 + lj]
                    
            for lj in range(2):
                for lk in range(2):
                    gj = j0 + lj
                    gk = k0 + lk
                    V[gj * p + gk][col] = v_vec[lj * 2 + lk]
                    
            for lk in range(2):
                for li in range(2):
                    gk = k0 + lk
                    gi = i0 + li
                    W[gk * n + gi][col] = w_vec[lk * 2 + li]
            
            col += 1
            
    # Add remaining schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in used:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
                    
    # Trim unused columns
    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]
    
    return {"U": U, "V": V, "W": W}
