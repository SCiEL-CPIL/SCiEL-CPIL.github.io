import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 145.0 # 5s margin
    
    n, m, p = 3, 5, 5
    
    # Partitions
    I_parts = [ [0, 1], [2] ]
    J_parts = [ [0, 1], [2, 3], [4] ]
    K_parts = [ [0, 1], [2, 3], [4] ]
    
    # Identify Strassen blocks (2x2x2)
    strassen_blocks = []
    for I in I_parts:
        if len(I) != 2: continue
        for J in J_parts:
            if len(J) != 2: continue
            for K in K_parts:
                if len(K) != 2: continue
                strassen_blocks.append((I, J, K))
                
    # Pre-allocate
    max_rank = n * m * p
    U = np.zeros((n * m, max_rank), dtype=int)
    V = np.zeros((m * p, max_rank), dtype=int)
    W = np.zeros((p * n, max_rank), dtype=int)
    rank = 0
    
    # Add Strassen terms
    for I, J, K in strassen_blocks:
        if time.time() > deadline:
            break
        i0, i1 = I
        j0, j1 = J
        k0, k1 = K
        
        # M1
        U[i0*m+j0, rank] = 1
        U[i1*m+j1, rank] = 1
        V[j0*p+k0, rank] = 1
        V[j1*p+k1, rank] = 1
        W[k0*n+i0, rank] = 1
        W[k1*n+i1, rank] = 1
        rank += 1
        
        # M2
        U[i1*m+j0, rank] = 1
        U[i1*m+j1, rank] = 1
        V[j0*p+k0, rank] = 1
        W[k0*n+i1, rank] = 1
        W[k1*n+i1, rank] = -1
        rank += 1
        
        # M3
        U[i0*m+j0, rank] = 1
        V[j0*p+k1, rank] = 1
        V[j1*p+k1, rank] = -1
        W[k1*n+i0, rank] = 1
        W[k1*n+i1, rank] = 1
        rank += 1
        
        # M4
        U[i1*m+j1, rank] = 1
        V[j1*p+k0, rank] = 1
        V[j0*p+k0, rank] = -1
        W[k0*n+i0, rank] = 1
        W[k0*n+i1, rank] = 1
        rank += 1
        
        # M5
        U[i0*m+j0, rank] = 1
        U[i0*m+j1, rank] = 1
        V[j1*p+k1, rank] = 1
        W[k0*n+i0, rank] = -1
        W[k1*n+i0, rank] = 1
        rank += 1
        
        # M6
        U[i1*m+j0, rank] = 1
        U[i0*m+j0, rank] = -1
        V[j0*p+k0, rank] = 1
        V[j0*p+k1, rank] = 1
        W[k1*n+i1, rank] = 1
        rank += 1
        
        # M7
        U[i0*m+j1, rank] = 1
        U[i1*m+j1, rank] = -1
        V[j1*p+k0, rank] = 1
        V[j1*p+k1, rank] = 1
        W[k0*n+i0, rank] = 1
        rank += 1
        
    # Add standard terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
                in_block = False
                for I, J, K in strassen_blocks:
                    if i in I and j in J and k in K:
                        in_block = True
                        break
                if not in_block:
                    U[i*m+j, rank] = 1
                    V[j*p+k, rank] = 1
                    W[k*n+i, rank] = 1
                    rank += 1
                    
    U = U[:, :rank]
    V = V[:, :rank]
    W = W[:, :rank]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
