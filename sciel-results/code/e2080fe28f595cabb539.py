import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def verify(U, V, W, n, m, p, tol=1e-9):
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = 2, 4, 8
    rank = 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def add_strassen_block(a_base, b_base, c_base, start_rank):
        ai, aj = a_base
        bj, bk = b_base
        ck, ci = c_base
        
        r = start_rank + 0
        U[(ai+0)*4 + (aj+0), r] += 1
        U[(ai+1)*4 + (aj+1), r] += 1
        V[(bj+0)*8 + (bk+0), r] += 1
        V[(bj+1)*8 + (bk+1), r] += 1
        W[(ck+0)*2 + (ci+0), r] += 1
        W[(ck+1)*2 + (ci+1), r] += 1
        
        r = start_rank + 1
        U[(ai+1)*4 + (aj+0), r] += 1
        U[(ai+1)*4 + (aj+1), r] += 1
        V[(bj+0)*8 + (bk+0), r] += 1
        W[(ck+0)*2 + (ci+1), r] += 1
        W[(ck+1)*2 + (ci+1), r] += -1
        
        r = start_rank + 2
        U[(ai+0)*4 + (aj+0), r] += 1
        V[(bj+1)*8 + (bk+0), r] += 1
        V[(bj+1)*8 + (bk+1), r] += -1
        W[(ck+1)*2 + (ci+0), r] += 1
        W[(ck+1)*2 + (ci+1), r] += 1
        
        r = start_rank + 3
        U[(ai+1)*4 + (aj+1), r] += 1
        V[(bj+0)*8 + (bk+1), r] += 1
        V[(bj+1)*8 + (bk+1), r] += 1
        W[(ck+0)*2 + (ci+0), r] += 1
        W[(ck+1)*2 + (ci+1), r] += 1
        
        r = start_rank + 4
        U[(ai+0)*4 + (aj+0), r] += 1
        U[(ai+1)*4 + (aj+0), r] += 1
        V[(bj+1)*8 + (bk+1), r] += 1
        W[(ck+0)*2 + (ci+0), r] += -1
        W[(ck+1)*2 + (ci+1), r] += -1
        
        r = start_rank + 5
        U[(ai+1)*4 + (aj+1), r] += 1
        U[(ai+0)*4 + (aj+0), r] += -1
        V[(bj+0)*8 + (bk+0), r] += 1
        V[(bj+1)*8 + (bk+0), r] += 1
        W[(ck+0)*2 + (ci+1), r] += 1
        W[(ck+1)*2 + (ci+1), r] += 1
        
        r = start_rank + 6
        U[(ai+0)*4 + (aj+1), r] += 1
        U[(ai+1)*4 + (aj+1), r] += -1
        V[(bj+1)*8 + (bk+1), r] += 1
        V[(bj+0)*8 + (bk+1), r] += 1
        W[(ck+0)*2 + (ci+0), r] += -1
        W[(ck+1)*2 + (ci+0), r] += 1

    add_strassen_block((0,0), (0,0), (0,0), 0)
    add_strassen_block((0,2), (2,0), (0,0), 7)
    add_strassen_block((0,0), (0,2), (2,0), 14)
    add_strassen_block((0,2), (2,2), (2,0), 21)
    add_strassen_block((0,0), (0,4), (4,0), 28)
    add_strassen_block((0,2), (2,4), (4,0), 35)
    add_strassen_block((0,0), (0,6), (6,0), 42)
    add_strassen_block((0,2), (2,6), (6,0), 49)
    
    return {"U": U, "V": V, "W": W}
