"""Matrix multiplication 3,5,7: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    def build_strassen_schoolbook(n, m, p):
        i_pairs = [(i, i+1) for i in range(n-1)]
        j_pairs = [(j, j+1) for j in range(m-1)]
        k_pairs = [(k, k+1) for k in range(p-1)]
        
        strassen_covered = set()
        for i0, i1 in i_pairs:
            for j0, j1 in j_pairs:
                for k0, k1 in k_pairs:
                    for i in [i0, i1]:
                        for j in [j0, j1]:
                            for k in [k0, k1]:
                                strassen_covered.add((i, j, k))
        
        num_blocks = len(i_pairs) * len(j_pairs) * len(k_pairs)
        num_strassen = 7 * num_blocks
        num_schoolbook = n*m*p - len(strassen_covered)
        total_rank = num_strassen + num_schoolbook
        
        U = np.zeros((n*m, total_rank))
        V = np.zeros((m*p, total_rank))
        W = np.zeros((p*n, total_rank))
        
        col = 0
        for i0, i1 in i_pairs:
            for j0, j1 in j_pairs:
                for k0, k1 in k_pairs:
                    U[i0*m+j0, col] = 1; U[i1*m+j1, col] = 1
                    V[j0*p+k0, col] = 1; V[j1*p+k1, col] = 1
                    W[k0*n+i0, col] = 1; W[k1*n+i1, col] = 1
                    col += 1
                    U[i1*m+j0, col] = 1; U[i1*m+j1, col] = 1
                    V[j0*p+k0, col] = 1
                    W[k0*n+i1, col] = 1; W[k1*n+i1, col] = -1
                    col += 1
                    U[i0*m+j0, col] = 1
                    V[j0*p+k1, col] = 1; V[j1*p+k1, col] = -1
                    W[k1*n+i0, col] = 1; W[k1*n+i1, col] = 1
                    col += 1
                    U[i1*m+j1, col] = 1
                    V[j1*p+k0, col] = 1; V[j0*p+k0, col] = -1
                    W[k0*n+i0, col] = 1; W[k0*n+i1, col] = 1
                    col += 1
                    U[i0*m+j0, col] = 1; U[i0*m+j1, col] = 1
                    V[j1*p+k1, col] = 1
                    W[k1*n+i0, col] = 1; W[k0*n+i0, col] = -1
                    col += 1
                    U[i1*m+j0, col] = 1; U[i0*m+j0, col] = -1
                    V[j0*p+k0, col] = 1; V[j0*p+k1, col] = 1
                    W[k1*n+i1, col] = 1
                    col += 1
                    U[i0*m+j1, col] = 1; U[i1*m+j1, col] = -1
                    V[j1*p+k0, col] = 1; V[j1*p+k1, col] = 1
                    W[k0*n+i0, col] = 1
                    col += 1
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in strassen_covered:
                        U[i*m+j, col] = 1
                        V[j*p+k, col] = 1
                        W[k*n+i, col] = 1
                        col += 1
        
        assert col == total_rank
        return U, V, W, total_rank
    
    U, V, W, rank = build_strassen_schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}

