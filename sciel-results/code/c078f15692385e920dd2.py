import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def _strassen_2x2():
    """Return U, V, W for 2x2 matrix multiplication with rank 7 (Strassen)."""
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # Strassen's 7 multiplications for 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    # M1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_6x6_rank189():
    """
    Builds 6x6 multiplication using 3x3 grid of 2x2 blocks with Strassen (rank 7).
    """
    n, m, p = 6, 6, 6
    rank = 189
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    U2, V2, W2 = _strassen_2x2()
    
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                base_col = (bi * 3 + bj) * 3 + bk
                col_offset = base_col * 7
                
                for r in range(7):
                    # U: A[bi,bk] -> A[2bi+lr, 2bk+lc]
                    # Index in U: (2bi+lr)*m + (2bk+lc)
                    for lr in range(2):
                        for lc in range(2):
                            u_row = (2 * bi + lr) * m + (2 * bk + lc)
                            U[u_row, col_offset + r] += U2[lr * 2 + lc, r]
                            
                    # V: B[bj,bk] -> B[2bj+lr, 2bk+lc]
                    # Index in V: (2bj+lr)*p + (2bk+lc)
                    for lr in range(2):
                        for lc in range(2):
                            v_row = (2 * bj + lr) * p + (2 * bk + lc)
                            V[v_row, col_offset + r] += V2[lr * 2 + lc, r]
                            
                    # W: C[bi,bk] -> C[2bi+lr, 2bk+lc]
                    # Index in W: p*(2bk+lc) + (2bi+lr)  (Trace convention: k*n+i)
                    # Wait, W[k*n+i, r] where k is col_C, i is row_C.
                    # So W[ (2bk+lc)*n + (2bi+lr), col ]
                    for lr in range(2):
                        for lc in range(2):
                            w_row = p * (2 * bk + lc) + (2 * bi + lr)
                            W[w_row, col_offset + r] += W2[lr * 2 + lc, r]
                        
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Build rank-189 decomposition
    U, V, W = _build_6x6_rank189()
    
    if time.time() > deadline - 10:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
