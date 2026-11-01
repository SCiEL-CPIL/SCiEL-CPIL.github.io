import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank

    # Strassen decomposition for 2x2x2
    # Returns U, V, W for a 2x2 block
    # Local indices:
    # A: 0,1,2,3
    # B: 0,1,2,3
    # C: 0,1,2,3
    # W index in global: k*n + i
    
    rank = 56
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    for u in range(2):
        for v in range(4):
            base_col = 7 * (4 * u + v)
            
            # A block indices (global)
            # A is 2x4. u is col block.
            # A[0, 2u], A[0, 2u+1], A[1, 2u], A[1, 2u+1]
            a11 = 0 * m + 2 * u
            a12 = 0 * m + 2 * u + 1
            a21 = 1 * m + 2 * u
            a22 = 1 * m + 2 * u + 1
            
            # B block indices (global)
            # B is 4x8. u is row block, v is col block.
            # B[2u, 2v], B[2u, 2v+1], B[2u+1, 2v], B[2u+1, 2v+1]
            b11 = (2 * u) * p + 2 * v
            b12 = (2 * u) * p + 2 * v + 1
            b21 = (2 * u + 1) * p + 2 * v
            b22 = (2 * u + 1) * p + 2 * v + 1
            
            # C block indices (global W index: k*n + i)
            # C is 2x8. v is col block.
            # C[0, 2v], C[0, 2v+1], C[1, 2v], C[1, 2v+1]
            # W index = k * n + i
            c11 = (2 * v) * n + 0
            c12 = (2 * v + 1) * n + 0
            c21 = (2 * v) * n + 1
            c22 = (2 * v + 1) * n + 1
            
            # s=0: M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
            U[a11, base_col] += 1
            U[a22, base_col] += 1
            V[b11, base_col] += 1
            V[b22, base_col] += 1
            W[c11, base_col] += 1
            W[c22, base_col] += 1
            
            # s=1: M2 = (A21+A22)B11 -> C21 += M2, C22 -= M2
            U[a21, base_col + 1] += 1
            U[a22, base_col + 1] += 1
            V[b11, base_col + 1] += 1
            W[c21, base_col + 1] += 1
            W[c22, base_col + 1] -= 1
            
            # s=2: M3 = A11(B12-B22) -> C12 += M3, C22 += M3
            U[a11, base_col + 2] += 1
            V[b12, base_col + 2] += 1
            V[b22, base_col + 2] -= 1
            W[c12, base_col + 2] += 1
            W[c22, base_col + 2] += 1
            
            # s=3: M4 = A22(B21-B11) -> C11 += M4, C21 += M4
            U[a22, base_col + 3] += 1
            V[b21, base_col + 3] += 1
            V[b11, base_col + 3] -= 1
            W[c11, base_col + 3] += 1
            W[c21, base_col + 3] += 1
            
            # s=4: M5 = (A11+A12)B22 -> C11 -= M5, C12 += M5
            U[a11, base_col + 4] += 1
            U[a12, base_col + 4] += 1
            V[b22, base_col + 4] += 1
            W[c11, base_col + 4] -= 1
            W[c12, base_col + 4] += 1
            
            # s=5: M6 = (A21-A11)(B11+B12) -> C22 += M6
            U[a21, base_col + 5] += 1
            U[a11, base_col + 5] -= 1
            V[b11, base_col + 5] += 1
            V[b12, base_col + 5] += 1
            W[c22, base_col + 5] += 1
            
            # s=6: M7 = (A12-A22)(B21+B22) -> C11 += M7
            U[a12, base_col + 6] += 1
            U[a22, base_col + 6] -= 1
            V[b21, base_col + 6] += 1
            V[b22, base_col + 6] += 1
            W[c11, base_col + 6] += 1
            
    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if not np.isclose(val, 1.0, atol=1e-9):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
            
    if not valid:
        U, V, W, rank = schoolbook()
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
