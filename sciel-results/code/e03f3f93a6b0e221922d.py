import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 2, 4, 8
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
    
    # Try rank-56 Strassen decomposition
    rank = 56
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Strassen terms for 2x2 * 2x2 -> 2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    strassen_coeffs = [
        # term, (a_idx, b_idx, c_idx, coeff)
        # a_idx: (i', j') in A block, b_idx: (j', k') in B block, c_idx: (i', k') in C block
        # M1
        [(0,0, (0,0), 1), (1,1, (0,0), 1), (0,0, (0,0), 1), (1,1, (0,0), 1),
         (0,0, (0,0), 1), (1,1, (1,1), 1)],
        # M2
        [(1,0, (0,0), 1), (1,1, (0,0), 1), (0,0, (0,0), 1),
         (1,0, (1,0), 1), (1,1, (1,1), -1)],
        # M3
        [(0,0, (0,0), 1), (0,1, (0,0), 1), (1,1, (0,0), -1),
         (0,1, (0,1), 1), (1,1, (1,1), 1)],
        # M4
        [(1,1, (0,0), 1), (1,0, (0,0), 1), (0,0, (0,0), -1),
         (0,0, (0,0), 1), (1,0, (1,0), 1)],
        # M5
        [(0,0, (0,0), 1), (0,1, (0,0), 1), (1,1, (0,0), 1),
         (0,0, (0,0), -1), (0,1, (0,1), 1)],
        # M6
        [(1,0, (0,0), 1), (0,0, (0,0), -1), (0,0, (0,0), 1), (0,1, (0,0), 1),
         (1,1, (1,1), 1)],
        # M7
        [(0,1, (0,0), 1), (1,1, (0,0), -1), (1,0, (0,0), 1), (1,1, (0,0), 1),
         (0,0, (0,0), 1)],
    ]
    
    col = 0
    for a in range(2):  # A column block: columns 2a, 2a+1
        for b in range(4):  # B column block: columns 2b, 2b+1
            for t in range(7):
                coeffs = strassen_coeffs[t]
                for (ai, bi, (ci, ki), coeff) in coeffs:
                    # A index: row ai, col (2a + bi)
                    u_idx = ai * m + (2*a + bi)
                    U[u_idx, col] += coeff
                    
                    # B index: row (2a + bi), col (2b + ki)
                    v_idx = (2*a + bi) * p + (2*b + ki)
                    V[v_idx, col] += coeff
                    
                    # C index: row ci, col (2b + ki) -> W index = (2b+ki)*n + ci
                    w_idx = (2*b + ki) * n + ci
                    W[w_idx, col] += coeff
                
                col += 1
    
    # Verify the decomposition
    valid = True
    if time.time() < deadline:
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
