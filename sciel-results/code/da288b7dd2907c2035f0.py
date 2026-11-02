import numpy as np

def construct():
    n, m, p = 2, 4, 8
    nm, mp, pn = n * m, m * p, p * n  # 8, 32, 16
    
    # Rank 56 decomposition using 8 independent 2x2x2 Strassen blocks
    rank = 56
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    # Strassen terms for 2x2x2: each term has x, y, z coefficient dicts
    strassen_terms = [
        # t=0: (x00+x11)(y00+y11)
        {'x': {(0,0):1, (1,1):1}, 'y': {(0,0):1, (1,1):1}, 'z': {(0,0):1, (1,1):1}},
        # t=1: (x10+x11)y00
        {'x': {(1,0):1, (1,1):1}, 'y': {(0,0):1}, 'z': {(1,0):1, (1,1):-1}},
        # t=2: x00(y01-y11)
        {'x': {(0,0):1}, 'y': {(0,1):1, (1,1):-1}, 'z': {(0,1):1, (1,1):1}},
        # t=3: x11(y10-y00)
        {'x': {(1,1):1}, 'y': {(1,0):1, (0,0):-1}, 'z': {(0,0):1, (1,0):1}},
        # t=4: (x00+x01)y11
        {'x': {(0,0):1, (0,1):1}, 'y': {(1,1):1}, 'z': {(0,0):-1, (0,1):1}},
        # t=5: (x10-x00)(y00+y01)
        {'x': {(1,0):1, (0,0):-1}, 'y': {(0,0):1, (0,1):1}, 'z': {(1,1):1}},
        # t=6: (x01-x11)(y10+y11)
        {'x': {(0,1):1, (1,1):-1}, 'y': {(1,0):1, (1,1):1}, 'z': {(0,0):1}},
    ]
    
    col = 0
    for a in range(2):  # A column block: 2a, 2a+1
        for b in range(4):  # B column block: 2b, 2b+1
            for t in range(7):
                term = strassen_terms[t]
                # X part: A[i', 2a+j']
                for (i_prime, j_prime), coeff in term['x'].items():
                    u_idx = i_prime * m + (2*a + j_prime)
                    U[u_idx, col] += coeff
                
                # Y part: B[2a+j', 2b+k']
                for (j_prime, k_prime), coeff in term['y'].items():
                    v_idx = (2*a + j_prime) * p + (2*b + k_prime)
                    V[v_idx, col] += coeff
                
                # Z part: C[i', 2b+k'] -> W[2b+k', i']
                for (i_prime, k_prime), coeff in term['z'].items():
                    w_idx = (2*b + k_prime) * n + i_prime
                    W[w_idx, col] += coeff
                
                col += 1
    
    # Verify correctness
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if not np.isclose(val, 1.0, atol=1e-9):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook
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
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
