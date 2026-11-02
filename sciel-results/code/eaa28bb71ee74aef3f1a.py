import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
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
    
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_slice = U[i*m+j, :rank]
                    v_slice = V[j*p+k, :rank]
                    w_slice = W[k*n+i, :rank]
                    val = np.dot(u_slice, v_slice * w_slice)
                    if not np.isclose(val, 1.0, atol=1e-9):
                        return False
        return True
    
    # Strassen coefficients for 2x2 * 2x2 -> 2x2
    # Each term: x_coeffs maps (i', j') to coeff for A[i', j']
    #             y_coeffs maps (j', k') to coeff for B[j', k']
    #             z_coeffs maps (i', k') to coeff for C[i', k']
    strassen_terms = [
        # t=0: M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
        {'x': {(0,0):1, (1,1):1}, 'y': {(0,0):1, (1,1):1}, 'z': {(0,0):1, (1,1):1}},
        # t=1: M2 = (A21+A22)B11 -> C21 += M2, C22 -= M2
        {'x': {(1,0):1, (1,1):1}, 'y': {(0,0):1}, 'z': {(1,0):1, (1,1):-1}},
        # t=2: M3 = A11(B12-B22) -> C12 += M3, C22 += M3
        {'x': {(0,0):1}, 'y': {(0,1):1, (1,1):-1}, 'z': {(0,1):1, (1,1):1}},
        # t=3: M4 = A22(B21-B11) -> C11 += M4, C21 += M4
        {'x': {(1,1):1}, 'y': {(1,0):1, (0,0):-1}, 'z': {(0,0):1, (1,0):1}},
        # t=4: M5 = (A11+A12)B22 -> C11 -= M5, C12 += M5
        {'x': {(0,0):1, (0,1):1}, 'y': {(1,1):1}, 'z': {(0,0):-1, (0,1):1}},
        # t=5: M6 = (A21-A11)(B11+B12) -> C22 += M6
        {'x': {(1,0):1, (0,0):-1}, 'y': {(0,0):1, (0,1):1}, 'z': {(1,1):1}},
        # t=6: M7 = (A12-A22)(B21+B22) -> C11 += M7
        {'x': {(0,1):1, (1,1):-1}, 'y': {(1,0):1, (1,1):1}, 'z': {(0,0):1}},
    ]
    
    rank = 56
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    for a in range(2):  # A block column: columns 2a, 2a+1
        for b in range(4):  # B column block: columns 2b, 2b+1
            for t in range(7):
                term = strassen_terms[t]
                
                # X part: A[i', 2a+j'] -> U index = i' * m + (2a + j')
                for (i_prime, j_prime), coeff in term['x'].items():
                    u_idx = i_prime * m + (2*a + j_prime)
                    U[u_idx, col] += coeff
                
                # Y part: B[2a+j', 2b+k'] -> V index = (2a + j') * p + (2b + k')
                for (j_prime, k_prime), coeff in term['y'].items():
                    v_idx = (2*a + j_prime) * p + (2*b + k_prime)
                    V[v_idx, col] += coeff
                
                # Z part: C[i', 2b+k'] -> W index = (2b + k') * n + i'
                for (i_prime, k_prime), coeff in term['z'].items():
                    w_idx = (2*b + k_prime) * n + i_prime
                    W[w_idx, col] += coeff
                
                col += 1
    
    if not verify(U, V, W, rank):
        U, V, W, rank = schoolbook()
    
    if time.time() > deadline:
        pass  # Return whatever we have
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
