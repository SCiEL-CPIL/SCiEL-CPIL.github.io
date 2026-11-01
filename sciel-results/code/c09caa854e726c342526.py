import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 2, 4, 8
    rank = 56
    
    U = np.zeros((n * m, rank))  # (8, 56)
    V = np.zeros((m * p, rank))  # (32, 56)
    W = np.zeros((p * n, rank))  # (16, 56)
    
    # Strassen coefficients for 2x2 * 2x2
    # Each term: (x_coeffs, y_coeffs, z_coeffs)
    # x: (i', j') -> coeff for A[i', j']
    # y: (j', k') -> coeff for B[j', k']  
    # z: (i', k') -> coeff for C[i', k']
    
    strassen_terms = [
        # t=0: (A00+A11)(B00+B11) -> C00 += M, C11 += M
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # t=1: (A10+A11)B00 -> C10 += M, C11 -= M
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # t=2: A00(B01-B11) -> C01 += M, C11 += M
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # t=3: A11(B10-B00) -> C00 += M, C10 += M
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # t=4: (A00+A01)B11 -> C00 -= M, C01 += M
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # t=5: (A10-A00)(B00+B01) -> C11 += M
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # t=6: (A01-A11)(B10+B11) -> C00 += M
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    col = 0
    for a in range(2):  # A column block: cols 2a, 2a+1
        for b in range(4):  # B column block: cols 2b, 2b+1
            for t in range(7):
                x_c, y_c, z_c = strassen_terms[t]
                
                # Update U: index = i * m + j, where j = 2a + j'
                for (i_prime, j_prime), coeff in x_c.items():
                    u_idx = i_prime * m + (2 * a + j_prime)
                    U[u_idx, col] += coeff
                
                # Update V: index = j * p + k, where j = 2a + j', k = 2b + k'
                for (j_prime, k_prime), coeff in y_c.items():
                    v_idx = (2 * a + j_prime) * p + (2 * b + k_prime)
                    V[v_idx, col] += coeff
                
                # Update W: index = k * n + i, where k = 2b + k', i = i'
                for (i_prime, k_prime), coeff in z_c.items():
                    w_idx = (2 * b + k_prime) * n + i_prime
                    W[w_idx, col] += coeff
                
                col += 1
    
    # Verify correctness
    if time.time() < deadline:
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            # Fallback to schoolbook
            rank = n * m * p
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j, col] = 1
                        V[j*p+k, col] = 1
                        W[k*n+i, col] = 1
                        col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
