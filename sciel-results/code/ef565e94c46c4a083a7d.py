import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 2, 4, 8
    
    # Strassen decomposition: split j into 2 pairs of 2, k into 4 pairs of 2
    # 8 independent 2x2 * 2x2 blocks, each rank 7, total rank 56
    rank = 56
    U = np.zeros((n * m, rank))  # (8, 56)
    V = np.zeros((m * p, rank))  # (32, 56)
    W = np.zeros((p * n, rank))  # (16, 56)
    
    # Strassen terms for 2x2 * 2x2
    # Each term: (U_coeffs, V_coeffs, W_coeffs)
    # U: (i, j) -> coeff, V: (j, k) -> coeff, W: (i, k) -> coeff
    strassen_terms = [
        # M1 = (a00+a11)(b00+b11) -> c00 += M, c11 += M
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2 = (a10+a11)b00 -> c10 += M, c11 -= M
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3 = a00(b01-b11) -> c01 += M, c11 += M
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4 = a11(b10-b00) -> c00 += M, c10 += M
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M5 = (a00+a01)b11 -> c00 -= M, c01 += M
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6 = (a10-a00)(b00+b01) -> c11 += M
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7 = (a01-a11)(b10+b11) -> c00 += M
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    col = 0
    for jp in range(2):      # j-pair: 0 -> j in {0,1}, 1 -> j in {2,3}
        for kp in range(4):  # k-pair: 0->k in {0,1}, 1->{2,3}, 2->{4,5}, 3->{6,7}
            for t in range(7):
                u_c, v_c, w_c = strassen_terms[t]
                
                # U: index = i*m + j, where j = 2*jp + j'
                for (i_prime, j_prime), coeff in u_c.items():
                    u_idx = i_prime * m + (2 * jp + j_prime)
                    U[u_idx, col] += coeff
                
                # V: index = j*p + k, where j = 2*jp + j', k = 2*kp + k'
                for (j_prime, k_prime), coeff in v_c.items():
                    v_idx = (2 * jp + j_prime) * p + (2 * kp + k_prime)
                    V[v_idx, col] += coeff
                
                # W: index = k*n + i, where k = 2*kp + k', i = i'
                for (i_prime, k_prime), coeff in w_c.items():
                    w_idx = (2 * kp + k_prime) * n + i_prime
                    W[w_idx, col] += coeff
                
                col += 1
    
    # Verify correctness
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
