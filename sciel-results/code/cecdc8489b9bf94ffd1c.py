import time
import numpy as np

DIMENSIONS = [2, 2, 12]

def construct():
    start_time = time.time()
    deadline = start_time + 130

    n, m, p = DIMENSIONS
    num_blocks = p // m  # 6 blocks of 2 columns
    rank = num_blocks * 7  # 42

    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    # Strassen 7-product decomposition for 2x2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[C00,C01],[C10,C11]]
    # Local indices:
    # A: 0=a, 1=b, 2=c, 3=d
    # B: 0=e, 1=f, 2=g, 3=h
    # C: 0=C00, 1=C01, 2=C10, 3=C11

    strassen_terms = [
        # M1 = (a+d)(e+h)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (c+d)e
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = a(f-h)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4 = d(g-e)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5 = (a+b)h
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6 = (c-a)(e+f)
        ({0: -1, 2: 1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (b-d)(g+h)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[s]

            # U: A indices are global (0..3), no offset
            for a_idx, val in u_coeffs.items():
                U[a_idx, col] = val

            # V: B indices are local to the block
            # Local B index 0 -> B[0, 2t] -> global index 0*p + 2t = 2t
            # Local B index 1 -> B[0, 2t+1] -> global index 0*p + 2t+1 = 2t+1
            # Local B index 2 -> B[1, 2t] -> global index 1*p + 2t = 12+2t
            # Local B index 3 -> B[1, 2t+1] -> global index 1*p + 2t+1 = 12+2t+1
            for local_b, val in v_coeffs.items():
                if local_b == 0:
                    global_b = 2 * t
                elif local_b == 1:
                    global_b = 2 * t + 1
                elif local_b == 2:
                    global_b = m * p // 2 + 2 * t
                else:
                    global_b = m * p // 2 + 2 * t + 1
                V[global_b, col] = val

            # W: C indices are local to the block
            # Local C index 0 -> C[0, 2t] -> W index k*n+i = (2t)*2 + 0 = 4t
            # Local C index 1 -> C[0, 2t+1] -> W index (2t+1)*2 + 0 = 4t+2
            # Local C index 2 -> C[1, 2t] -> W index (2t)*2 + 1 = 4t+1
            # Local C index 3 -> C[1, 2t+1] -> W index (2t+1)*2 + 1 = 4t+3
            for local_c, val in w_coeffs.items():
                if local_c == 0:
                    global_w = (2 * t) * n + 0
                elif local_c == 1:
                    global_w = (2 * t + 1) * n + 0
                elif local_c == 2:
                    global_w = (2 * t) * n + 1
                else:
                    global_w = (2 * t + 1) * n + 1
                W[global_w, col] = val

    # Verify the decomposition
    def verify(U, V, W):
        T_recon = np.zeros((n*m, m*p, p*n))
        for r in range(U.shape[1]):
            T_recon += np.outer(np.outer(U[:, r], V[:, r]), W[:, r]).reshape(n*m, m*p, p*n)
        
        T_orig = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                u = i * m + j
                for k in range(p):
                    v = j * p + k
                    w = k * n + i
                    T_orig[u, v, w] = 1.0
        
        return np.allclose(T_recon, T_orig, atol=1e-10)

    if not verify(U, V, W):
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
