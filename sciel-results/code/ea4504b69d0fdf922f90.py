import time
import numpy as np

DIMENSIONS = [2, 2, 12]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    # Current best known rank for 2x2x12 is 42 (6 blocks of 2x2x2 Strassen)
    # We construct this explicitly.
    
    num_blocks = p // 2  # 6 blocks
    rank = num_blocks * 7  # 42
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 7-product for 2x2x2
    # A indices: 0=a, 1=b, 2=c, 3=d
    # B local indices: 0=e, 1=f, 2=g, 3=h
    # C local indices: 0=C00, 1=C01, 2=C10, 3=C11
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
            
            for a_idx, val in u_coeffs.items():
                U[a_idx, col] = val
            
            for local_b, val in v_coeffs.items():
                # Map local B index to global B index
                # B is 2 x 12.
                # Block t covers columns 2*t and 2*t+1.
                # Local 0 -> row 0, col 2*t
                # Local 1 -> row 1, col 2*t
                # Local 2 -> row 0, col 2*t+1
                # Local 3 -> row 1, col 2*t+1
                # V index is (row_B, col_B) flattened.
                # Row 0: 0..11
                # Row 1: 12..23
                if local_b == 0:
                    global_b = 0 * p + (2 * t)
                elif local_b == 1:
                    global_b = 1 * p + (2 * t)
                elif local_b == 2:
                    global_b = 0 * p + (2 * t + 1)
                else:
                    global_b = 1 * p + (2 * t + 1)
                V[global_b, col] = val
            
            for local_c, val in w_coeffs.items():
                # Map local C index to global C index
                # C is 12 x 2.
                # Block t covers rows 2*t and 2*t+1 of C.
                # Local 0 -> C[2*t, 0]
                # Local 1 -> C[2*t+1, 0]
                # Local 2 -> C[2*t, 1]
                # Local 3 -> C[2*t+1, 1]
                # W index is (row_C, col_C) flattened.
                # Row 0: 0..1
                # Row 1: 2..3
                # ...
                # Row 11: 22..23
                # W[k*n+i, r] convention?
                # Task says: W[k*n+i,r]
                # This means W index is (k, i) where k is col of B (row of C) and i is row of A (col of C).
                # So W is flattened as k * n + i.
                # k = 2*t or 2*t+1
                # i = 0 or 1
                
                if local_c == 0:
                    # C00: row 2*t, col 0
                    global_w = (2 * t) * n + 0
                elif local_c == 1:
                    # C01: row 2*t+1, col 0
                    global_w = (2 * t + 1) * n + 0
                elif local_c == 2:
                    # C10: row 2*t, col 1
                    global_w = (2 * t) * n + 1
                else:
                    # C11: row 2*t+1, col 1
                    global_w = (2 * t + 1) * n + 1
                W[global_w, col] = val
    
    # Verify
    if time.time() < deadline:
        T_recon = np.zeros((n * m, m * p, p * n))
        for r in range(rank):
            T_recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        
        T_orig = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_orig[i * m + j, j * p + k, k * n + i] = 1.0
        
        if not np.allclose(T_recon, T_orig, atol=1e-10):
            # Fallback to schoolbook if verification fails (should not happen)
            rank = n * m * p
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, col] = 1
                        V[j * p + k, col] = 1
                        W[k * n + i, col] = 1
                        col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
