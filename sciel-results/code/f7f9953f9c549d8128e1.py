import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x6 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    """
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 8, 24, 12
    
    # Rank 42 decomposition using Strassen
    # Split A (2x4) into A1 (2x2) and A2 (2x2)
    # Split B (4x6) into 3 blocks of (4x2): B1, B2, B3
    # Each block computation C_b = A1*B1_b + A2*B2_b
    # Use Strassen (rank 7) for each 2x2x2 product.
    # 2 products per block * 3 blocks = 6 products * 7 = 42 rank.
    
    rank = 42
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    r_idx = 0
    
    # Strassen coefficients for 2x2x2
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y10-Y11)
    # M4 = X11(Y00+Y11)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y10)
    # M7 = (X01-X11)(Y11+Y10)
    
    # Output Z:
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # Define Strassen terms as (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict (i,j) -> val
    # V_coeffs: dict (j,k) -> val
    # W_coeffs: dict (k,i) -> val
    
    strassen_terms = [
        # M1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1}), # Z10 gets M2. Z10 is (1,0). W[k=0, i=1]
        # M3
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,0):1, (1,1):1}), # Z01 gets M3. Z01 is (0,1). W[k=1, i=0]. Z11 gets M3. Z11 is (1,1). W[k=1, i=1]
        # M4
        ({(1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (0,1):1}), # Z00 gets M4. Z10 gets M4.
        # M5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,1):1}), # Z00 gets -M5. Z01 gets M5.
        # M6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}, {(1,1):1}), # Z11 gets M6.
        # M7
        ({(0,1):1, (1,1):-1}, {(1,1):1, (1,0):1}, {(0,0):1, (1,1):1}), # Z00 gets M7. Z11 gets M7.
    ]
    
    # Helper to add a Strassen product to the global matrices
    # a_offset: (i_start, j_start) for A block
    # b_offset: (j_start, k_start) for B block
    # c_offset: (i_start, k_start) for C block (same as a_offset for i, b_offset for k)
    
    def add_strassen_product(a_i, a_j, b_j, b_k, c_i, c_k):
        nonlocal r_idx
        for u_c, v_c, w_c in strassen_terms:
            for (di, dj), val in u_c.items():
                i = a_i + di
                j = a_j + dj
                U[i * m + j][r_idx] += val
            for (dj, dk), val in v_c.items():
                j = b_j + dj
                k = b_k + dk
                V[j * p + k][r_idx] += val
            for (dk, di), val in w_c.items():
                k = c_k + dk
                i = c_i + di
                W[k * n + i][r_idx] += val
            r_idx += 1

    # Block 0: B cols 0-1
    # A1 (0,0) * B1 (0,0)
    add_strassen_product(0, 0, 0, 0, 0, 0)
    # A2 (0,2) * B2 (2,0)
    add_strassen_product(0, 2, 2, 0, 0, 0)
    
    # Block 1: B cols 2-3
    # A1 (0,0) * B1 (0,2)
    add_strassen_product(0, 0, 0, 2, 0, 2)
    # A2 (0,2) * B2 (2,2)
    add_strassen_product(0, 2, 2, 2, 0, 2)
    
    # Block 2: B cols 4-5
    # A1 (0,0) * B1 (0,4)
    add_strassen_product(0, 0, 0, 4, 0, 4)
    # A2 (0,2) * B2 (2,4)
    add_strassen_product(0, 2, 2, 4, 0, 4)
    
    # Verify
    def verify_numpy(U, V, W, rank):
        Ua = np.array(U, dtype=np.float64)
        Va = np.array(V, dtype=np.float64)
        Wa = np.array(W, dtype=np.float64)
        
        T = np.zeros((n, m, p), dtype=np.float64)
        for r in range(rank):
            u_c = Ua[:, r].reshape(n, m)
            v_c = Va[:, r].reshape(m, p)
            w_c = Wa[:, r].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += u_c[i, j] * v_c[j, k] * w_c[k, i]
        
        return np.allclose(T, np.ones((n, m, p)), atol=1e-9)

    if verify_numpy(U, V, W, rank):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
