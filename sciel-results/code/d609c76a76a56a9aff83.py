import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 6x6 matrix by a 6x6 matrix.
    Uses a 2x2 block decomposition with 3x3 blocks, applying Strassen's 7-product algorithm
    at the outer block level and schoolbook (rank-27) for each 3x3 block multiplication.
    Total rank = 7 * 27 = 189.
    """
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # all 6

    # 2x2 block structure: each 6x6 matrix is split into four 3x3 blocks.
    # Strassen at the 2x2 block level (7 products of 3x3 matrices).
    # Each 3x3 multiplication uses schoolbook (27 scalar products).
    # Total rank = 7 * 27 = 189.

    rank = 189
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    col = 0

    # Strassen steps for 2x2 matrix multiplication
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # Block indices: 0, 1
    # A blocks: A[ib, jb]
    # B blocks: B[jb, kb]
    # C blocks: C[ib, kb]
    
    strassen_steps = [
        # (X_blocks, Y_blocks, C_contribution)
        {
            'X': {(0,0): 1, (1,1): 1},
            'Y': {(0,0): 1, (1,1): 1},
            'C': {(0,0): 1, (1,1): 1}
        },
        {
            'X': {(1,0): 1, (1,1): 1},
            'Y': {(0,0): 1},
            'C': {(1,0): 1, (1,1): -1}
        },
        {
            'X': {(0,0): 1},
            'Y': {(0,1): 1, (1,1): -1},
            'C': {(0,1): 1, (1,1): 1}
        },
        {
            'X': {(1,1): 1},
            'Y': {(1,0): 1, (0,0): -1},
            'C': {(0,0): 1, (1,0): 1}
        },
        {
            'X': {(0,0): 1, (0,1): 1},
            'Y': {(1,1): 1},
            'C': {(0,0): -1, (0,1): 1}
        },
        {
            'X': {(1,0): 1, (0,0): -1},
            'Y': {(0,0): 1, (0,1): 1},
            'C': {(1,1): 1}
        },
        {
            'X': {(0,1): 1, (1,1): -1},
            'Y': {(1,0): 1, (1,1): 1},
            'C': {(0,0): 1}
        }
    ]

    block_size = 3

    for step in strassen_steps:
        X_coeffs = step['X']
        Y_coeffs = step['Y']
        C_coeffs = step['C']
        
        # For each of the 27 scalar products in the 3x3 matrix multiplication
        # X[r, s] * Y[s, c] -> M[r, c]
        for r in range(block_size):
            for s in range(block_size):
                for c in range(block_size):
                    
                    # U part: X[r, s] = sum_{ib,jb} X_coeffs[(ib,jb)] * A[ib*3+r, jb*3+s]
                    for (ib, jb), x_coeff in X_coeffs.items():
                        u_row = ib * block_size + r
                        u_col = jb * block_size + s
                        u_idx = u_row * m + u_col
                        U[u_idx, col] += x_coeff
                    
                    # V part: Y[s, c] = sum_{jb,kb} Y_coeffs[(jb,kb)] * B[jb*3+s, kb*3+c]
                    for (jb, kb), y_coeff in Y_coeffs.items():
                        v_row = jb * block_size + s
                        v_col = kb * block_size + c
                        v_idx = v_row * p + v_col
                        V[v_idx, col] += y_coeff
                    
                    # W part: M[r, c] contributes to C blocks
                    # C[ib, kb, r, c] += C_coeffs[(ib,kb)] * M[r,c]
                    # W index: k*n + i, where k = kb*3+c, i = ib*3+r
                    for (ib, kb), c_coeff in C_coeffs.items():
                        w_k = kb * block_size + c
                        w_i = ib * block_size + r
                        w_idx = w_k * n + w_i
                        W[w_idx, col] += c_coeff
                    
                    col += 1
        
        if time.time() > deadline:
            break

    # Verification
    # Check all 216 valid entries
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U[u_idx, :], V[v_idx, :]) # Wait, this is sum U*V, but we need sum U*V*W
                # The term is sum_r U[u, r] * V[v, r] * W[w, r]
                # This is not a simple dot product of U and V.
                # It is an element-wise product of three vectors, summed.
                # val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                if abs(val - 1.0) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    # Check some invalid entries
    if valid:
        import random
        random.seed(42)
        for _ in range(1000):
            u_idx = random.randint(0, n*m - 1)
            v_idx = random.randint(0, m*p - 1)
            w_idx = random.randint(0, p*n - 1)
            
            # Check if this is a valid chain
            i = u_idx // m
            j = u_idx % m
            j_v = v_idx // p
            k = v_idx % p
            k_w = w_idx // n
            i_w = w_idx % n
            
            if j == j_v and k == k_w and i == i_w:
                continue # Skip valid chains
            
            val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
            if abs(val) > 1e-10:
                valid = False
                break

    if valid:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook 216
        rank_fallback = n * m * p
        U_f = np.zeros((n * m, rank_fallback), dtype=np.float64)
        V_f = np.zeros((m * p, rank_fallback), dtype=np.float64)
        W_f = np.zeros((p * n, rank_fallback), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j, col] = 1
                    V_f[j * p + k, col] = 1
                    W_f[k * n + i, col] = 1
                    col += 1
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
