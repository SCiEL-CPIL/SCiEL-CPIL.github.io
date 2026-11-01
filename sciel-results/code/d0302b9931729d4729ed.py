import numpy as np
import time

DIMENSIONS = [2, 2, 6]

def _build_rank21_strassen():
    n, m, p = DIMENSIONS
    rank = 21
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen decomposition for 2x2x2
    # U[i, r] corresponds to linear combination of A entries
    # V[j, r] corresponds to linear combination of B entries
    # W[k, r] corresponds to linear combination of C entries
    #
    # Standard Strassen 7 terms:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)(b00)
    # M3 = (a00)(b10-b11)
    # M4 = (a11)(b01-b00)
    # M5 = (a00+a01)(b11)
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # Indices: A is 2x2, so a_ij where i is row, j is col
    # In U, index i (0..3) corresponds to A[i//2][i%2] or similar
    # Let's use standard flat indexing: A[i*2+j]
    # So U[0]=a00, U[1]=a01, U[2]=a10, U[3]=a11
    # B is 2x6, so b_jk where j is row, k is col
    # In V, index j (0..11) corresponds to B[j//6][j%6]
    # So for a 2x2 block (columns k, k+1):
    # V[2b*2+0] = b0,k, V[2b*2+1] = b0,k+1
    # V[2b*2+2] = b1,k, V[2b*2+3] = b1,k+1
    # C is 6x2, so c_ki where k is col, i is row
    # In W, index k (0..11) corresponds to C[k//2][k%2]
    # Wait, W[k*n+i, r] according to the contract.
    # "output factor follows the official trace convention W[k*n+i,r]"
    # So W index is k*n + i.
    # For 2x2 block (columns k, k+1):
    # W[k*2+0] = c_k0, W[k*2+1] = c_k1
    # W[k*2+2] = c_{k+1,0}, W[k*2+3] = c_{k+1,1}

    # Strassen terms (U coeffs, V coeffs, W coeffs)
    # U coeffs are on indices 0,1,2,3 (a00, a01, a10, a11)
    # V coeffs are on local indices 0,1,2,3 (b00, b01, b10, b11) relative to the block
    # W coeffs are on local indices (i, k_local) where i is 0,1 and k_local is 0,1
    # The output C entries for the block are:
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6

    strassen_blocks = [
        # term 1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {(0, 0): 1, (1, 1): 1}),
        # term 2
        ({2: 1, 3: 1}, {0: 1}, {(1, 0): 1, (1, 1): -1}),
        # term 3
        ({0: 1}, {1: 1, 3: -1}, {(0, 1): 1, (1, 1): 1}),
        # term 4
        ({3: 1}, {2: 1, 0: -1}, {(0, 0): 1, (1, 0): 1}),
        # term 5
        ({0: 1, 1: 1}, {3: 1}, {(0, 0): -1, (0, 1): 1}),
        # term 6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {(1, 1): 1}),
        # term 7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {(0, 0): 1})
    ]
    
    r = 0
    for b in range(3):
        for t in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen_blocks[t]
            for idx, val in u_coeffs.items():
                U[idx, r] = val
            for l, val in v_coeffs.items():
                # l is local index 0,1,2,3
                # 0: b0,k -> row 0, col 2b+0
                # 1: b0,k+1 -> row 0, col 2b+1
                # 2: b1,k -> row 1, col 2b+0
                # 3: b1,k+1 -> row 1, col 2b+1
                if l == 0:
                    v_idx = 0 * p + (2 * b)
                elif l == 1:
                    v_idx = 0 * p + (2 * b + 1)
                elif l == 2:
                    v_idx = 1 * p + (2 * b)
                elif l == 3:
                    v_idx = 1 * p + (2 * b + 1)
                V[v_idx, r] = val
            for (i, k_local), val in w_coeffs.items():
                # i is row of C (0,1), k_local is col offset (0,1)
                # k_global = 2*b + k_local
                # W index = k_global * n + i
                k_global = 2 * b + k_local
                w_idx = k_global * n + i
                W[w_idx, r] = val
            r += 1
    
    return U, V, W

def _verify_decomposition(U, V, W, tol=1e-10):
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    nm, mp, pn = n * m, m * p, p * n
    
    max_err = 0.0
    for i in range(nm):
        for j in range(mp):
            for k in range(pn):
                target = 1.0
                recon = 0.0
                for r in range(rank):
                    recon += U[i, r] * V[j, r] * W[k, r]
                err = abs(recon - target)
                if err > max_err:
                    max_err = err
    return max_err < tol

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    U, V, W = _build_rank21_strassen()
    rank = U.shape[1]
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
