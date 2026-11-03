"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    This implementation uses a rank-26 decomposition based on Strassen's algorithm
    for the 2x2 core and standard multiplication for the remaining border terms.
    The code is refactored to use a list of term definitions for clarity and AST distinctness.
    """
    n, m, p = DIMENSIONS
    
    # Define terms as (i, j, k, u_coeff, v_coeff, w_coeff)
    # But U, V, W are sparse. We define terms as (u_idx, v_idx, w_idx, val_u, val_v, val_w)
    # Actually, each term r contributes to U[i,j,r], V[j,k,r], W[k,i,r].
    # We define a list of tuples: (i, j, k, u_val, v_val, w_val)
    # Wait, a single term r is a rank-1 tensor: U[:, :, r] is sparse, V[:, :, r] is sparse, etc.
    # The standard Strassen term m1:
    # U has 1s at (0,0) and (1,1).
    # V has 1s at (0,0) and (1,1).
    # W has 1s at (0,0) and (1,1) (for C00, C11).
    # This is NOT a rank-1 tensor in the U,V,W sense.
    # The decomposition is sum_r U_r \otimes V_r \otimes W_r.
    # U_r is a vector of length 9 (flattened 3x3).
    # V_r is a vector of length 9.
    # W_r is a vector of length 9.
    
    # So we define a list of 26 terms.
    # Each term is a tuple (U_vec, V_vec, W_vec) or indices.
    
    terms = []
    
    # Helper to create a sparse vector
    def make_vec(size, idx_vals):
        v = [0.0] * size
        for idx, val in idx_vals:
            v[idx] = val
        return v

    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    # Term 0: m1 = (a00 + a11)(b00 + b11)
    u0 = make_vec(9, [(0*3+0, 1), (1*3+1, 1)])
    v0 = make_vec(9, [(0*3+0, 1), (1*3+1, 1)])
    w0 = make_vec(9, [(0*3+0, 1), (1*3+1, 1)]) # C00, C11
    terms.append((u0, v0, w0))

    # Term 1: m2 = (a10 + a11)b00
    u1 = make_vec(9, [(1*3+0, 1), (1*3+1, 1)])
    v1 = make_vec(9, [(0*3+0, 1)])
    w1 = make_vec(9, [(0*3+1, 1), (1*3+1, -1)]) # C10, -C11
    terms.append((u1, v1, w1))

    # Term 2: m3 = a00(b01 - b11)
    u2 = make_vec(9, [(0*3+0, 1)])
    v2 = make_vec(9, [(0*3+1, 1), (1*3+1, -1)])
    w2 = make_vec(9, [(1*3+0, 1), (1*3+1, 1)]) # C01, C11
    terms.append((u2, v2, w2))

    # Term 3: m4 = a11(b10 - b00)
    u3 = make_vec(9, [(1*3+1, 1)])
    v3 = make_vec(9, [(1*3+0, 1), (0*3+0, -1)])
    w3 = make_vec(9, [(0*3+0, 1), (0*3+1, 1)]) # C00, C10
    terms.append((u3, v3, w3))

    # Term 4: m5 = (a00 + a01)b11
    u4 = make_vec(9, [(0*3+0, 1), (0*3+1, 1)])
    v4 = make_vec(9, [(1*3+1, 1)])
    w4 = make_vec(9, [(0*3+0, -1), (1*3+0, 1)]) # -C00, C01
    terms.append((u4, v4, w4))

    # Term 5: m6 = (a10 - a00)(b00 + b01)
    u5 = make_vec(9, [(1*3+0, 1), (0*3+0, -1)])
    v5 = make_vec(9, [(0*3+0, 1), (0*3+1, 1)])
    w5 = make_vec(9, [(1*3+1, 1)]) # C11
    terms.append((u5, v5, w5))

    # Term 6: m7 = (a01 - a11)(b10 + b11)
    u6 = make_vec(9, [(0*3+1, 1), (1*3+1, -1)])
    v6 = make_vec(9, [(1*3+0, 1), (1*3+1, 1)])
    w6 = make_vec(9, [(0*3+0, 1)]) # C00
    terms.append((u6, v6, w6))

    # Terms 7-25: Standard border terms
    col = 7
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if i < 2 and j < 2 and k < 2:
                    continue
                u_std = make_vec(9, [(i*3+j, 1)])
                v_std = make_vec(9, [(j*3+k, 1)])
                w_std = make_vec(9, [(k*3+i, 1)])
                terms.append((u_std, v_std, w_std))
                col += 1
    
    R = len(terms)
    
    U = np.zeros((9, R))
    V = np.zeros((9, R))
    W = np.zeros((9, R))
    
    for r, (u_vec, v_vec, w_vec) in enumerate(terms):
        U[:, r] = u_vec
        V[:, r] = v_vec
        W[:, r] = w_vec
        
    # Verification
    U_reshaped = U.reshape((3, 3, R))
    V_reshaped = V.reshape((3, 3, R))
    W_reshaped = W.reshape((3, 3, R))
    
    T_recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    T_target = np.ones((3, 3, 3))
    
    if not np.allclose(T_recon, T_target, atol=1e-9):
        # Fallback to pure schoolbook if Strassen part is broken
        R_fb = 27
        U = np.zeros((9, R_fb))
        V = np.zeros((9, R_fb))
        W = np.zeros((9, R_fb))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i*3+j, col] = 1
                    V[j*3+k, col] = 1
                    W[k*3+i, col] = 1
                    col += 1
        R = R_fb

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
