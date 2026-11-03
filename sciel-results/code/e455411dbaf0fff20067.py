"""Matrix multiplication 4x4x6 via mixed Strassen decomposition."""
DIMENSIONS = [4, 4, 6]

import time
import numpy as np

def _get_strassen_2x2_factors():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indices: U[i*2+j], V[j*2+k], W[k*2+i].
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    prods = [
        ([(0,0,1.0), (1,1,1.0)], [(0,0,1.0), (1,1,1.0)], [(0,0,1.0), (1,1,1.0)]),
        ([(1,0,1.0), (1,1,1.0)], [(0,0,1.0)], [(1,0,1.0), (1,1,-1.0)]),
        ([(0,0,1.0)], [(0,1,1.0), (1,1,-1.0)], [(0,1,1.0), (1,1,1.0)]),
        ([(1,1,1.0)], [(1,0,1.0), (0,0,-1.0)], [(0,0,1.0), (1,0,1.0)]),
        ([(0,0,1.0), (0,1,1.0)], [(1,1,1.0)], [(0,0,-1.0), (0,1,1.0)]),
        ([(1,0,1.0), (0,0,-1.0)], [(0,0,1.0), (0,1,1.0)], [(1,1,1.0)]),
        ([(0,1,1.0), (1,1,-1.0)], [(1,0,1.0), (1,1,1.0)], [(0,0,1.0)]),
    ]

    for m_idx, (Ub, Vb, Wb) in enumerate(prods):
        for (i, j, c) in Ub:
            U[i * m + j, m_idx] += c
        for (j, k, c) in Vb:
            V[j * p + k, m_idx] += c
        for (i, k, c) in Wb:
            W[k * n + i, m_idx] += c

    return U, V, W

def _verify(U, V, W, n, m, p, rank):
    """Check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all (i,j,k)."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    err = np.max(np.abs(T - 1.0))
    return err

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    t0 = time.time()
    deadline = t0 + 110.0

    U_s, V_s, W_s = _get_strassen_2x2_factors()
    
    rank_444 = 49
    rank_442 = 28
    total_rank = rank_444 + rank_442
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # 1. 4x4x4 Strassen (columns 0-48)
    # We can vectorize the construction of U, V, W for the 4x4x4 part.
    # U_444[i*4+j, 7t+s] = U_s[2I+J, t] * U_s[2i0+j0, s]
    
    # Create indices for I, J, i0, j0
    # i = 2I + i0, j = 2J + j0
    # I, J in 0..1, i0, j0 in 0..1
    
    # We can just loop over the 4x4 blocks of indices, it's small.
    # But we can use broadcasting.
    
    # Let's stick to loops for clarity and safety, but they are fast enough.
    # Actually, let's try to be slightly more efficient by pre-computing the outer products.
    
    # U_s is 4x7.
    # U_444 row index is i*4+j.
    # Col index is 7t+s.
    
    # For each i, j in 0..3:
    # I = i//2, i0 = i%2
    # J = j//2, j0 = j%2
    # U[i*4+j, :] = sum_t U_s[2I+J, t] * U_s[2i0+j0, :] (outer product summed over t? No)
    # Wait. The formula is:
    # U_444[i*4+j, 7t+s] = U_s[2I+J, t] * U_s[2i0+j0, s]
    # This is an element-wise product of two vectors?
    # Let u1 = U_s[2I+J, :] (length 7)
    # Let u2 = U_s[2i0+j0, :] (length 7)
    # Then U_444[i*4+j, 7t+s] = u1[t] * u2[s]
    # This is the outer product of u1 and u2, reshaped into a vector of length 49.
    # Specifically, if we flatten the 7x7 matrix (u1[:, None] * u2[None, :]) in row-major order (t, s),
    # then index 7t+s corresponds to element [t, s].
    
    for i in range(4):
        I = i // 2
        i0 = i % 2
        for j in range(4):
            J = j // 2
            j0 = j % 2
            u1 = U_s[2 * I + J, :]
            u2 = U_s[2 * i0 + j0, :]
            # Outer product
            outer = np.outer(u1, u2)
            U[i * 4 + j, :49] = outer.flatten()
            
    # V_444[j*6+k, 7t+s] = V_s[2J+K, t] * V_s[2j0+k0, s]
    # Note: k is in 0..3 for the 4x4x4 part.
    for j in range(4):
        J = j // 2
        j0 = j % 2
        for k in range(4):
            K = k // 2
            k0 = k % 2
            v1 = V_s[2 * J + K, :]
            v2 = V_s[2 * j0 + k0, :]
            outer = np.outer(v1, v2)
            V[j * 6 + k, :49] = outer.flatten()
            
    # W_444[k*4+i, 7t+s] = W_s[2K+I, t] * W_s[2k0+i0, s]
    for k in range(4):
        K = k // 2
        k0 = k % 2
        for i in range(4):
            I = i // 2
            i0 = i % 2
            w1 = W_s[2 * K + I, :]
            w2 = W_s[2 * k0 + i0, :]
            outer = np.outer(w1, w2)
            W[k * 4 + i, :49] = outer.flatten()

    # 2. 4x4x2 Strassen-block (columns 49-76)
    products = [
        (0, 0, 0), # I, J, is_bot
        (0, 1, 1), # I, J, is_bot
        (1, 0, 0), # I, J, is_bot
        (1, 1, 1), # I, J, is_bot
    ]
    
    for p_idx, (I, J, is_bot) in enumerate(products):
        col_offset = rank_444 + p_idx * 7
        
        # A_IJ
        for i0 in range(2):
            for j0 in range(2):
                i = 2 * I + i0
                j = 2 * J + j0
                U[i * 4 + j, col_offset:col_offset+7] += U_s[2 * i0 + j0, :]
                
        # B part
        for j0 in range(2):
            for k0 in range(2):
                j_global = 2 * is_bot + j0
                k_global = 4 + k0
                V[j_global * 6 + k_global, col_offset:col_offset+7] += V_s[2 * j0 + k0, :]
                
        # C part
        for i0 in range(2):
            for k0 in range(2):
                i_global = 2 * I + i0
                k_global = 4 + k0
                W[k_global * 4 + i_global, col_offset:col_offset+7] += W_s[2 * k0 + i0, :]

    err = _verify(U, V, W, n, m, p, total_rank)
    
    if err > 1e-6:
        # Fallback
        rank = n * m * p
        U_f = np.zeros((n * m, rank), dtype=float)
        V_f = np.zeros((m * p, rank), dtype=float)
        W_f = np.zeros((p * n, rank), dtype=float)
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j, column] = 1
                    V_f[j * p + k, column] = 1
                    W_f[k * n + i, column] = 1
                    column += 1
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
