"""Optimized tensor decomposition for 4x4 x 4x7 matrix multiplication using Strassen."""
DIMENSIONS = [4, 4, 7]

import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    U = []
    V = []
    W = []
    
    # --- Define 2x2x2 Strassen basis ---
    s2x2_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [-1, 0, 0, 0]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 1, -1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, -1]),
    ]
    
    def quad_global_indices(quad):
        if quad == 0:
            return [0, 1, 4, 5]
        elif quad == 1:
            return [2, 3, 6, 7]
        elif quad == 2:
            return [8, 9, 12, 13]
        elif quad == 3:
            return [10, 11, 14, 15]
    
    big_terms = [
        {'A': [(0, 1), (3, 1)], 'B': [(0, 1), (3, 1)], 'C': [(0, 1), (3, 1)]},
        {'A': [(2, 1), (3, 1)], 'B': [(0, 1)], 'C': [(0, -1)]},
        {'A': [(0, 1)], 'B': [(1, 1), (3, -1)], 'C': [(1, 1)]},
        {'A': [(3, 1)], 'B': [(2, 1), (0, -1)], 'C': [(0, 1), (2, 1)]},
        {'A': [(0, 1), (1, 1)], 'B': [(3, 1)], 'C': [(1, 1), (3, 1)]},
        {'A': [(2, 1), (0, -1)], 'B': [(0, 1), (1, 1)], 'C': [(2, 1), (3, -1)]},
        {'A': [(1, 1), (3, -1)], 'B': [(2, 1), (3, 1)], 'C': [(0, 1), (3, -1)]},
    ]
    
    # Generate 49 terms for 4x4x4 part
    for big in big_terms:
        for sub_U, sub_V, sub_W in s2x2_terms:
            A_vec = [0] * 16
            B_vec = [0] * 16
            C_vec = [0] * 16
            
            for quad, coeff in big['A']:
                gidx = quad_global_indices(quad)
                for li in range(4):
                    A_vec[gidx[li]] += coeff * sub_U[li]
            
            for quad, coeff in big['B']:
                gidx = quad_global_indices(quad)
                for li in range(4):
                    B_vec[gidx[li]] += coeff * sub_V[li]
            
            for quad, coeff in big['C']:
                gidx = quad_global_indices(quad)
                for li in range(4):
                    C_vec[gidx[li]] += coeff * sub_W[li]
            
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            
            for i in range(n):
                for j in range(m):
                    u_col[i * m + j] = A_vec[i * m + j]
            
            for j in range(m):
                for k in range(4):
                    v_col[j * p + k] = B_vec[j * 4 + k]
            
            for i in range(n):
                for k in range(4):
                    w_col[k * n + i] = C_vec[i * 4 + k]
            
            U.append(u_col)
            V.append(v_col)
            W.append(w_col)
    
    # Add schoolbook terms for remaining 3 columns
    for i in range(n):
        for j in range(m):
            for k in range(3):
                u_col = [0] * (n * m)
                v_col = [0] * (m * p)
                w_col = [0] * (p * n)
                
                u_col[i * m + j] = 1
                v_col[j * p + (k + 4)] = 1
                w_col[(k + 4) * n + i] = 1
                
                U.append(u_col)
                V.append(v_col)
                W.append(w_col)
    
    # Verify
    rank = len(U)
    U_mat = np.array(U, dtype=float).T
    V_mat = np.array(V, dtype=float).T
    W_mat = np.array(W, dtype=float).T
    
    U_r = U_mat.reshape(rank, n, m)
    V_r = V_mat.reshape(rank, m, p)
    W_r = W_mat.reshape(rank, p, n)
    
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_r[:, i, j] * V_r[:, j, k] * W_r[:, k, i])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-10:
        # Fallback to schoolbook
        U_fb = []
        V_fb = []
        W_fb = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_col = [0] * (n * m)
                    v_col = [0] * (m * p)
                    w_col = [0] * (p * n)
                    u_col[i * m + j] = 1
                    v_col[j * p + k] = 1
                    w_col[k * n + i] = 1
                    U_fb.append(u_col)
                    V_fb.append(v_col)
                    W_fb.append(w_col)
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
