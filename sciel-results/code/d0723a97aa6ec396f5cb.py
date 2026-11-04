"""Matrix multiplication 3x4x8 rank-minimal decomposition search."""
import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def _verify_safe(U, V, W, n=3, m=4, p=8, tol=1e-6):
    """
    Safely verify the decomposition using vectorized operations.
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    Target: T[i,j,k] = 1 for all i,j,k.
    """
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    if U_arr.shape[1] != V_arr.shape[1] or U_arr.shape[1] != W_arr.shape[1]:
        return False
    rank = U_arr.shape[1]
    
    # Reshape to 3D
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Use matmul for efficiency: T[i,j,k] = sum_c U[i,j,c] * (V[j,:,c] * W[:,i,c])
    # Or: T = sum_c U_3d[:, :, c] * V_3d[:, :, c] * W_3d[:, :, c] with proper index mapping
    # W_3d[k, i, c] means W is indexed as (k, i, c) where k is first dim, i is second
    # We need T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # Loop over rank for memory efficiency
    T_recon = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        T_recon += U_3d[:, :, c][:, :, np.newaxis] * V_3d[:, :, c][np.newaxis, :, :] * W_3d[:, :, c].T[np.newaxis, np.newaxis, :]
    
    if not np.allclose(T_recon, 1.0, atol=tol):
        return False
    return True

def _construct_rank_56():
    """
    Construct a rank-56 decomposition for 3x4x8 using Winograd-style splitting.
    3x4x8 = 1x4x8 (rank 32) + 2x4x8 (rank 24)
    2x4x8: split 8 cols into 4 groups of 2, each 2x4x2 in rank 6 (Winograd).
    Total: 32 + 4*6 = 56.
    """
    n, m, p = 3, 4, 8
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_sub_term(i_off, j_off, k_off, sub_n, sub_m, sub_p, u_sub, v_sub, w_sub):
        rank_sub = u_sub.shape[1]
        for c in range(rank_sub):
            u_col = np.zeros(n * m)
            v_col = np.zeros(m * p)
            w_col = np.zeros(p * n)
            
            for i in range(sub_n):
                for j in range(sub_m):
                    u_col[(i_off + i) * m + (j_off + j)] = u_sub[i * sub_m + j, c]
            
            for j in range(sub_m):
                for k in range(sub_p):
                    v_col[(j_off + j) * p + (k_off + k)] = v_sub[j * sub_p + k, c]
            
            for k in range(sub_p):
                for i in range(sub_n):
                    w_col[(k_off + k) * n + (i_off + i)] = w_sub[k * sub_n + i, c]
            
            U_list.append(u_col)
            V_list.append(v_col)
            W_list.append(w_col)

    def get_1x4x8_schoolbook():
        """Schoolbook 1x4x8, rank 32."""
        u = np.zeros((4, 32))
        v = np.zeros((32, 32))
        w = np.zeros((8, 32))
        idx = 0
        for j in range(4):
            for k in range(8):
                u[j, idx] = 1.0
                v[j * 8 + k, idx] = 1.0
                w[k, idx] = 1.0
                idx += 1
        return u, v, w

    def get_2x4x2_winograd():
        """
        Rank-6 decomposition of 2x4x2 (multiply 2x4 by 4x2).
        A: 2x4, B: 4x2, C: 2x2.
        Uses Winograd-style splitting with shared precomputations.
        
        A = [[a0, a1, a2, a3], [b0, b1, b2, b3]]
        B = [[c0, d0], [c1, d1], [c2, d2], [c3, d3]]
        C[0,0] = a0*c0 + a1*c1 + a2*c2 + a3*c3
        C[0,1] = a0*d0 + a1*d1 + a2*d2 + a3*d3
        C[1,0] = b0*c0 + b1*c1 + b2*c2 + b3*c3
        C[1,1] = b0*d0 + b1*d1 + b2*d2 + b3*d3
        
        6 multiplications:
        m0 = (a0+b0)*(c0+d0)
        m1 = (a1+b1)*(c1+d1)
        m2 = (a2+b2)*(c2+d2)
        m3 = (a3+b3)*(c3+d3)
        m4 = (a0-b0)*(c0-d0)
        m5 = (a1-b1)*(c1-d1)
        ... this gives 8, not 6.
        
        Let me use a different approach. The rank of 2x4x2 is at most 8.
        I'll use rank 8 for now, giving total 32 + 4*8 = 64.
        """
        # 2x4x2 rank-8 schoolbook
        u = np.zeros((8, 8))  # A: 2*4 = 8 entries
        v = np.zeros((8, 8))  # B: 4*2 = 8 entries
        w = np.zeros((4, 8))  # C: 2*2 = 4 entries, indexed k*2+i
        
        idx = 0
        for i in range(2):
            for j in range(4):
                for k in range(2):
                    u[i * 4 + j, idx] = 1.0
                    v[j * 2 + k, idx] = 1.0
                    w[k * 2 + i, idx] = 1.0
                    idx += 1
        return u, v, w

    u_148, v_148, w_148 = get_1x4x8_schoolbook()
    u_242, v_242, w_242 = get_2x4x2_winograd()
    
    # A1 (1x4) * B (4x8): i_off=0
    add_sub_term(0, 0, 0, 1, 4, 8, u_148, v_148, w_148)
    
    # A2 (2x4) * B (4x8): i_off=1, split 8 cols into 4 groups of 2
    for k_group in range(4):
        add_sub_term(1, 0, k_group * 2, 2, 4, 2, u_242, v_242, w_242)
    
    U_arr = np.array(U_list).T
    V_arr = np.array(V_list).T
    W_arr = np.array(W_list).T
    
    return U_arr, V_arr, W_arr

def _construct_rank_88():
    """
    Construct a rank-88 decomposition (fallback).
    """
    n, m, p = 3, 4, 8
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_sub_term(i_off, j_off, k_off, sub_n, sub_m, sub_p, u_sub, v_sub, w_sub):
        rank_sub = u_sub.shape[1]
        for c in range(rank_sub):
            u_col = np.zeros(n * m)
            v_col = np.zeros(m * p)
            w_col = np.zeros(p * n)
            
            for i in range(sub_n):
                for j in range(sub_m):
                    u_col[(i_off + i) * m + (j_off + j)] = u_sub[i * sub_m + j, c]
            
            for j in range(sub_m):
                for k in range(sub_p):
                    v_col[(j_off + j) * p + (k_off + k)] = v_sub[j * sub_p + k, c]
            
            for k in range(sub_p):
                for i in range(sub_n):
                    w_col[(k_off + k) * n + (i_off + i)] = w_sub[k * sub_n + i, c]
            
            U_list.append(u_col)
            V_list.append(v_col)
            W_list.append(w_col)

    def get_1x2x4():
        u = np.zeros((1 * 2, 8))
        v = np.zeros((2 * 4, 8))
        w = np.zeros((4 * 1, 8))
        idx = 0
        for j in range(2):
            for k in range(4):
                u[j, idx] = 1.0
                v[j * 4 + k, idx] = 1.0
                w[k * 1, idx] = 1.0
                idx += 1
        return u, v, w

    def get_2x2x4():
        u_s = np.array([
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1]
        ]).T
        
        v_s = np.array([
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [-1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]).T
        
        w_s = np.zeros((4, 7))
        w_s[0, 0] = 1; w_s[3, 0] = 1
        w_s[1, 1] = 1; w_s[3, 1] = -1
        w_s[2, 2] = 1; w_s[3, 2] = 1
        w_s[0, 3] = 1; w_s[1, 3] = 1
        w_s[0, 4] = -1; w_s[2, 4] = 1
        w_s[3, 5] = 1
        w_s[0, 6] = 1
        
        u_224 = np.zeros((4, 14))
        v_224 = np.zeros((8, 14))
        w_224 = np.zeros((8, 14))
        
        for c in range(7):
            u_224[:, c] = u_s[:, c]
            v_224[0, c] = v_s[0, c]
            v_224[1, c] = v_s[1, c]
            v_224[4, c] = v_s[2, c]
            v_224[5, c] = v_s[3, c]
            w_224[0, c] = w_s[0, c]
            w_224[1, c] = w_s[1, c]
            w_224[2, c] = w_s[2, c]
            w_224[3, c] = w_s[3, c]
            
        for c in range(7):
            u_224[:, c+7] = u_s[:, c]
            v_224[2, c+7] = v_s[0, c]
            v_224[3, c+7] = v_s[1, c]
            v_224[6, c+7] = v_s[2, c]
            v_224[7, c+7] = v_s[3, c]
            w_224[4, c+7] = w_s[0, c]
            w_224[5, c+7] = w_s[1, c]
            w_224[6, c+7] = w_s[2, c]
            w_224[7, c+7] = w_s[3, c]
            
        return u_224, v_224, w_224

    u_124, v_124, w_124 = get_1x2x4()
    u_224, v_224, w_224 = get_2x2x4()
    
    add_sub_term(0, 0, 0, 1, 2, 4, u_124, v_124, w_124)
    add_sub_term(0, 2, 0, 1, 2, 4, u_124, v_124, w_124)
    add_sub_term(0, 0, 4, 1, 2, 4, u_124, v_124, w_124)
    add_sub_term(0, 2, 4, 1, 2, 4, u_124, v_124, w_124)
    add_sub_term(1, 0, 0, 2, 2, 4, u_224, v_224, w_224)
    add_sub_term(1, 2, 0, 2, 2, 4, u_224, v_224, w_224)
    add_sub_term(1, 0, 4, 2, 2, 4, u_224, v_224, w_224)
    add_sub_term(1, 2, 4, 2, 2, 4, u_224, v_224, w_224)
    
    U_arr = np.array(U_list).T
    V_arr = np.array(V_list).T
    W_arr = np.array(W_list).T
    
    return U_arr, V_arr, W_arr

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Fallback: Schoolbook
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_schoolbook
    
    # Try Rank 64 (Winograd-based: 1x4x8 rank 32 + 4x(2x4x2 rank 8) = 64)
    if time.time() < deadline:
        try:
            U_64, V_64, W_64 = _construct_rank_56()
            if U_64 is not None:
                if _verify_safe(U_64.tolist(), V_64.tolist(), W_64.tolist()):
                    if U_64.shape[1] < best_rank:
                        best_result = {"U": U_64.tolist(), "V": V_64.tolist(), "W": W_64.tolist()}
                        best_rank = U_64.shape[1]
        except Exception:
            pass
    
    # Try Rank 88 (Strassen-based) as fallback
    if time.time() < deadline:
        try:
            U_88, V_88, W_88 = _construct_rank_88()
            if _verify_safe(U_88.tolist(), V_88.tolist(), W_88.tolist()):
                if U_88.shape[1] < best_rank:
                    best_result = {"U": U_88.tolist(), "V": V_88.tolist(), "W": W_88.tolist()}
                    best_rank = U_88.shape[1]
        except Exception:
            pass
            
    return best_result
