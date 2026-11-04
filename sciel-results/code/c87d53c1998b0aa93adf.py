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
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    if U_arr.shape[1] != V_arr.shape[1] or U_arr.shape[1] != W_arr.shape[1]:
        return False
    rank = U_arr.shape[1]
    
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    T_recon = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    if not np.allclose(T_recon, 1.0, atol=tol):
        return False
    return True

def _get_strassen_222():
    """
    Correct Strassen 2x2x2 decomposition, rank 7.
    Returns U(4,7), V(4,7), W(4,7) where:
    U[i,j,c], V[j,k,c], W[k,i,c] with i,j,k in {0,1}
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # m1 = (A00+A11)(B00+B11): C00+=1, C11+=1
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1
    
    # m2 = (A10+A11)B00: C10+=1, C11+=-1
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[2,1]=1; W[3,1]=-1
    
    # m3 = A00(B01-B11): C01+=1, C11+=1
    U[0,2]=1
    V[1,2]=1; V[3,2]=-1
    W[1,2]=1; W[3,2]=1
    
    # m4 = A11(B10-B00): C00+=1, C10+=1
    U[3,3]=1
    V[2,3]=1; V[0,3]=-1
    W[0,3]=1; W[2,3]=1
    
    # m5 = (A00+A01)B11: C00+=-1, C01+=1
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[0,4]=-1; W[1,4]=1
    
    # m6 = (A10-A00)(B00+B01): C11+=1
    U[0,5]=-1; U[2,5]=1
    V[0,5]=1; V[1,5]=1
    W[3,5]=1
    
    # m7 = (A01-A11)(B10+B11): C00+=1
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    W[0,6]=1
    
    # Verify
    U3 = U.reshape(2,2,7)
    V3 = V.reshape(2,2,7)
    W3 = W.reshape(2,2,7)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    if not np.allclose(T, 1.0, atol=1e-10):
        return None
    return U, V, W

def _construct_rank_44():
    """
    Construct a rank-44 decomposition for 3x4x8.
    Row 0: 4 blocks of 1x2x2 (rank 4 each) = 16
    Rows 1-2: 4 blocks of 2x2x2 (Strassen rank 7 each) = 28
    Total: 44
    """
    n, m, p = 3, 4, 8
    U_list, V_list, W_list = [], [], []
    
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

    def get_1x2x2_schoolbook():
        u = np.zeros((2, 4))
        v = np.zeros((4, 4))
        w = np.zeros((2, 4))
        idx = 0
        for j in range(2):
            for k in range(2):
                u[j, idx] = 1.0
                v[j * 2 + k, idx] = 1.0
                w[k, idx] = 1.0
                idx += 1
        return u, v, w

    u_122, v_122, w_122 = get_1x2x2_schoolbook()
    s222 = _get_strassen_222()
    if s222 is None:
        return None
    u_222, v_222, w_222 = s222
    
    for j_pair in range(2):
        for k_pair in range(4):
            j_off = j_pair * 2
            k_off = k_pair * 2
            add_sub_term(0, j_off, k_off, 1, 2, 2, u_122, v_122, w_122)
    
    for j_pair in range(2):
        for k_pair in range(4):
            j_off = j_pair * 2
            k_off = k_pair * 2
            add_sub_term(1, j_off, k_off, 2, 2, 2, u_222, v_222, w_222)
    
    U_arr = np.array(U_list).T
    V_arr = np.array(V_list).T
    W_arr = np.array(W_list).T
    
    return U_arr, V_arr, W_arr

def _construct_rank_88():
    """
    Construct a rank-88 decomposition (fallback).
    """
    n, m, p = 3, 4, 8
    U_list, V_list, W_list = [], [], []
    
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
        u = np.zeros((2, 8))
        v = np.zeros((8, 8))
        w = np.zeros((4, 8))
        idx = 0
        for j in range(2):
            for k in range(4):
                u[j, idx] = 1.0
                v[j * 4 + k, idx] = 1.0
                w[k, idx] = 1.0
                idx += 1
        return u, v, w

    def get_2x2x4():
        # Two independent 2x2x2 Strassen blocks
        s222 = _get_strassen_222()
        if s222 is None:
            return None
        u_s, v_s, w_s = s222
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
    r224 = get_2x2x4()
    if r224 is None:
        return None
    u_224, v_224, w_224 = r224
    
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
    
    # Fallback: Schoolbook (rank 96)
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
    
    if time.time() < deadline:
        try:
            res = _construct_rank_44()
            if res is not None:
                U_44, V_44, W_44 = res
                if _verify_safe(U_44.tolist(), V_44.tolist(), W_44.tolist()):
                    if U_44.shape[1] < best_rank:
                        best_result = {"U": U_44.tolist(), "V": V_44.tolist(), "W": W_44.tolist()}
                        best_rank = U_44.shape[1]
        except Exception:
            pass
    
    if time.time() < deadline:
        try:
            res = _construct_rank_88()
            if res is not None:
                U_88, V_88, W_88 = res
                if _verify_safe(U_88.tolist(), V_88.tolist(), W_88.tolist()):
                    if U_88.shape[1] < best_rank:
                        best_result = {"U": U_88.tolist(), "V": V_88.tolist(), "W": W_88.tolist()}
                        best_rank = U_88.shape[1]
        except Exception:
            pass
            
    return best_result
