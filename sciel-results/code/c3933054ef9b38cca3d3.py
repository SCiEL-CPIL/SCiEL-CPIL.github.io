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
    
    # Reshape to 3D
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    T_recon = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    if not np.allclose(T_recon, 1.0, atol=tol):
        return False
    return True

def _construct_rank_64():
    """
    Construct a rank-64 decomposition for 3x4x8.
    Split A (3x4) into A1 (1x4) and A2 (2x4).
    A1 * B: schoolbook 1x4x8, rank 32
    A2 * B: 2x4x8 using four 2x4x2 blocks of rank 8 each, total 32
    Grand total: 32 + 32 = 64
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

    def get_2x4x2_rank8():
        """
        Rank-8 decomposition of 2x4x2 (multiply 2x4 by 4x2).
        A is 2x4, B is 4x2, C is 2x2.
        
        C[0,0] = a*B00 + b*B10 + c*B20 + d*B30
        C[0,1] = a*B01 + b*B11 + c*B21 + d*B31
        C[1,0] = e*B00 + f*B10 + g*B20 + h*B30
        C[1,1] = e*B01 + f*B11 + g*B21 + h*B31
        
        Using Winograd F(2) on pairs of j:
        For j-pair (0,1):
          m0 = (a+e)*(B00+B01) -- no, let me use a direct rank-8 construction.
          
        Direct rank-8: use the identity that we can write C as sum of 8 rank-1 tensors.
        Actually, the simplest correct rank-8 is just schoolbook with 8 terms:
        We need 8 bilinear forms. Since 2*4*2=16 > 8, we need structure.
        
        Key insight: C[i,k] = sum_j A[i,j]*B[j,k].
        For rank 8, we use:
        m_j = (A[0,j] + A[1,j]) * B[j,0] for j=0..3  (4 mults)
        n_j = (A[0,j] + A[1,j]) * B[j,1] for j=0..3  (4 mults)
        o_j = (A[0,j] - A[1,j]) * B[j,0] for j=0..3  (4 mults)
        p_j = (A[0,j] - A[1,j]) * B[j,1] for j=0..3  (4 mults)
        That's 16, not 8.
        
        Let me just use schoolbook rank 16 for 2x4x2 and accept rank 96 total... no.
        
        Actually, I realize the tensor rank of 2x4x2 multiplication is exactly 16
        (it's a direct sum of 4 independent 2x1x2 problems, each rank 4... no).
        
        Let me just verify: can I do 2x4x2 in rank 8?
        Lower bound: max(2*2, 2*4, 4*2) = 8. So rank >= 8.
        
        Construction for rank 8:
        Think of it as: C = A * B where A is 2x4, B is 4x2.
        Reshape A to 2x2x2 (split columns into pairs) and B to 2x2x2 (split rows into pairs).
        C[i,k] = sum_{j=0}^{1} A[i, 2j, :] . B[2j:2j+2, k]
               = sum_{j=0}^{1} sum_{l=0}^{1} A[i, 2j+l] * B[2j+l, k]
        
        This is a 2x2x2 tensor (over the j index) of 2x2x2 matrix products.
        Each 2x2x2 uses Strassen rank 7, total 14. Worse than 16.
        
        Alternatively, just use 8 terms directly:
        The 2x4x2 tensor T[i,j,k] = delta (1 if valid index).
        It can be written as sum of 16 rank-1 tensors (schoolbook).
        Can we merge any? The 16 tensors are:
        T_c = e_i * e_j * e_k for (i,j,k) in {(0,0,0),(0,0,1),(0,1,0),(0,1,1),(0,2,0),(0,2,1),(0,3,0),(0,3,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1),(1,2,0),(1,2,1),(1,3,0),(1,3,1)}
        
        These are all linearly independent as vectors in R^{2*4*2} = R^{16}, so the rank is exactly 16.
        Wait, that means rank(2x4x2) = 16, not 8!
        
        So my rank-64 plan is wrong. 2x4x2 has rank 16, so 2x4x8 = 4*16 = 64, plus 32 = 96 total.
        That's just schoolbook!
        
        I need a different approach. Let me use Strassen on 2x2x2 blocks.
        2x4x8: split into 4 blocks of 2x4x2. Each 2x4x2: split A into two 2x2, B into two 2x2 (vertically).
        C = A_top * B_top + A_bot * B_bot, each 2x2x2.
        Use Strassen for each: 7+7 = 14 per 2x4x2 block.
        4 blocks * 14 = 56 for 2x4x8.
        Plus 32 for 1x4x8 = 88 total.
        
        That's the rank-88 we already have! Let me just return None and keep rank-88.
        """
        # The 2x4x2 tensor has rank 16 (all 16 basis tensors are linearly independent).
        # So we cannot do better than schoolbook for this sub-problem.
        # Return None to signal we can't construct a better solution here.
        return None, None, None

    u_248, v_248, w_248 = get_2x4x2_rank8()
    if u_248 is None:
        return None, None, None
    
    u_148, v_148, w_148 = get_1x4x8_schoolbook()
    
    add_sub_term(0, 0, 0, 1, 4, 8, u_148, v_148, w_148)
    
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
    
    # Try Rank 64 (Winograd-based)
    if time.time() < deadline:
        try:
            U_64, V_64, W_64 = _construct_rank_64()
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
