"""Matrix multiplication 3x4x8 rank-minimal decomposition search."""
import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def _verify_safe(U, V, W, n=3, m=4, p=8, tol=1e-6):
    """
    Verify the decomposition using vectorized operations.
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    Target: T[i,j,k] = 1 for all i,j,k.
    """
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    if U_arr.shape[1] != V_arr.shape[1] or U_arr.shape[1] != W_arr.shape[1]:
        return False
    rank = U_arr.shape[1]
    
    # Vectorized verification
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Use einsum for efficiency
    T_recon = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    if not np.allclose(T_recon, 1.0, atol=tol):
        return False
    return True

def _construct_rank_88():
    """
    Construct a rank-88 decomposition.
    Split A (3x4) into A1 (1x4) and A2 (2x4).
    Split B (4x8) into B1 (4x4) and B2 (4x4).
    
    A1 B1: 1x4x4 -> Split into 1x2x4 (rank 8) + 1x2x4 (rank 8) = 16
    A1 B2: 1x4x4 -> Split into 1x2x4 (rank 8) + 1x2x4 (rank 8) = 16
    A2 B1: 2x4x4 -> Split into 2x2x4 (rank 14) + 2x2x4 (rank 14) = 28
    A2 B2: 2x4x4 -> Split into 2x2x4 (rank 14) + 2x2x4 (rank 14) = 28
    Total: 16 + 16 + 28 + 28 = 88.
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
        # Rank 8 schoolbook
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
        # Rank 14 using two 2x2x2 Strassen decompositions
        # Strassen 2x2x2
        u_s = np.array([
            [1, 0, 1, 0, 1, 0, 0],
            [0, 1, 1, 1, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0], # Placeholder, will fix
        ])
        # Correct Strassen U (A parts)
        # A11, A12, A21, A22
        u_s = np.array([
            [1, 0, 1, 0, 1, 0, 0], # M1: A11+A22? No.
            # M1 = (A11+A22)(B11+B22) -> A: [1, 0, 0, 1]
            # M2 = (A21+A22)B11 -> A: [0, 0, 1, 1]
            # M3 = A11(B12-B22) -> A: [1, 0, 0, 0]
            # M4 = A22(B21-B11) -> A: [0, 0, 0, 1]
            # M5 = (A11+A12)B22 -> A: [1, 1, 0, 0]
            # M6 = (A21-A11)(B11+B12) -> A: [-1, 0, 1, 0]
            # M7 = (A12-A22)(B21+B22) -> A: [0, 1, 0, -1]
        ])
        u_s = np.array([
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1]
        ]).T
        
        # V (B parts)
        # B11, B12, B21, B22
        v_s = np.array([
            [1, 0, 0, 1], # M1: B11+B22
            [1, 0, 0, 0], # M2: B11
            [0, 1, 0, -1], # M3: B12-B22
            [-1, 0, 1, 0], # M4: B21-B11
            [0, 0, 0, 1], # M5: B22
            [1, 1, 0, 0], # M6: B11+B12
            [0, 0, 1, 1] # M7: B21+B22
        ]).T
        
        # W (C parts)
        # C11, C12, C21, C22
        # Indices in W: k*n+i. n=2.
        # C11 (i=0, k=0) -> 0
        # C21 (i=1, k=0) -> 1
        # C12 (i=0, k=1) -> 2
        # C22 (i=1, k=1) -> 3
        w_s = np.zeros((4, 7))
        # M1: C11 + C22
        w_s[0, 0] = 1; w_s[3, 0] = 1
        # M2: C21 - C22
        w_s[1, 1] = 1; w_s[3, 1] = -1
        # M3: C12 + C22
        w_s[2, 2] = 1; w_s[3, 2] = 1
        # M4: C11 + C21
        w_s[0, 3] = 1; w_s[1, 3] = 1
        # M5: -C11 + C12
        w_s[0, 4] = -1; w_s[2, 4] = 1
        # M6: C22
        w_s[3, 5] = 1
        # M7: C11
        w_s[0, 6] = 1
        
        # Construct 2x2x4
        u_224 = np.zeros((4, 14))
        v_224 = np.zeros((8, 14))
        w_224 = np.zeros((8, 14))
        
        # First half (k=0,1)
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
            
        # Second half (k=2,3)
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
    
    # A1 (1x4) x B1 (4x4)
    add_sub_term(0, 0, 0, 1, 2, 4, u_124, v_124, w_124)
    add_sub_term(0, 2, 0, 1, 2, 4, u_124, v_124, w_124)
    
    # A1 (1x4) x B2 (4x4)
    add_sub_term(0, 0, 4, 1, 2, 4, u_124, v_124, w_124)
    add_sub_term(0, 2, 4, 1, 2, 4, u_124, v_124, w_124)
    
    # A2 (2x4) x B1 (4x4)
    add_sub_term(1, 0, 0, 2, 2, 4, u_224, v_224, w_224)
    add_sub_term(1, 2, 0, 2, 2, 4, u_224, v_224, w_224)
    
    # A2 (2x4) x B2 (4x4)
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
    
    # Try Rank 88
    try:
        U_88, V_88, W_88 = _construct_rank_88()
        if _verify_safe(U_88, V_88, W_88):
            if U_88.shape[1] < best_rank:
                best_result = {"U": U_88.tolist(), "V": V_88.tolist(), "W": W_88.tolist()}
                best_rank = U_88.shape[1]
    except Exception:
        pass
        
    return best_result
