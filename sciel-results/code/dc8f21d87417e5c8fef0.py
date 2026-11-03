"""Matrix multiplication 3x4x8 rank-minimal decomposition search."""
import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def _verify_safe(U, V, W, n=3, m=4, p=8, tol=1e-6):
    """
    Verify the decomposition using explicit indexing.
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    Target: T[i,j,k] = 1 for all i,j,k.
    """
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    if U_arr.shape[1] != V_arr.shape[1] or U_arr.shape[1] != W_arr.shape[1]:
        return False
    rank = U_arr.shape[1]
    
    # Build tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Use broadcasting: U[i*m+j, c] has shape (n, m, rank)
    # V[j*p+k, c] has shape (m, p, rank)
    # W[k*n+i, c] has shape (p, n, rank)
    
    # Reshape
    U_3d = U_arr.reshape(n, m, rank)  # (n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)  # (m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)  # (p, n, rank)
    
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # We need to align indices: U[i,j,c], V[j,k,c], W[k,i,c]
    # Use einsum with proper index labels
    # U: i j c, V: j k c, W: k i c -> result: i j k
    T_recon = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    if not np.allclose(T_recon, 1.0, atol=tol):
        return False
    return True

def _construct_rank_88():
    """
    Construct a rank-88 decomposition using block structure.
    Split A (3x4) into A1 (1x4) and A2 (2x4).
    Split B (4x8) into B1 (4x4) and B2 (4x4).
    
    For 1x2x4 blocks: schoolbook rank 8
    For 2x2x4 blocks: two 2x2x2 Strassen blocks, rank 14 each = 28
    
    Total: 4*(8) + 4*(28) = 32 + 112... wait let me recalculate.
    
    Actually:
    A1(1x4) x B1(4x4): split B1 into two 4x2? No.
    
    Let me redo:
    A1 (1x4) x B1 (4x4) -> 1x4x4, split into two 1x2x4: rank 8+8=16
    A1 (1x4) x B2 (4x4) -> 1x4x4, split into two 1x2x4: rank 8+8=16
    A2 (2x4) x B1 (4x4) -> 2x4x4, split into two 2x2x4: rank 14+14=28
    A2 (2x4) x B2 (4x4) -> 2x4x4, split into two 2x2x4: rank 14+14=28
    
    Total: 16+16+28+28 = 88
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
        """Schoolbook 1x2x4, rank 8"""
        rank = 2 * 4
        u = np.zeros((1 * 2, rank))
        v = np.zeros((2 * 4, rank))
        w = np.zeros((4 * 1, rank))
        idx = 0
        for j in range(2):
            for k in range(4):
                u[j, idx] = 1.0
                v[j * 4 + k, idx] = 1.0
                w[k * 1, idx] = 1.0
                idx += 1
        return u, v, w

    def get_2x2x4():
        """
        2x2x4 = two independent 2x2x2 Strassen blocks.
        Rank = 7 + 7 = 14.
        """
        # Strassen 2x2x2 factors
        # U (A parts): 4 entries (A11,A12,A21,A22), 7 columns
        u_s = np.array([
            [1,  0,  1,  0,  1, -1,  0],  # A11
            [0,  0,  0,  0,  1,  0,  1],  # A12
            [0,  1,  0,  0,  0,  1,  0],  # A21
            [1,  1,  0,  1,  0,  0, -1]   # A22
        ])  # shape (4, 7)
        
        # V (B parts): 4 entries (B11,B12,B21,B22), 7 columns
        v_s = np.array([
            [1,  1,  0, -1,  0,  1,  0],  # B11
            [0,  0,  1,  0,  0,  1,  0],  # B12
            [0,  0,  0,  1,  0,  0,  1],  # B21
            [1, -1, -1,  0,  1,  0,  1]   # B22
        ])  # shape (4, 7)
        
        # W (C parts): 4 entries (C11,C21,C12,C22), 7 columns
        # Index: k*n+i where n=2, so C11=(0,0)->0, C21=(0,1)->1, C12=(1,0)->2, C22=(1,1)->3
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
        
        # Build 2x2x4 from two 2x2x2 blocks
        u_224 = np.zeros((4, 14))
        v_224 = np.zeros((8, 14))
        w_224 = np.zeros((8, 14))
        
        # First block: columns 0-3 of B (k=0,1)
        for c in range(7):
            u_224[:, c] = u_s[:, c]
            # V: B11 at (0,0), B12 at (0,1), B21 at (1,0), B22 at (1,1)
            v_224[0, c] = v_s[0, c]   # B11
            v_224[1, c] = v_s[1, c]   # B12
            v_224[4, c] = v_s[2, c]   # B21
            v_224[5, c] = v_s[3, c]   # B22
            # W: C11 at (0,0), C21 at (0,1), C12 at (1,0), C22 at (1,1)
            w_224[0, c] = w_s[0, c]   # C11
            w_224[1, c] = w_s[1, c]   # C21
            w_224[2, c] = w_s[2, c]   # C12
            w_224[3, c] = w_s[3, c]   # C22
        
        # Second block: columns 4-7 of B (k=2,3)
        for c in range(7):
            u_224[:, c+7] = u_s[:, c]
            # V: B11 at (0,2), B12 at (0,3), B21 at (1,2), B22 at (1,3)
            v_224[2, c+7] = v_s[0, c]   # B11
            v_224[3, c+7] = v_s[1, c]   # B12
            v_224[6, c+7] = v_s[2, c]   # B21
            v_224[7, c+7] = v_s[3, c]   # B22
            # W: C11 at (2,0), C21 at (2,1), C12 at (3,0), C22 at (3,1)
            w_224[4, c+7] = w_s[0, c]   # C11
            w_224[5, c+7] = w_s[1, c]   # C21
            w_224[6, c+7] = w_s[2, c]   # C12
            w_224[7, c+7] = w_s[3, c]   # C22
        
        return u_224, v_224, w_224

    u_124, v_124, w_124 = get_1x2x4()
    u_224, v_224, w_224 = get_2x2x4()
    
    # A1 (i=0) x B1 (k=0..3): split B into j=0..1 and j=2..3
    add_sub_term(0, 0, 0, 1, 2, 4, u_124, v_124, w_124)
    add_sub_term(0, 2, 0, 1, 2, 4, u_124, v_124, w_124)
    
    # A1 (i=0) x B2 (k=4..7): split B into j=0..1 and j=2..3
    add_sub_term(0, 0, 4, 1, 2, 4, u_124, v_124, w_124)
    add_sub_term(0, 2, 4, 1, 2, 4, u_124, v_124, w_124)
    
    # A2 (i=1..2) x B1 (k=0..3): split B into j=0..1 and j=2..3
    add_sub_term(1, 0, 0, 2, 2, 4, u_224, v_224, w_224)
    add_sub_term(1, 2, 0, 2, 2, 4, u_224, v_224, w_224)
    
    # A2 (i=1..2) x B2 (k=4..7): split B into j=0..1 and j=2..3
    add_sub_term(1, 0, 4, 2, 2, 4, u_224, v_224, w_224)
    add_sub_term(1, 2, 4, 2, 2, 4, u_224, v_224, w_224)
    
    U_arr = np.array(U_list).T
    V_arr = np.array(V_list).T
    W_arr = np.array(W_list).T
    
    return U_arr, V_arr, W_arr

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Fallback: Schoolbook rank 96
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
