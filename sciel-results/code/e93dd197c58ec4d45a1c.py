"""Matrix multiplication 3x4x8 rank-minimal decomposition search."""
import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def _verify_safe(U, V, W, n=3, m=4, p=8, tol=1e-6):
    """
    Safely verify the decomposition using explicit loops to avoid einsum broadcasting issues.
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    Target: T[i,j,k] = 1 for all i,j,k.
    """
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    rank = U_arr.shape[1]
    
    T_recon = np.zeros((n, m, p))
    for c in range(rank):
        # Extract vectors for this term
        u_c = U_arr[:, c]  # size n*m
        v_c = V_arr[:, c]  # size m*p
        w_c = W_arr[:, c]  # size p*n
        
        # Reshape to 2D
        u_2d = u_c.reshape(n, m)
        v_2d = v_c.reshape(m, p)
        w_2d = w_c.reshape(p, n)
        
        # Add outer product-like term
        # Term[i,j,k] = u_2d[i,j] * v_2d[j,k] * w_2d[k,i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] += u_2d[i, j] * v_2d[j, k] * w_2d[k, i]
    
    if not np.allclose(T_recon, 1.0, atol=tol):
        return False
    return True

def _strassen_2x2x2():
    """
    Returns (u, v, w) matrices for 2x2x2 Strassen decomposition.
    u: shape (4, 7) - A flattened
    v: shape (4, 7) - B flattened
    w: shape (4, 7) - C flattened with convention k*2+i
    """
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # U vectors (A parts): A11, A12, A21, A22
    u = np.array([
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1]   # M7
    ]).T  # Shape (4, 7)
    
    # V vectors (B parts): B11, B12, B21, B22
    v = np.array([
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [-1, 0, 1, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1]    # M7
    ]).T  # Shape (4, 7)
    
    # W vectors (C parts): C11 (k=0,i=0)->0, C21 (k=0,i=1)->1, C12 (k=1,i=0)->2, C22 (k=1,i=1)->3
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    w = np.zeros((4, 7))
    # M1: C11 + C22
    w[0, 0] = 1
    w[3, 0] = 1
    # M2: C21 - C22
    w[1, 1] = 1
    w[3, 1] = -1
    # M3: C12 + C22
    w[2, 2] = 1
    w[3, 2] = 1
    # M4: C11 + C21
    w[0, 3] = 1
    w[1, 3] = 1
    # M5: -C11 + C12
    w[0, 4] = -1
    w[2, 4] = 1
    # M6: C22
    w[3, 5] = 1
    # M7: C11
    w[0, 6] = 1
    
    return u, v, w

def _get_1x2x4():
    """
    Returns (u, v, w) for 1x2x4 schoolbook decomposition. Rank 8.
    u: shape (2, 8) - A flattened (1x2)
    v: shape (8, 8) - B flattened (2x4)
    w: shape (4, 8) - C flattened (4x1) with convention k*1+i = k
    """
    sn, sm, sp = 1, 2, 4
    u = np.zeros((sn * sm, 8))  # (2, 8)
    v = np.zeros((sm * sp, 8))  # (8, 8)
    w = np.zeros((sp * sn, 8))  # (4, 8)
    
    idx = 0
    for j in range(sm):
        for k in range(sp):
            u[j, idx] = 1.0
            v[j * sp + k, idx] = 1.0
            w[k, idx] = 1.0
            idx += 1
    return u, v, w

def _get_2x2x4():
    """
    Returns (u, v, w) for 2x2x4 using Strassen for two 2x2x2 blocks. Rank 14.
    u: shape (4, 14)
    v: shape (8, 14)
    w: shape (8, 14) - C flattened (4x2) with convention k*2+i
    """
    sn, sm, sp = 2, 2, 4
    u_2x2, v_2x2, w_2x2 = _strassen_2x2x2()
    
    u_2x2x4 = np.zeros((4, 14))
    v_2x2x4 = np.zeros((8, 14))
    w_2x2x4 = np.zeros((8, 14))
    
    # First half (k=0,1)
    for c in range(7):
        u_2x2x4[:, c] = u_2x2[:, c]
        # V indices: B(j,k). j=0,1; k=0,1
        # B11(j=0,k=0)->0, B12(j=0,k=1)->1, B21(j=1,k=0)->4, B22(j=1,k=1)->5
        v_2x2x4[0, c] = v_2x2[0, c]
        v_2x2x4[1, c] = v_2x2[1, c]
        v_2x2x4[4, c] = v_2x2[2, c]
        v_2x2x4[5, c] = v_2x2[3, c]
        
        # W indices: C(k,i). k=0,1; i=0,1
        # C11(i=0,k=0)->0, C21(i=1,k=0)->1, C12(i=0,k=1)->2, C22(i=1,k=1)->3
        w_2x2x4[0, c] = w_2x2[0, c]
        w_2x2x4[1, c] = w_2x2[1, c]
        w_2x2x4[2, c] = w_2x2[2, c]
        w_2x2x4[3, c] = w_2x2[3, c]
    
    # Second half (k=2,3)
    for c in range(7):
        u_2x2x4[:, c+7] = u_2x2[:, c]
        # V indices: B(j,k). j=0,1; k=2,3
        # B11(j=0,k=2)->2, B12(j=0,k=3)->3, B21(j=1,k=2)->6, B22(j=1,k=3)->7
        v_2x2x4[2, c+7] = v_2x2[0, c]
        v_2x2x4[3, c+7] = v_2x2[1, c]
        v_2x2x4[6, c+7] = v_2x2[2, c]
        v_2x2x4[7, c+7] = v_2x2[3, c]
        
        # W indices: C(k,i). k=2,3; i=0,1
        # C11(i=0,k=2)->4, C21(i=1,k=2)->5, C12(i=0,k=3)->6, C22(i=1,k=3)->7
        w_2x2x4[4, c+7] = w_2x2[0, c]
        w_2x2x4[5, c+7] = w_2x2[1, c]
        w_2x2x4[6, c+7] = w_2x2[2, c]
        w_2x2x4[7, c+7] = w_2x2[3, c]
    
    return u_2x2x4, v_2x2x4, w_2x2x4

def _get_2x1x4():
    """
    Returns (u, v, w) for 2x1x4 schoolbook decomposition. Rank 8.
    u: shape (2, 8) - A flattened (2x1)
    v: shape (4, 8) - B flattened (1x4)
    w: shape (8, 8) - C flattened (4x2) with convention k*2+i
    """
    sn, sm, sp = 2, 1, 4
    u = np.zeros((sn * sm, 8))  # (2, 8)
    v = np.zeros((sm * sp, 8))  # (4, 8)
    w = np.zeros((sp * sn, 8))  # (8, 8)
    
    idx = 0
    for i in range(sn):
        for k in range(sp):
            u[i * sm, idx] = 1.0
            v[k, idx] = 1.0
            w[k * sn + i, idx] = 1.0
            idx += 1
    return u, v, w

def _get_3x4x4_strassen():
    """
    Returns (u, v, w) for 3x4x4 decomposition using Strassen block splitting.
    Split A(3x4) into A1(1x4) and A2(2x4).
    Split B(4x4) into B1(4x2) and B2(4x2).
    Terms: A1B1(1x4x2, rank8) + A1B2(1x4x2, rank8) + A2B1(2x4x2, rank14) + A2B2(2x4x2, rank14)
    Total rank: 8+8+14+14 = 44.
    """
    u_cols = []
    v_cols = []
    w_cols = []
    
    def add_term(i_off, j_off, k_off, sub_n, sub_m, sub_p, u_sub, v_sub, w_sub):
        rank_sub = u_sub.shape[1]
        n, m, p = 3, 4, 4
        for c in range(rank_sub):
            u_col = np.zeros(12)
            v_col = np.zeros(16)
            w_col = np.zeros(12)
            
            for i in range(sub_n):
                for j in range(sub_m):
                    u_col[(i_off + i) * m + (j_off + j)] = u_sub[i * sub_m + j, c]
            
            for j in range(sub_m):
                for k in range(sub_p):
                    v_col[(j_off + j) * p + (k_off + k)] = v_sub[j * sub_p + k, c]
            
            for k in range(sub_p):
                for i in range(sub_n):
                    w_col[(k_off + k) * n + (i_off + i)] = w_sub[k * sub_n + i, c]
            
            u_cols.append(u_col)
            v_cols.append(v_col)
            w_cols.append(w_col)
    
    # Get sub-decompositions
    # 1x4x2: Schoolbook rank 8
    def get_1x4x2():
        sn, sm, sp = 1, 4, 2
        u = np.zeros((sm, 8))
        v = np.zeros((sm * sp, 8))
        w = np.zeros((sp * sn, 8))
        idx = 0
        for j in range(sm):
            for k in range(sp):
                u[j, idx] = 1.0
                v[j * sp + k, idx] = 1.0
                w[k, idx] = 1.0
                idx += 1
        return u, v, w
    
    # 2x4x2: Use Strassen for two 2x2x1? No, 2x4x2.
    # Split into two 2x2x1 blocks? Rank of 2x2x1 is 4 (schoolbook). 4+4=8.
    # Or use Strassen 2x2x2 for the whole 2x2 part and replicate?
    # 2x4x2: A(2x4) B(4x2).
    # Schoolbook rank = 2*4*2 = 16.
    # Better: Split A into A1(2x2), A2(2x2). B into B1(2x2), B2(2x2).
    # Terms: A1B1(2x2x2, rank7), A1B2(2x2x2, rank7), A2B1(2x2x2, rank7), A2B2(2x2x2, rank7).
    # Total rank = 28.
    # Or keep B as one 4x2 and A as 2x4.
    # Actually, let's just use schoolbook for 2x4x2 to be safe and simple, rank 16.
    # Wait, 14 is claimed in the previous code for 2x2x4.
    # Let's check 2x4x2 vs 2x2x4.
    # Previous code used 2x2x4 for A2B1 where A2(2x2) and B1(2x4).
    # Here I am splitting B(4x4) into B1(4x2) and B2(4x2).
    # So A2(2x4) and B1(4x2). This is 2x4x2.
    # Rank(2x4x2) schoolbook is 16.
    # Can we do better?
    # Strassen 2x2x2 gives rank 7 for 2x2x2.
    # If we split 2x4x2 into two 2x2x1 blocks, rank is 4+4=8.
    # If we split 2x4x2 into two 2x2x2 blocks? No, dimensions don't match.
    # Let's use the 2x2x4 decomposition we already have and swap A and B roles?
    # The tensor M_{2,4,2} is isomorphic to M_{2,2,4} via index permutation?
    # A[i,j] B[j,k].
    # If I use the _get_2x2x4() result, it assumes A is 2x2 and B is 2x4.
    # I need A to be 2x4 and B to be 4x2.
    # I can construct 2x4x2 by taking two 2x2x1 decompositions.
    # 2x2x1 schoolbook rank 4.
    # Term 1: A[:, 0:2] B[0:2, :] -> 2x2x1. Rank 4.
    # Term 2: A[:, 2:4] B[2:4, :] -> 2x2x1. Rank 4.
    # Total rank 8. This is much better than 14.
    
    def get_2x2x1():
        sn, sm, sp = 2, 2, 1
        u = np.zeros((4, 4))
        v = np.zeros((2, 4))
        w = np.zeros((2, 4))
        idx = 0
        for i in range(sn):
            for j in range(sm):
                u[i * sm + j, idx] = 1.0
                v[j, idx] = 1.0
                w[i, idx] = 1.0 # k=0 always, so index is 0*2+i = i
                idx += 1
        return u, v, w
    
    def get_2x4x2():
        # Two 2x2x1 blocks
        u_221, v_221, w_221 = get_2x2x1()
        u_full = np.zeros((4, 8))
        v_full = np.zeros((8, 8))
        w_full = np.zeros((4, 8))
        
        # Block 1: j=0,1
        for c in range(4):
            u_full[:, c] = u_221[:, c]
            # V indices: j*2+k. j=0,1; k=0.
            # B11(j=0,k=0)->0. B21(j=1,k=0)->2.
            v_full[0, c] = v_221[0, c]
            v_full[2, c] = v_221[1, c]
            # W indices: k*2+i. k=0.
            # C11(i=0)->0. C21(i=1)->1.
            w_full[0, c] = w_221[0, c]
            w_full[1, c] = w_221[1, c]
        
        # Block 2: j=2,3
        for c in range(4):
            u_full[:, c+4] = u_221[:, c]
            # V indices: j=2,3; k=0.
            # B11(j=2,k=0)->4. B21(j=3,k=0)->6.
            v_full[4, c+4] = v_221[0, c]
            v_full[6, c+4] = v_221[1, c]
            # W indices: same.
            w_full[0, c+4] = w_221[0, c]
            w_full[1, c+4] = w_221[1, c]
            
        return u_full, v_full, w_full
    
    u_1x4x2, v_1x4x2, w_1x4x2 = get_1x4x2()
    u_2x4x2, v_2x4x2, w_2x4x2 = get_2x4x2()
    
    # A1B1: A1(1x4), B1(4x2). i_off=0, j_off=0, k_off=0
    add_term(0, 0, 0, 1, 4, 2, u_1x4x2, v_1x4x2, w_1x4x2)
    # A1B2: A1(1x4), B2(4x2). i_off=0, j_off=0, k_off=2
    add_term(0, 0, 2, 1, 4, 2, u_1x4x2, v_1x4x2, w_1x4x2)
    # A2B1: A2(2x4), B1(4x2). i_off=1, j_off=0, k_off=0
    add_term(1, 0, 0, 2, 4, 2, u_2x4x2, v_2x4x2, w_2x4x2)
    # A2B2: A2(2x4), B2(4x2). i_off=1, j_off=0, k_off=2
    add_term(1, 0, 2, 2, 4, 2, u_2x4x2, v_2x4x2, w_2x4x2)
    
    u = np.array(u_cols).T
    v = np.array(v_cols).T
    w = np.array(w_cols).T
    
    return u, v, w

def _get_3x4x4():
    """
    Returns (u, v, w) for 3x4x4.
    Uses the 3x4x4 Strassen hybrid above. Rank 44.
    """
    return _get_3x4x4_strassen()

def _get_3x4x8():
    """
    Returns (u, v, w) for 3x4x8.
    Split B(4x8) into B1(4x4) and B2(4x4).
    C = A B1 + A B2.
    A B1 is 3x4x4 (rank 44).
    A B2 is 3x4x4 (rank 44).
    Total rank = 88.
    """
    n, m, p = 3, 4, 8
    u_cols = []
    v_cols = []
    w_cols = []
    
    u_344, v_344, w_344 = _get_3x4x4()
    
    # Term 1: A B1. k_off=0.
    rank_sub = u_344.shape[1]
    for c in range(rank_sub):
        u_col = np.zeros(12)
        v_col = np.zeros(32)
        w_col = np.zeros(24)
        
        # U is same
        u_col[:] = u_344[:, c]
        
        # V: B(j,k). k=0..3.
        # v_344 has indices for B(j, k) where k=0..3.
        # In global, k=0..3 maps to 0..3.
        # v_344 shape is (16, rank). Index j*4+k.
        # Global V index is j*8+k.
        for j in range(4):
            for k in range(4):
                v_col[j * 8 + k] = v_344[j * 4 + k, c]
        
        # W: C(k,i). k=0..3.
        # w_344 has indices for C(k, i) where k=0..3.
        # In global, k=0..3 maps to 0..3.
        # w_344 shape is (12, rank). Index k*3+i.
        # Global W index is k*3+i.
        for k in range(4):
            for i in range(3):
                w_col[k * 3 + i] = w_344[k * 3 + i, c]
        
        u_cols.append(u_col)
        v_cols.append(v_col)
        w_cols.append(w_col)
    
    # Term 2: A B2. k_off=4.
    for c in range(rank_sub):
        u_col = np.zeros(12)
        v_col = np.zeros(32)
        w_col = np.zeros(24)
        
        # U is same
        u_col[:] = u_344[:, c]
        
        # V: B(j,k). k=4..7.
        # v_344 has indices for B(j, k') where k'=0..3.
        # Global k = k' + 4.
        for j in range(4):
            for k_prime in range(4):
                global_k = k_prime + 4
                v_col[j * 8 + global_k] = v_344[j * 4 + k_prime, c]
        
        # W: C(k,i). k=4..7.
        # w_344 has indices for C(k', i) where k'=0..3.
        # Global k = k' + 4.
        for k_prime in range(4):
            global_k = k_prime + 4
            for i in range(3):
                w_col[global_k * 3 + i] = w_344[k_prime * 3 + i, c]
        
        u_cols.append(u_col)
        v_cols.append(v_col)
        w_cols.append(w_col)
    
    u = np.array(u_cols).T
    v = np.array(v_cols).T
    w = np.array(w_cols).T
    
    return u, v, w

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    best_result = None
    best_rank = float('inf')
    
    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    if time.time() < deadline:
        try:
            U_new, V_new, W_new = _get_3x4x8()
            if _verify_safe(U_new.tolist(), V_new.tolist(), W_new.tolist()):
                rank_new = U_new.shape[1]
                if rank_new < best_rank:
                    best_result = {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
                    best_rank = rank_new
        except Exception as e:
            print(f"Error in new decomposition: {e}")
    
    return best_result

# EVOLVE-BLOCK-START
# (The construct function is already defined above)
# EVOLVE-BLOCK-END
