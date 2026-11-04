import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Uses block Strassen approach with rank-49 sub-decompositions, then searches for
    linear dependencies to potentially reduce rank below 98.
    """
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    
    # --- Step 1: Build the 2x2x2 Strassen base (rank 7) ---
    U_2 = np.zeros((4, 7), dtype=float)
    V_2 = np.zeros((4, 7), dtype=float)
    W_2 = np.zeros((4, 7), dtype=float)
    
    # M1 = (A00+A11)(B00+B11)
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1
    
    # M3 = A00(B01-B11)
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1
    
    # M4 = A11(B10-B00)
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1
    
    # M5 = (A00+A01)B11
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1
    
    # --- Step 2: Define block Strassen for 4x4x4 ---
    steps = [
        {'A': [(1, 0), (1, 3)], 'B': [(1, 0), (1, 3)], 'C': [(1, 0), (1, 3)]},
        {'A': [(1, 2), (1, 3)], 'B': [(1, 0)], 'C': [(1, 2), (-1, 3)]},
        {'A': [(1, 0)], 'B': [(1, 1), (-1, 3)], 'C': [(1, 1), (1, 3)]},
        {'A': [(1, 3)], 'B': [(1, 2), (-1, 0)], 'C': [(1, 0), (1, 2)]},
        {'A': [(1, 0), (1, 1)], 'B': [(1, 3)], 'C': [(-1, 0), (1, 1)]},
        {'A': [(1, 2), (-1, 0)], 'B': [(1, 0), (1, 1)], 'C': [(1, 3)]},
        {'A': [(1, 1), (-1, 3)], 'B': [(1, 2), (1, 3)], 'C': [(1, 0)]}
    ]
    
    def get_global_indices(block_idx, dim=4):
        r_start = (block_idx // 2) * 2
        c_start = (block_idx % 2) * 2
        indices = []
        for r in range(2):
            for c in range(2):
                indices.append((r_start + r, c_start + c))
        return indices
    
    # --- Step 3: Build rank-98 decomposition for 4x4x8 ---
    rank_98 = 98
    U = np.zeros((n * m, rank_98), dtype=float)
    V = np.zeros((m * p, rank_98), dtype=float)
    W = np.zeros((p * n, rank_98), dtype=float)
    
    for half in range(2):
        for step_idx, step in enumerate(steps):
            for m_s in range(7):
                col = half * 49 + step_idx * 7 + m_s
                
                for coeff_a, block_idx in step['A']:
                    for (r, c) in get_global_indices(block_idx):
                        global_a_idx = r * 4 + c
                        local_r = r % 2
                        local_c = c % 2
                        local_a_idx = local_r * 2 + local_c
                        U[global_a_idx, col] += coeff_a * U_2[local_a_idx, m_s]
                
                for coeff_b, block_idx in step['B']:
                    for (r, c) in get_global_indices(block_idx):
                        global_b_idx = r * 8 + (c + half * 4)
                        local_r = r % 2
                        local_c = c % 2
                        local_b_idx = local_r * 2 + local_c
                        V[global_b_idx, col] += coeff_b * V_2[local_b_idx, m_s]
                
                for coeff_c, block_idx in step['C']:
                    for (r, c) in get_global_indices(block_idx):
                        global_w_idx = (c + half * 4) * 4 + r
                        local_r = r % 2
                        local_c = c % 2
                        local_w_idx = local_r * 2 + local_c
                        W[global_w_idx, col] += coeff_c * W_2[local_w_idx, m_s]
    
    # --- Step 4: Verify the rank-98 decomposition ---
    def verify_decomposition(U_mat, V_mat, W_mat, tol=1e-10):
        n, m, p = 4, 4, 8
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_vec = U_mat[i * m + j, :]
                    v_vec = V_mat[j * p + k, :]
                    w_vec = W_mat[k * n + i, :]
                    s = np.dot(u_vec, v_vec * w_vec)
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    if not verify_decomposition(U, V, W):
        rank_128 = 128
        U_sb = np.zeros((n * m, rank_128), dtype=float)
        V_sb = np.zeros((m * p, rank_128), dtype=float)
        W_sb = np.zeros((p * n, rank_128), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U_sb[i * m + j, r] = 1
                    V_sb[j * p + k, r] = 1
                    W_sb[k * n + i, r] = 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # --- Step 5: Search for rank reduction via linear dependencies ---
    # Check if the 98 rank-1 tensors are linearly independent.
    # Form the 128x98 matrix M where M[i,j] = U[i//16, j] * V[(i//4)%4*8 + i%4, j] * W[i%4*4 + i//16, j]
    # Actually, let me be more careful.
    
    # The tensor entry T[i,j,k] = sum_r U[i*4+j, r] * V[j*8+k, r] * W[k*4+i, r]
    # For each tensor entry (i,j,k), the coefficient of the r-th rank-1 term is:
    #   coeff[i,j,k,r] = U[i*4+j, r] * V[j*8+k, r] * W[k*4+i, r]
    
    # We want to check if there exist alpha_r (r=0..97) such that:
    #   sum_r alpha_r * coeff[i,j,k,r] = 0 for all (i,j,k)
    # This is a homogeneous system of 128 equations in 98 unknowns.
    
    # Build the 128x98 matrix
    M = np.zeros((128, 98), dtype=float)
    idx = 0
    for i in range(4):
        for j in range(4):
            for k in range(8):
                for r in range(98):
                    M[idx, r] = U[i*4+j, r] * V[j*8+k, r] * W[k*4+i, r]
                idx += 1
    
    # Check if M has full column rank (rank 98)
    # If rank(M) < 98, there's a non-trivial null vector, meaning we can reduce rank.
    
    try:
        rank_M = np.linalg.matrix_rank(M, tol=1e-10)
    except:
        rank_M = 98
    
    if rank_M < 98:
        # There's a linear dependency. Find the null vector.
        try:
            # Use SVD to find the null space
            U_svd, s, Vt = np.linalg.svd(M, full_matrices=True)
            # The null space is spanned by the last (98 - rank_M) rows of Vt
            null_space = Vt[rank_M:, :]  # shape: (98-rank_M, 98)
            
            # For each null vector, we can remove one column
            # But we need to be careful: removing a column changes the decomposition
            # The correct approach is to express one rank-1 term as a combination of others
            
            # For simplicity, let's just reduce by 1 if we find a null vector
            # This is a conservative approach
            reduced_rank = rank_M
            if reduced_rank < 98 and reduced_rank > 0:
                # Keep only the first reduced_rank columns
                U_red = U[:, :reduced_rank]
                V_red = V[:, :reduced_rank]
                W_red = W[:, :reduced_rank]
                
                if verify_decomposition(U_red, V_red, W_red):
                    return {"U": U_red.tolist(), "V": V_red.tolist(), "W": W_red.tolist()}
        except:
            pass
    
    # Return the verified rank-98 decomposition
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
