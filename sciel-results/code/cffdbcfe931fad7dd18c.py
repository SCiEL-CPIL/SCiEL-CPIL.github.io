"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    start_time = time.time()
    deadline = start_time + 115.0

    # Strassen 2x2x2 rank-7 factors
    # Rows of U correspond to (i, j) in A (2x2)
    # Rows of V correspond to (j, k) in B (2x2)
    # Rows of W correspond to (k, i) in C (2x2)
    
    # Standard Strassen:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # U (A) coefficients:
    # a11: M1, M3, M5, M6
    # a12: M5, M7
    # a21: M2, M6
    # a22: M1, M2, M4, M7
    
    U_2x2 = np.array([
        [1, 0, 1, 0, 1, 1, 0], # a11
        [0, 0, 0, 0, 1, 0, 1], # a12
        [0, 1, 0, 0, 0, -1, 0],# a21
        [1, 1, 0, 1, 0, 0, -1] # a22
    ])
    
    # V (B) coefficients:
    # b11: M1, M2, M4, M6
    # b12: M3, M6
    # b21: M4, M7
    # b22: M1, M5, M7
    
    V_2x2 = np.array([
        [1, 1, 0, -1, 0, 1, 0], # b11
        [0, 0, 1, 0, 0, 1, 0],  # b12
        [0, 0, 0, 1, 0, 0, 1],  # b21
        [1, 0, -1, 0, 1, 0, 1]  # b22
    ])
    
    # W (C) coefficients:
    # c11: M1, M4, M5, M7
    # c12: M3, M5
    # c21: M2, M4
    # c22: M1, M2, M3, M6
    
    W_2x2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # c11
        [0, 0, 1, 0, 1, 0, 0],   # c12
        [0, 1, 0, 1, 0, 0, 0],   # c21
        [1, -1, 1, 0, 0, 1, 0]   # c22
    ])
    
    # Construct the 84-rank decomposition
    # 3 blocks of 4x4x2
    # Each 4x4x2 block is composed of 4 independent 2x2x2 interactions
    # Total 12 interactions * 7 = 84
    
    num_interactions = 12
    rank_per_interaction = 7
    total_rank = num_interactions * rank_per_interaction
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # Map interaction index to block positions
    # Interaction idx -> (block_of_2, sub_block_of_2, sub_block_of_2)
    # We have 3 output blocks (k=0,1,2) each of size 2
    # For each output block, we have 2x2 = 4 input block interactions
    # A is 2x2 blocks (0,0 to 1,1)
    # B is 2x2 blocks (0,0 to 1,1) but only 2 columns? 
    # Wait, B is 4x6. Split into 3 chunks of 4x2.
    # Each 4x2 chunk is split into two 2x2 blocks (top and bottom).
    
    col_offset = 0
    for chunk_idx in range(3): # 3 chunks of width 2
        for a_row_block in range(2): # 0, 1
            for a_col_block in range(2): # 0, 1
                for b_row_block in range(2): # 0, 1
                    # b_col_block is determined by chunk_idx?
                    # No, B is 4x6.
                    # B_chunk is 4x2.
                    # B_chunk is split into B_top (2x2) and B_bot (2x2).
                    # So b_row_block is 0 or 1.
                    # The column index in B is chunk_idx * 2 + b_col_block?
                    # No, B_chunk is 4x2.
                    # If we treat B_chunk as two 2x2 matrices stacked,
                    # then b_row_block corresponds to the row of the 2x2 matrix in B?
                    # Yes.
                    
                    # Interaction: A[a_row_block, a_col_block] * B[b_row_block, :2]
                    # Result C[a_row_block, chunk_idx]
                    
                    for c in range(7):
                        col = col_offset + c
                        
                        # U part: A[a_row_block, a_col_block]
                        i_base = a_row_block * 2
                        j_base = a_col_block * 2
                        for di in range(2):
                            for dj in range(2):
                                i = i_base + di
                                j = j_base + dj
                                U[i * m + j, col] = U_2x2[di * 2 + dj, c]
                        
                        # V part: B[b_row_block, chunk_idx]
                        j_base = b_row_block * 2
                        k_base = chunk_idx * 2
                        for dj in range(2):
                            for dk in range(2):
                                j = j_base + dj
                                k = k_base + dk
                                V[j * p + k, col] = V_2x2[dj * 2 + dk, c]
                        
                        # W part: C[a_row_block, chunk_idx]
                        i_base = a_row_block * 2
                        k_base = chunk_idx * 2
                        for dk in range(2):
                            for di in range(2):
                                k = k_base + dk
                                i = i_base + di
                                W[k * n + i, col] = W_2x2[dk * 2 + di, c]
                    
                    col_offset += 7

    def verify_decomposition(U_mat, V_mat, W_mat):
        """
        Verify if the decomposition U, V, W correctly computes AB.
        T_decomp[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        Should be 1 for all i,j,k.
        """
        rank = U_mat.shape[1]
        # We only need to check if the tensor matches the identity tensor
        # T[i,j,k] = 1
        # Efficient check:
        # For each r, the contribution is outer(U[:,r], V[:,r]) * W[:,r]
        # We can accumulate the tensor.
        
        # To be fast, we can check a few random points or use matrix multiplication properties.
        # But for small sizes, full check is okay.
        
        T_check = np.zeros((n, m, p), dtype=float)
        
        # Optimization: Precompute V_cols and W_cols?
        # Or just loop r.
        for r in range(rank):
            u_col = U_mat[:, r]
            v_col = V_mat[:, r]
            w_col = W_mat[:, r]
            
            # T[i, j, k] += u[i,j] v[j,k] w[k,i]
            # We can do:
            # M = np.outer(u_col, v_col) # n*m x m*p
            # Then for each i,k, sum over j?
            # T[i,k] += sum_j M[i*m+j, j*p+k] w[k*n+i]
            
            # Simpler:
            # Reshape u to n x m, v to m x p, w to p x n
            # T += np.einsum('ij,jk,ki->ik', u_reshaped, v_reshaped, w_reshaped)? No.
            # T[i,j,k] = u[i,j] v[j,k] w[k,i]
            
            u_reshaped = u_col.reshape(n, m)
            v_reshaped = v_col.reshape(m, p)
            w_reshaped = w_col.reshape(p, n)
            
            # This is a rank-1 update to the 3D tensor
            # T[i,j,k] += u_reshaped[i,j] * v_reshaped[j,k] * w_reshaped[k,i]
            
            # We can do this with broadcasting
            # term[i,j,k] = u[i,j] * v[j,k] * w[k,i]
            # This is slow in Python loop, so let's use vectorization if possible
            # Or just accept the loop for verification (rank is ~80, size is 96)
            
            # Actually, we can check the decomposition by multiplying random matrices?
            # If we pick random A, B, and check if C = AB matches the decomposition?
            # That is more robust and faster to code.
            
        # Let's use the random test method for verification speed
        # But we need to return the decomposition, so we must be sure it's exact.
        # The Strassen decomposition is exact.
        # The rank reduction might break exactness if not careful.
        # So we need a robust check.
        
        # Check tensor identity
        # T[i,j,k] = 1
        # Let's just compute the full tensor once for the base case to be sure.
        # For the reduction steps, we will check if the new tensor equals the old one.
        
        pass # Placeholder, will implement specific check below

    # Initial verification of the 84-rank decomposition
    # We trust the Strassen construction, but let's do a quick check.
    # We can verify by checking if U @ V @ W structure holds?
    # No, let's just proceed to rank reduction.
    
    def is_valid_decomposition(U_mat, V_mat, W_mat, target_tensor=None):
        """
        Check if decomposition matches target tensor (default all 1s).
        Uses random projection for speed? No, exact check.
        """
        rank = U_mat.shape[1]
        # Compute T[i,j,k]
        # T = sum_r u_r[i,j] v_r[j,k] w_r[k,i]
        
        # We can optimize this check by using the fact that T should be 1.
        # Let's check a few specific indices to avoid full O(n*m*p*rank) loop if possible?
        # But n*m*p = 96, rank ~ 80. 96*80 = 7680 ops. Very fast.
        
        T_val = np.zeros((n, m, p), dtype=float)
        
        # Vectorized accumulation
        # u: (n*m, rank), v: (m*p, rank), w: (p*n, rank)
        # T[i,j,k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # This is hard to vectorize easily without large intermediate arrays.
        # But 7680 ops is trivial.
        
        for r in range(rank):
            u_r = U_mat[:, r]
            v_r = V_mat[:, r]
            w_r = W_mat[:, r]
            
            # u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
            # Let's do it in chunks or just loop i,k
            for i in range(n):
                for k in range(p):
                    w_val = w_r[k*n + i]
                    if w_val == 0: continue
                    # Sum over j
                    # u_vals = u_r[i*m : (i+1)*m]
                    # v_vals = v_r[k::p][:m] # j*p+k for j in 0..m-1
                    # sum(u_vals * v_vals) * w_val
                    u_slice = u_r[i*m:(i+1)*m]
                    v_slice = v_r[k::p][:m]
                    T_val[i, :, k] += np.dot(u_slice, v_slice) * w_val
                    
        return np.allclose(T_val, 1.0, atol=1e-9)

    # Verify base decomposition
    if not is_valid_decomposition(U, V, W):
        # If base is invalid, return it anyway (should not happen)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Rank Reduction
    # Try to remove columns one by one.
    # Order: Random or sorted by norm?
    # We'll try random shuffles and greedy removal.
    
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    best_rank = total_rank
    
    # Pre-compute tensor for base to compare against?
    # No, is_valid_decomposition is fast enough.
    
    # Shuffle column indices
    indices = np.arange(total_rank)
    np.random.shuffle(indices)
    
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = total_rank
    removed = [False] * total_rank
    
    # We have ~115 seconds.
    # Each check is ~0.1ms?
    # We can do many checks.
    
    checks_done = 0
    max_checks = 500 # Safety
    
    for idx in indices:
        if time.time() > deadline - 10:
            break
        if checks_done > max_checks:
            break
        if removed[idx]:
            continue
            
        # Try removing idx
        # Create new arrays without idx
        # To be efficient, we can just mask?
        # But we need to return clean arrays.
        # For checking, we can just skip this column in the sum?
        
        # Let's modify is_valid to take a mask?
        # Or just create subarrays.
        # Creating subarrays is O(R). R~80. Fast.
        
        new_U = np.delete(current_U, idx, axis=1)
        new_V = np.delete(current_V, idx, axis=1)
        new_W = np.delete(current_W, idx, axis=1)
        
        if is_valid_decomposition(new_U, new_V, new_W):
            current_U = new_U
            current_V = new_V
            current_W = new_W
            current_rank -= 1
            removed[idx] = True
            if current_rank < best_rank:
                best_U = current_U.copy()
                best_V = current_V.copy()
                best_W = current_W.copy()
                best_rank = current_rank
                
        checks_done += 1
        
    # Second pass with different shuffle
    if time.time() < deadline - 20:
        np.random.shuffle(indices)
        # Reset to best found so far? Or continue from current?
        # Continue from current to try to remove more.
        
        for idx in indices:
            if time.time() > deadline - 5:
                break
            if removed[idx]:
                continue
                
            new_U = np.delete(current_U, idx, axis=1)
            new_V = np.delete(current_V, idx, axis=1)
            new_W = np.delete(current_W, idx, axis=1)
            
            if is_valid_decomposition(new_U, new_V, new_W):
                current_U = new_U
                current_V = new_V
                current_W = new_W
                current_rank -= 1
                removed[idx] = True
                if current_rank < best_rank:
                    best_U = current_U.copy()
                    best_V = current_V.copy()
                    best_W = current_W.copy()
                    best_rank = current_rank

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
