import numpy as np
import time

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 5x5 matrix multiplication.
    Returns dict with 'U', 'V', 'W' matrices of shapes (25, rank), (25, rank), (25, rank).
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit

    n = m = p = 5

    # Naive schoolbook decomposition (rank 125)
    rank_naive = n * m * p
    U = np.zeros((n * m, rank_naive), dtype=float)
    V = np.zeros((m * p, rank_naive), dtype=float)
    W = np.zeros((p * n, rank_naive), dtype=float)

    r = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k, r] = 1.0
                V[k * p + j, r] = 1.0
                W[j * n + i, r] = 1.0
                r += 1

    best_U, best_V, best_W = U, V, W
    best_rank = rank_naive

    # Block decomposition: partition 5x5 into blocks of sizes [2, 2, 1]
    # This gives us a block structure where we can apply Strassen's algorithm
    # to 2x2 blocks and handle the 1x1 boundary terms separately.
    
    # Block sizes
    block_sizes = [2, 2, 1]
    
    # Build the decomposition using block-based approach
    # For each combination of row-block i, col-block j, and middle-block k:
    # - If all three are 2x2 blocks, use Strassen (rank 7)
    # - Otherwise, use schoolbook for that block combination
    
    # The total number of block combinations:
    # Row blocks: 3 (sizes 2,2,1)
    # Col blocks: 3 (sizes 2,2,1)  
    # Middle blocks: 3 (sizes 2,2,1)
    
    # Let's construct the full decomposition by iterating over block positions
    U_block = []
    V_block = []
    W_block = []
    
    # Compute block offsets
    row_offsets = [0]
    col_offsets = [0]
    mid_offsets = [0]
    for s in block_sizes:
        row_offsets.append(row_offsets[-1] + s)
        col_offsets.append(col_offsets[-1] + s)
        mid_offsets.append(mid_offsets[-1] + s)
    
    # For each combination of (row_block, mid_block, col_block), add the appropriate decomposition
    # Strassen's algorithm for 2x2x2 blocks uses 7 multiplications
    # For non-2x2x2 combinations, use schoolbook
    
    r_idx = 0
    for bi in range(3):  # row block index
        for bk in range(3):  # middle block index
            for bj in range(3):  # col block index
                si = block_sizes[bi]
                sk = block_sizes[bk]
                sj = block_sizes[bj]
                
                if si == 2 and sk == 2 and sj == 2:
                    # Use Strassen's algorithm for this 2x2x2 block
                    # Strassen uses 7 rank-1 terms
                    # The 7 terms for multiplying 2x2 matrices:
                    # M1 = (A11 + A22)(B11 + B22)
                    # M2 = (A21 + A22)B11
                    # M3 = A11(B21 - B22)
                    # M4 = A22(B12 - B11)
                    # M5 = (A11 + A12)B22
                    # M6 = (A21 - A11)(B11 + B12)
                    # M7 = (A12 - A22)(B21 + B22)
                    # C11 = M1 + M4 - M5 + M7
                    # C12 = M3 + M5
                    # C21 = M2 + M4
                    # C22 = M1 - M2 + M3 + M6
                    
                    # We need to express this in terms of the global indices
                    # Row offset for block bi: row_offsets[bi]
                    # Col offset for block bj: col_offsets[bj]
                    # Mid offset for block bk: mid_offsets[bk]
                    
                    ro = row_offsets[bi]
                    co = col_offsets[bj]
                    mo = mid_offsets[bk]
                    
                    # The 7 Strassen terms
                    # Each term has specific U, V, W vectors
                    strassen_terms = [
                        # Term 1: (A11+A22)(B11+B22)
                        ([1,1,0,0], [1,0,1,0], [1,0,0,1]),  # W maps to C11, C22
                        # Term 2: (A21+A22)B11
                        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
                        # Term 3: A11(B21-B22)
                        ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),
                        # Term 4: A22(B12-B11)
                        ([0,0,0,1], [1,-1,0,0], [1,0,0,0]),
                        # Term 5: (A11+A12)B22
                        ([1,1,0,0], [0,0,0,1], [0,0,-1,1]),
                        # Term 6: (A21-A11)(B11+B12)
                        ([0,0,-1,1], [1,1,0,0], [0,0,0,1]),
                        # Term 7: (A12-A22)(B21+B22)
                        ([0,1,0,-1], [0,1,1,0], [1,0,0,0]),
                    ]
                    
                    for u_vec, v_vec, w_vec in strassen_terms:
                        # Add this rank-1 term
                        for idx in range(si * sk):
                            u_val = u_vec[idx]
                            if u_val != 0:
                                row_in_block = idx // sk
                                mid_in_block = idx % sk
                                global_row = ro + row_in_block
                                global_mid = mo + mid_in_block
                                U_block.append((global_row * m + global_mid, u_val))
                        
                        for idx in range(sk * sj):
                            v_val = v_vec[idx]
                            if v_val != 0:
                                mid_in_block = idx // sj
                                col_in_block = idx % sj
                                global_mid = mo + mid_in_block
                                global_col = co + col_in_block
                                V_block.append((global_mid * p + global_col, v_val))
                        
                        for idx in range(sj * si):
                            w_val = w_vec[idx]
                            if w_val != 0:
                                col_in_block = idx // si
                                row_in_block = idx % si
                                global_col = co + col_in_block
                                global_row = ro + row_in_block
                                W_block.append((global_col * n + global_row, w_val))
                        r_idx += 1
                else:
                    # Use schoolbook for this block combination
                    for i in range(si):
                        for k in range(sk):
                            for j in range(sj):
                                global_row = row_offsets[bi] + i
                                global_mid = mid_offsets[bk] + k
                                global_col = col_offsets[bj] + j
                                U_block.append((global_row * m + global_mid, 1.0))
                                V_block.append((global_mid * p + global_col, 1.0))
                                W_block.append((global_col * n + global_row, 1.0))
                                r_idx += 1
    
    # Now we have a list of (index, value) tuples for each of U, V, W
    # We need to group them by the rank index (r_idx)
    
    # Reconstruct U, V, W from the block contributions
    # Each rank-1 term contributes to one column of U, V, W
    U_new = np.zeros((n * m, r_idx), dtype=float)
    V_new = np.zeros((m * p, r_idx), dtype=float)
    W_new = np.zeros((p * n, r_idx), dtype=float)
    
    # We need to re-do this more carefully
    # Let's rebuild from scratch with proper tracking
    
    U_list = []
    V_list = []
    W_list = []
    
    r_idx = 0
    for bi in range(3):
        for bk in range(3):
            for bj in range(3):
                si = block_sizes[bi]
                sk = block_sizes[bk]
                sj = block_sizes[bj]
                
                if si == 2 and sk == 2 and sj == 2:
                    ro = row_offsets[bi]
                    co = col_offsets[bj]
                    mo = mid_offsets[bk]
                    
                    strassen_terms = [
                        ([1,1,0,0], [1,0,1,0], [1,0,0,1]),
                        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
                        ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),
                        ([0,0,0,1], [1,-1,0,0], [1,0,0,0]),
                        ([1,1,0,0], [0,0,0,1], [0,0,-1,1]),
                        ([0,0,-1,1], [1,1,0,0], [0,0,0,1]),
                        ([0,1,0,-1], [0,1,1,0], [1,0,0,0]),
                    ]
                    
                    for u_vec, v_vec, w_vec in strassen_terms:
                        u_col = np.zeros(n * m)
                        v_col = np.zeros(m * p)
                        w_col = np.zeros(p * n)
                        
                        for idx in range(si * sk):
                            u_val = u_vec[idx]
                            if u_val != 0:
                                row_in_block = idx // sk
                                mid_in_block = idx % sk
                                global_row = ro + row_in_block
                                global_mid = mo + mid_in_block
                                u_col[global_row * m + global_mid] = u_val
                        
                        for idx in range(sk * sj):
                            v_val = v_vec[idx]
                            if v_val != 0:
                                mid_in_block = idx // sj
                                col_in_block = idx % sj
                                global_mid = mo + mid_in_block
                                global_col = co + col_in_block
                                v_col[global_mid * p + global_col] = v_val
                        
                        for idx in range(sj * si):
                            w_val = w_vec[idx]
                            if w_val != 0:
                                col_in_block = idx // si
                                row_in_block = idx % si
                                global_col = co + col_in_block
                                global_row = ro + row_in_block
                                w_col[global_col * n + global_row] = w_val
                        
                        U_list.append(u_col)
                        V_list.append(v_col)
                        W_list.append(w_col)
                        r_idx += 1
                else:
                    for i in range(si):
                        for k in range(sk):
                            for j in range(sj):
                                u_col = np.zeros(n * m)
                                v_col = np.zeros(m * p)
                                w_col = np.zeros(p * n)
                                
                                global_row = row_offsets[bi] + i
                                global_mid = mid_offsets[bk] + k
                                global_col = col_offsets[bj] + j
                                
                                u_col[global_row * m + global_mid] = 1.0
                                v_col[global_mid * p + global_col] = 1.0
                                w_col[global_col * n + global_row] = 1.0
                                
                                U_list.append(u_col)
                                V_list.append(v_col)
                                W_list.append(w_col)
                                r_idx += 1
    
    U_block = np.column_stack(U_list) if U_list else np.zeros((n*m, 0))
    V_block = np.column_stack(V_list) if V_list else np.zeros((m*p, 0))
    W_block = np.column_stack(W_list) if W_list else np.zeros((p*n, 0))
    
    # Verify the block decomposition
    def verify_decomp(U_mat, V_mat, W_mat):
        """Verify that the tensor decomposition correctly computes matrix multiplication."""
        np.random.seed(42)
        A = np.random.randint(-3, 3, (n, m)).astype(float)
        B = np.random.randint(-3, 3, (m, p)).astype(float)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        rank = U_mat.shape[1]
        for s in range(rank):
            a_part = np.zeros(n)
            b_part = np.zeros(p)
            for i in range(n):
                for k in range(m):
                    a_part[i] += A[i, k] * U_mat[i*m+k, s]
            for j in range(p):
                for k in range(m):
                    b_part[j] += B[k, j] * V_mat[k*p+j, s]
            for i in range(n):
                for j in range(p):
                    C_computed[i, j] += W_mat[j*n+i, s] * a_part[i] * b_part[j]
        
        return np.allclose(C_computed, C_expected, atol=1e-10)
    
    if verify_decomp(U_block, V_block, W_block):
        best_U, best_V, best_W = U_block, V_block, W_block
        best_rank = U_block.shape[1]
        print(f"Block decomposition verified with rank {best_rank}")
    else:
        print("Block decomposition verification failed, using naive")
    
    # Check time
    if time.time() > deadline:
        return {'U': best_U, 'V': best_V, 'W': best_W}
    
    return {'U': best_U, 'V': best_V, 'W': best_W}
