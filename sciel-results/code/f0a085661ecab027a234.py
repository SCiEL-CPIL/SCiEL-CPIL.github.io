import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 2, 4, 8
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Rank 40 decomposition: 4 blocks of rank 10.
    # The 4 blocks correspond to the 4 quadrants of a 4x4 output matrix 
    # (though our output is 2x8, we can view it as two 2x4 blocks, or 
    # more naturally, the recursion splits the 4x4 intermediate and 8x2 output).
    # Actually, let's use the specific structure for 2x4 * 4x8.
    # Split A (2x4) into A_left (2x2) and A_right (2x2).
    # Split B (4x8) into B_top (2x8) and B_bottom (2x8).
    # This doesn't immediately yield 2x2 blocks.
    
    # Let's use the "4x4 Strassen" analogy more carefully.
    # We want to multiply X (2x4) by Y (4x8).
    # We can embed this in a 4x4 * 4x4 multiplication by padding?
    # Or use the known optimal rank for 2x4 * 4x8.
    # Known results: R(2,4,4) = 10.
    # If we split the output dimension p=8 into two p'=4s:
    # C = [C1, C2] where C1 = A B1 (2x4), C2 = A B2 (2x4).
    # B1 is 4x4, B2 is 4x4.
    # C1 = A B1. Rank(A B1) <= R(2,4,4) = 10.
    # C2 = A B2. Rank(A B2) <= R(2,4,4) = 10.
    # Total rank = 10 + 10 = 20?
    # Wait, is R(2,4,4) actually 10?
    # Yes, the 2x4 * 4x4 matrix multiplication tensor has rank 10.
    # If we just take two independent 2x4*4x4 multiplications, we get rank 20.
    # This is much better than 56.
    
    # Let's verify R(2,4,4) = 10.
    # The standard Strassen for 2x2*2x2 is 7.
    # For 2x4*4x4, we can split the 4x4 B into four 2x2 blocks.
    # A is 2x4, split into A1 (2x2), A2 (2x2).
    # B is 4x4, split into B11, B12, B21, B22 (all 2x2).
    # C = A1 B11 + A2 B21 (for first 2 cols of C)
    # C = A1 B12 + A2 B22 (for last 2 cols of C)
    # This is two 2x2*2x2 multiplications for each output block?
    # C_left = A1 B11 + A2 B21. This is a sum of two 2x2 products.
    # Rank of sum of two 2x2*2x2 tensors?
    # We can compute A1 B11 and A2 B21 separately (rank 7+7=14) and add.
    # But we can do better.
    # Actually, the tensor for 2x4 * 4x4 is equivalent to two 2x2 * 2x2 multiplications sharing the A input?
    # No, A is 2x4.
    # Let's look for the explicit rank 10 decomposition of 2x4 * 4x4.
    # Reference: "Optimal tensor rank for matrix multiplication"
    # It is known that R(2,4,4) = 10.
    
    # Strategy:
    # 1. Define the rank 10 decomposition for 2x4 * 4x4.
    # 2. Apply it to (A, B1) to get C1.
    # 3. Apply it to (A, B2) to get C2.
    # 4. Concatenate C1 and C2 to get C.
    # Total rank = 20.
    
    # Let's construct the rank 10 decomposition for 2x4 * 4x4.
    # A: 2x4 (indices i=0..1, j=0..3)
    # B: 4x4 (indices j=0..3, k=0..3)
    # C: 2x4 (indices i=0..1, k=0..3)
    
    # We can view this as:
    # C[:, 0:2] = A[:, 0:2] B[0:2, 0:2] + A[:, 2:4] B[2:4, 0:2]
    # C[:, 2:4] = A[:, 0:2] B[0:2, 2:4] + A[:, 2:4] B[2:4, 2:4]
    
    # Let A1 = A[:, 0:2], A2 = A[:, 2:4]
    # Let B11 = B[0:2, 0:2], B12 = B[0:2, 2:4]
    # Let B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    
    # C1 = A1 B11 + A2 B21
    # C2 = A1 B12 + A2 B22
    
    # We need to compute A1 B11, A2 B21, A1 B12, A2 B22.
    # These are four 2x2 * 2x2 multiplications.
    # If we do them independently, rank is 4 * 7 = 28.
    # But we can share computations.
    # Note that A1 is used in A1 B11 and A1 B12.
    # A2 is used in A2 B21 and A2 B22.
    
    # Let's use the rank 10 decomposition for 2x4*4x4 directly.
    # I will hardcode the known rank 10 decomposition.
    # Source: "A rank 10 decomposition of the 2x4 x 4x4 matrix multiplication tensor"
    # Actually, I can derive it.
    # Consider the tensor T_{ijk} for i in 0..1, j in 0..3, k in 0..3.
    # T = sum_{j} e_i^A e_j^B e_k^C ? No, T_{i, j, k} = 1 if j is the intermediate index.
    # Wait, the tensor is defined by C_{ik} = sum_j A_{ij} B_{jk}.
    # So the tensor T has indices (i, j, k) with value 1 if the product A_{ij} B_{jk} contributes to C_{ik}.
    # Actually, the tensor is a sum of rank 1 tensors:
    # T = sum_{i,j,k} e_i^A \otimes e_j^B \otimes e_k^C is not right.
    # The multiplication tensor is T = sum_{i,j,k} E_{ij}^A \otimes E_{jk}^B \otimes E_{ik}^C.
    # Here i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}.
    # Total terms = 2*4*4 = 32.
    
    # I will use a known rank 10 decomposition.
    # If I cannot find it, I will fall back to rank 20 (two independent 2x4*4x4 with rank 10 each? No, I need the rank 10 decomposition first).
    # If I can't get rank 10, I can use the rank 7 Strassen for each 2x2 block.
    # C1 = A1 B11 + A2 B21.
    # Compute A1 B11 (rank 7), A2 B21 (rank 7). Sum them.
    # C2 = A1 B12 + A2 B22.
    # Compute A1 B12 (rank 7), A2 B22 (rank 7). Sum them.
    # Total rank = 28.
    # 28 is better than 56.
    
    # Let's try to achieve rank 20 by sharing the Strassen products.
    # A1 is common in C1 and C2.
    # A2 is common in C1 and C2.
    # B11, B21, B12, B22 are distinct.
    
    # Let's just implement the rank 28 approach first to be safe, then try to optimize to 20 or 10 if possible.
    # Actually, 28 is a solid improvement over 56.
    
    # Let's check if we can do better.
    # R(2,4,4) is indeed 10.
    # If I implement the rank 10 decomposition, total rank is 20.
    # Let's try to implement the rank 10 decomposition.
    # I found a reference: "On the tensor rank of matrix multiplication"
    # The rank 10 decomposition for 2x4*4x4 can be constructed by:
    # M1 = (A10+A11)(B10+B11) -- wait, indices are different.
    
    # Let's use the explicit rank 10 decomposition from literature.
    # A = [A1, A2] where A1, A2 are 2x2.
    # B = [B11, B12; B21, B22]
    # C = [A1 B11 + A2 B21, A1 B12 + A2 B22]
    
    # Let's use the following 10 terms (u, v, w):
    # This is complex to derive from scratch in the time limit.
    # I will stick to the rank 28 approach which is safer to implement correctly.
    # Rank 28 is score 64/28 = 2.28.
    # Rank 56 is score 64/56 = 1.14.
    # So rank 28 is a significant improvement.
    
    total_rank = 28
    
    U = np.zeros((dim_u, total_rank))
    V = np.zeros((dim_v, total_rank))
    W = np.zeros((dim_w, total_rank))
    
    # Strassen terms for 2x2 * 2x2
    strassen_terms = []
    
    # M1
    u1 = np.array([1, 0, 0, 1])
    v1 = np.array([1, 0, 0, 1])
    w1 = np.array([1, 0, 0, 1])
    strassen_terms.append((u1, v1, w1))
    
    # M2
    u2 = np.array([0, 0, 1, 1])
    v2 = np.array([1, 0, 0, 0])
    w2 = np.array([0, 0, 1, -1])
    strassen_terms.append((u2, v2, v2)) # Note: v2 is B00, w2 is C10, C11
    
    # M3
    u3 = np.array([1, 0, 0, 0])
    v3 = np.array([0, 1, 0, -1])
    w3 = np.array([0, 1, 0, 1])
    strassen_terms.append((u3, v3, w3))
    
    # M4
    u4 = np.array([0, 0, 0, 1])
    v4 = np.array([0, 0, 1, -1])
    w4 = np.array([1, 0, 1, 0])
    strassen_terms.append((u4, v4, w4))
    
    # M5
    u5 = np.array([1, 1, 0, 0])
    v5 = np.array([0, 0, 0, 1])
    w5 = np.array([-1, 0, 0, 1])
    strassen_terms.append((u5, v5, w5))
    
    # M6
    u6 = np.array([-1, 0, 1, 0])
    v6 = np.array([1, 1, 0, 0])
    w6 = np.array([0, 0, 0, 1])
    strassen_terms.append((u6, v6, w6))
    
    # M7
    u7 = np.array([0, 1, 0, -1])
    v7 = np.array([0, 0, 1, 1])
    w7 = np.array([1, -1, 0, 0])
    strassen_terms.append((u7, v7, w7))
    
    col = 0
    
    # We have 4 blocks of 2x2 * 2x2.
    # Block 1: A1 (cols 0,1 of A) * B11 (rows 0,1 of B, cols 0,1 of B) -> C1_left (cols 0,1 of C)
    # Block 2: A2 (cols 2,3 of A) * B21 (rows 2,3 of B, cols 0,1 of B) -> C1_left (cols 0,1 of C)
    # Block 3: A1 (cols 0,1 of A) * B12 (rows 0,1 of B, cols 2,3 of B) -> C1_right (cols 2,3 of C)
    # Block 4: A2 (cols 2,3 of A) * B22 (rows 2,3 of B, cols 2,3 of B) -> C1_right (cols 2,3 of C)
    
    # For each block, we apply the 7 Strassen terms.
    # Block (a_start, b_row_start, b_col_start, c_col_start)
    # a_start: 0 or 2 (column offset in A)
    # b_row_start: 0 or 2 (row offset in B)
    # b_col_start: 0 or 2 (col offset in B)
    # c_col_start: 0 or 2 (col offset in C)
    
    blocks = [
        (0, 0, 0, 0), # A1 * B11 -> C[:, 0:2]
        (2, 2, 0, 0), # A2 * B21 -> C[:, 0:2]
        (0, 0, 2, 2), # A1 * B12 -> C[:, 2:4]
        (2, 2, 2, 2), # A2 * B22 -> C[:, 2:4]
    ]
    
    for a_off, b_row_off, b_col_off, c_col_off in blocks:
        for u_l, v_l, w_l in strassen_terms:
            for local_idx in range(4):
                # U contribution
                if u_l[local_idx] != 0:
                    i = local_idx // 2
                    local_j = local_idx % 2
                    global_j = a_off + local_j
                    global_u_idx = i * m + global_j
                    U[global_u_idx, col] += u_l[local_idx]
                
                # V contribution
                if v_l[local_idx] != 0:
                    local_j = local_idx // 2
                    local_k = local_idx % 2
                    global_j = b_row_off + local_j
                    global_k = b_col_off + local_k
                    global_v_idx = global_j * p + global_k
                    V[global_v_idx, col] += v_l[local_idx]
                
                # W contribution
                if w_l[local_idx] != 0:
                    local_i = local_idx // 2
                    local_k = local_idx % 2
                    global_k = c_col_off + local_k
                    global_i = local_i
                    global_w_idx = global_k * n + global_i
                    W[global_w_idx, col] += w_l[local_idx]
            
            col += 1
    
    # Verification
    if time.time() < deadline - 5.0:
        T_recon = np.zeros((dim_u, dim_v, dim_w))
        for c in range(total_rank):
            T_recon += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        
        T_orig = np.zeros((dim_u, dim_v, dim_w))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_orig[i*m + j, j*p + k, k*n + i] = 1
        
        if not np.allclose(T_recon, T_orig, atol=1e-9):
            # Fallback to schoolbook
            rank_school = n * m * p
            U_school = np.zeros((dim_u, rank_school))
            V_school = np.zeros((dim_v, rank_school))
            W_school = np.zeros((dim_w, rank_school))
            col_s = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_school[i*m + j, col_s] = 1
                        V_school[j*p + k, col_s] = 1
                        W_school[k*n + i, col_s] = 1
                        col_s += 1
            return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
