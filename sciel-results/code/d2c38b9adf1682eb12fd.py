import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    # n=2, m=5, p=6
    # Rank 54 decomposition
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i_base, j_base, k_base):
        nonlocal col
        # 7 columns for one 2x2 * 2x2 block
        
        # m0: (x00+x11)(y00+y11) -> Z00+=1, Z11+=1
        U[i_base * m + j_base][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # m1: (x10+x11)y00 -> Z10+=1, Z11-=1
        U[(i_base+1) * m + j_base][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] -= 1
        col += 1
        
        # m2: x00(y01-y11) -> Z01+=1, Z11+=1
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] -= 1
        W[(k_base+1) * n + i_base][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # m3: x11(y10-y00) -> Z00+=1, Z10+=1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[(j_base+1) * p + k_base][col] += 1
        V[j_base * p + k_base][col] -= 1
        W[k_base * n + i_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        col += 1
        
        # m4: (x00+x01)y11 -> Z00-=1, Z01+=1
        U[i_base * m + j_base][col] += 1
        U[i_base * m + (j_base+1)][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] -= 1
        W[(k_base+1) * n + i_base][col] += 1
        col += 1
        
        # m5: (x10-x00)(y00+y01) -> Z11+=1
        U[(i_base+1) * m + j_base][col] += 1
        U[i_base * m + j_base][col] -= 1
        V[j_base * p + k_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # m6: (x01-x11)(y10+y11) -> Z00+=1
        U[i_base * m + (j_base+1)][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] -= 1
        V[(j_base+1) * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1

    # Blocks for A_R (j=3,4) * B_R (j=3,4)
    # k splits: 0-1, 2-3, 4-5
    add_strassen_block(0, 3, 0)
    add_strassen_block(0, 3, 2)
    add_strassen_block(0, 3, 4)
    
    # Blocks for A_L1 (j=0,1) * B_L1 (j=0,1)
    # k splits: 0-1, 2-3, 4-5
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    
    # Outer product for A_L2 (j=2) * B_L2 (j=2)
    # A_L2 is 2x1, B_L2 is 1x6
    for k in range(p):
        # C[0, k] += A[0, 2] * B[2, k]
        U[0 * m + 2][col] = 1
        V[2 * p + k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
        
        # C[1, k] += A[1, 2] * B[2, k]
        U[1 * m + 2][col] = 1
        V[2 * p + k][col] = 1
        W[k * n + 1][col] = 1
        col += 1
    
    # Check time
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Build the full tensor once for fast verification
    # T[i,j,k] should equal 1 for all (i,j,k)
    # We'll compute the tensor from the decomposition and track which terms contribute
    
    def compute_tensor(U_mat, V_mat, W_mat, n, m, p):
        """Compute the full tensor from decomposition."""
        r = len(U_mat[0])
        tensor = [[0.0]*p for _ in range(m)]  # tensor[j][k] for each i
        # Actually let's use a flat structure
        T = [[[0.0]*p for _ in range(m)] for _ in range(n)]
        for r in range(r):
            for i in range(n):
                for j in range(m):
                    u_val = U_mat[i*m+j][r]
                    if u_val == 0:
                        continue
                    for k in range(p):
                        v_val = V_mat[j*p+k][r]
                        if v_val == 0:
                            continue
                        w_val = W_mat[k*n+i][r]
                        T[i][j][k] += u_val * v_val * w_val
        return T
    
    def verify_tensor(T, n, m, p):
        """Verify tensor equals 1 everywhere."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(T[i][j][k] - 1.0) > 1e-9:
                        return False
        return True
    
    # Compute initial tensor
    current_U = [row[:] for row in U]
    current_V = [row[:] for row in V]
    current_W = [row[:] for row in W]
    current_rank = rank
    
    # Verify initial decomposition
    init_T = compute_tensor(current_U, current_V, current_W, n, m, p)
    if not verify_tensor(init_T, n, m, p):
        # Decomposition is wrong, return as-is
        return {"U": current_U, "V": current_V, "W": current_W}
    
    best_U = [row[:] for row in current_U]
    best_V = [row[:] for row in current_V]
    best_W = [row[:] for row in current_W]
    best_rank = current_rank
    
    # Multiple passes of greedy elimination
    # After removing a term, other terms may become removable
    max_passes = 5
    for pass_num in range(max_passes):
        if time.time() > deadline - 15:  # Leave 15s buffer
            break
        
        removed_any = False
        # Try removing each active term
        for trial in range(current_rank):
            if time.time() > deadline - 15:
                break
            
            # Check if term 'trial' is active (has any non-zero entries)
            is_active = False
            for i in range(n * m):
                if current_U[i][trial] != 0:
                    is_active = True
                    break
            if not is_active:
                for j in range(m * p):
                    if current_V[j][trial] != 0:
                        is_active = True
                        break
            if not is_active:
                for k in range(p * n):
                    if current_W[k][trial] != 0:
                        is_active = True
                        break
            if not is_active:
                continue
            
            # Compute contribution of term 'trial'
            # delta_T[i][j][k] = U[i*m+j][trial] * V[j*p+k][trial] * W[k*n+i][trial]
            # New tensor = current_T - delta_T
            # Check if new tensor is still all 1s
            
            # Efficient check: for each (i,j,k), current_T[i][j][k] - delta_T[i][j][k] should be 1
            # i.e., delta_T[i][j][k] should equal current_T[i][j][k] - 1 = 0
            # So we need delta_T[i][j][k] == 0 for all (i,j,k) where current_T[i][j][k] != 1
            # But current_T should be all 1s, so we need delta_T[i][j][k] == 0 for all (i,j,k)
            # Wait, that's not right. Let me reconsider.
            
            # If we remove term r, the new tensor is T - delta_r
            # We need T - delta_r = 1 (all ones)
            # Since T = 1, we need delta_r = 0
            # So the term must have zero contribution everywhere!
            
            # Actually no. The tensor T is the sum of all terms. If T = 1 everywhere,
            # and we remove term r, we get T - delta_r. For this to equal 1,
            # we need delta_r = 0 everywhere.
            
            # But that means a term can only be removed if it contributes nothing,
            # which would mean it was already redundant.
            
            # Wait, I think I'm confusing myself. Let me re-examine.
            # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            # If T[i,j,k] = 1 for all (i,j,k), and we remove term r,
            # new_T[i,j,k] = T[i,j,k] - U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
            # For new_T to equal 1, we need U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 0 for all (i,j,k)
            
            # So a term can be removed iff its rank-1 tensor is identically zero.
            # That means the term is already "dead" - it doesn't contribute.
            
            # But that's not the right approach for rank reduction. The issue is that
            # the terms might be linearly dependent in a more subtle way.
            
            # Actually, for tensor rank decomposition, removing a term is valid if
            # the remaining terms can still produce the same tensor. This is NOT
            # simply checking if the removed term is zero.
            
            # Let me just do the full verification.
            
            test_U = [row[:] for row in current_U]
            test_V = [row[:] for row in current_V]
            test_W = [row[:] for row in current_W]
            
            # Zero out term 'trial'
            for i in range(n * m):
                test_U[i][trial] = 0
            for j in range(m * p):
                test_V[j][trial] = 0
            for k in range(p * n):
                test_W[k][trial] = 0
            
            # Verify by computing tensor
            test_T = compute_tensor(test_U, test_V, test_W, n, m, p)
            if verify_tensor(test_T, n, m, p):
                # Term can be removed
                current_U = test_U
                current_V = test_V
                current_W = test_W
                current_rank -= 1
                removed_any = True
                best_U = [row[:] for row in current_U]
                best_V = [row[:] for row in current_V]
                best_W = [row[:] for row in current_W]
                best_rank = current_rank
    
        if not removed_any:
            break  # No more terms can be removed
    
    # Return the best decomposition found
    # Trim to the actual rank
    final_U = []
    for row in best_U:
        final_U.append(row[:best_rank])
    final_V = []
    for row in best_V:
        final_V.append(row[:best_rank])
    final_W = []
    for row in best_W:
        final_W.append(row[:best_rank])
    
    return {"U": final_U, "V": final_V, "W": final_W}
# EVOLVE-BLOCK-END
