import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 (or potentially 48) decomposition for 4x4 matrix multiplication.
    
    1. Constructs the recursive Strassen rank-49 decomposition.
    2. Verifies it.
    3. Attempts to reduce rank to 48 by checking if any rank-1 term 
       is a linear combination of the others.
    4. Returns the minimal valid decomposition found within time limits.
    """
    t0 = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    n = 4
    rank_49 = 49
    
    # --- Step 1: Construct Rank-49 Decomposition ---
    
    # 2x2 Strassen decomposition (Rank 7)
    # Indices for 2x2: U/V/W are 4x7
    # U[i*2+k, r], V[k*2+j, r], W[j*2+i, r]
    U2 = np.zeros((4, 7), dtype=np.float64)
    V2 = np.zeros((4, 7), dtype=np.float64)
    W2 = np.zeros((4, 7), dtype=np.float64)
    
    # s0: (A00+A11)(B00+B11) -> C00+=s0, C11+=s0
    U2[0,0]=1; U2[3,0]=1; V2[0,0]=1; V2[3,0]=1; W2[0,0]=1; W2[3,0]=1
    # s1: (A10+A11)B00 -> C10+=s1, C11-=s1
    U2[2,1]=1; U2[3,1]=1; V2[0,1]=1; W2[1,1]=1; W2[3,1]=-1
    # s2: A00(B01-B11) -> C01+=s2, C11+=s2
    U2[0,2]=1; V2[1,2]=1; V2[3,2]=-1; W2[2,2]=1; W2[3,2]=1
    # s3: A11(B10-B00) -> C00+=s3, C10+=s3
    U2[3,3]=1; V2[2,3]=1; V2[0,3]=-1; W2[0,3]=1; W2[1,3]=1
    # s4: (A00+A01)B11 -> C00-=s4, C01+=s4
    U2[0,4]=1; U2[1,4]=1; V2[3,4]=1; W2[0,4]=-1; W2[2,4]=1
    # s5: (A10-A00)(B00+B01) -> C11+=s5
    U2[2,5]=1; U2[0,5]=-1; V2[0,5]=1; V2[1,5]=1; W2[3,5]=1
    # s6: (A01-A11)(B10+B11) -> C00+=s6
    U2[1,6]=1; U2[3,6]=-1; V2[2,6]=1; V2[3,6]=1; W2[0,6]=1

    # Build 4x4 decomposition by Kronecker product logic
    # U4[i*4+k, r1*7+r2] = U2[i*2+k1, r1] * U2[k2*2+j? No.
    # The recursive Strassen treats 4x4 as 2x2 blocks of 2x2.
    # A = [[A00, A01], [A10, A11]] where Aij are 2x2.
    # The "Outer" Strassen applies to the 2x2 block structure.
    # The "Inner" Strassen applies to the 2x2 element structure.
    # This is equivalent to the tensor product of the 2x2 decomposition with itself.
    # U4 = U2 kron U2? 
    # Let's verify dimensions.
    # U2 is 4x7. U2 kron U2 is 16x49.
    # Indexing: U4[(i1*2+i2)*4 + (k1*2+k2), r1*7+r2] = U2[i1*2+k1, r1] * U2[i2*2+k2, r2]
    # Wait, the standard Kronecker product of matrices A (m x n) and B (p x q) is mp x nq.
    # Here we want to map the 4x4 indices.
    # Let's just use the explicit Kronecker product of the U2, V2, W2 matrices.
    
    U49 = np.kron(U2, U2)
    V49 = np.kron(V2, V2)
    W49 = np.kron(W2, W2)
    
    # Verify the Kronecker product ordering matches the required index convention.
    # Required: U[i*4+k, r]. 
    # np.kron(U2, U2) with U2 indexed [row, col] where row = i*2+k.
    # The resulting row index is i1*4 + i2? No.
    # np.kron(A, B) row index is i_A * rows_B + i_B.
    # If A=U2 (4x7) and B=U2 (4x7), result is 16x49.
    # Row index = i_A * 4 + i_B.
    # i_A corresponds to the "outer" 2x2 index (i1, k1). i_A = i1*2 + k1.
    # i_B corresponds to the "inner" 2x2 index (i2, k2). i_B = i2*2 + k2.
    # So Row = (i1*2 + k1)*4 + (i2*2 + k2) = i1*8 + k1*4 + i2*2 + k2.
    # Required Row = i*4 + k.
    # Here i = i1*2 + i2? and k = k1*2 + k2?
    # If we map global i to (i1, i2) and global k to (k1, k2) such that
    # i = i1*2 + i2 and k = k1*2 + k2,
    # then i*4 + k = (i1*2 + i2)*4 + (k1*2 + k2) = i1*8 + i2*4 + k1*2 + k2.
    # This does NOT match i1*8 + k1*4 + i2*2 + k2.
    # The Kronecker product np.kron(U2, U2) uses the "block" ordering where the first index varies slowest?
    # Actually, np.kron(A, B) creates a matrix where block A[i,j] is scaled by B.
    # Row r in A, Col c in A -> Block at r, c is B.
    # Global Row = r * rows(B) + r_B.
    # So Global Row = (i1*2+k1)*4 + (i2*2+k2).
    # We need Global Row = i*4 + k.
    # Let's define the mapping of logical indices.
    # Let global i = 2*i1 + i2.
    # Let global k = 2*k1 + k2.
    # Then i*4 + k = (2*i1 + i2)*4 + (2*k1 + k2) = 8*i1 + 4*i2 + 2*k1 + k2.
    # The Kronecker index is 8*i1 + 4*k1 + 2*i2 + k2.
    # These are different.
    
    # We need to permute the rows/columns of the Kronecker product to match the index convention.
    # Or, simpler: just construct it directly using the loops as in the original code, which is known to work.
    # The original code used explicit loops. Let's stick to that for correctness, but optimize if needed.
    # The loops are O(49 * 16 * 16) which is very small.
    
    U49 = np.zeros((16, 49), dtype=np.float64)
    V49 = np.zeros((16, 49), dtype=np.float64)
    W49 = np.zeros((16, 49), dtype=np.float64)
    
    # Outer Strassen terms (for the 2x2 block structure)
    # Same as U2/V2/W2 but interpreted as block coefficients.
    # We can just reuse the U2/V2/W2 logic for the "outer" layer and U2/V2/W2 for the "inner" layer.
    
    for r_outer in range(7):
        for r_inner in range(7):
            r_total = r_outer * 7 + r_inner
            
            # Extract coefficients for this rank-1 term
            # U part: u_outer[i_block, k_block] * u_inner[i_elem, k_elem]
            # Global i = i_block*2 + i_elem
            # Global k = k_block*2 + k_elem
            # Index = i*4 + k
            
            for i_block in range(2):
                for k_block in range(2):
                    u_outer_val = U2[i_block*2 + k_block, r_outer] # Wait, U2 index is i*2+k?
                    # In U2 definition: U2[i*2+k, r]. 
                    # So for block (i_block, k_block), the index in U2 is i_block*2 + k_block.
                    if u_outer_val == 0:
                        continue
                    for i_elem in range(2):
                        for k_elem in range(2):
                            u_inner_val = U2[i_elem*2 + k_elem, r_inner]
                            if u_inner_val == 0:
                                continue
                            u_val = u_outer_val * u_inner_val
                            i_global = i_block * 2 + i_elem
                            k_global = k_block * 2 + k_elem
                            u_idx = i_global * 4 + k_global
                            U49[u_idx, r_total] = u_val
            
            # V part: v_outer[k_block, j_block] * v_inner[k_elem, j_elem]
            # Global k = k_block*2 + k_elem
            # Global j = j_block*2 + j_elem
            # Index = k*4 + j
            for k_block in range(2):
                for j_block in range(2):
                    v_outer_val = V2[k_block*2 + j_block, r_outer]
                    if v_outer_val == 0:
                        continue
                    for k_elem in range(2):
                        for j_elem in range(2):
                            v_inner_val = V2[k_elem*2 + j_elem, r_inner]
                            if v_inner_val == 0:
                                continue
                            v_val = v_outer_val * v_inner_val
                            k_global = k_block * 2 + k_elem
                            j_global = j_block * 2 + j_elem
                            v_idx = k_global * 4 + j_global
                            V49[v_idx, r_total] = v_val
            
            # W part: w_outer[j_block, i_block] * w_inner[j_elem, i_elem]
            # Note: W index convention is j*n + i.
            # In W2 definition: W2[j*2+i, r].
            # So for block (j_block, i_block), index is j_block*2 + i_block.
            for j_block in range(2):
                for i_block in range(2):
                    w_outer_val = W2[j_block*2 + i_block, r_outer]
                    if w_outer_val == 0:
                        continue
                    for j_elem in range(2):
                        for i_elem in range(2):
                            w_inner_val = W2[j_elem*2 + i_elem, r_inner]
                            if w_inner_val == 0:
                                continue
                            w_val = w_outer_val * w_inner_val
                            j_global = j_block * 2 + j_elem
                            i_global = i_block * 2 + i_elem
                            w_idx = j_global * 4 + i_global
                            W49[w_idx, r_total] = w_val

    # Verify Rank 49
    # C[i,j] = sum_k A[i,k] B[k,j]
    # Tensor T[i,k,j] = 1 if i,k,j are valid? No, T is the matmul tensor.
    # T[i,k,j] = 1 if we are computing C[i,j] from A[i,k] and B[k,j].
    # The decomposition is sum_r U[i,k,r] V[k,j,r] W[j,i,r] = 1 for all i,j,k?
    # Actually, the identity is: sum_k A[i,k]B[k,j] = sum_r (sum_k U[i,k,r]A[i,k]) (sum_j V[k,j,r]B[k,j]) W[j,i,r] ?
    # No, the standard form is:
    # C = A B
    # C[i,j] = sum_k A[i,k] B[k,j]
    # Decomposition: sum_r u_r(i,k) v_r(k,j) w_r(j,i)
    # Where u_r, v_r, w_r are the columns of U, V, W.
    # The tensor T_matmul has entries T[i,k,j] = 1.
    # The rank-1 term is u_r(i,k) * v_r(k,j) * w_r(j,i).
    # We need sum_r T_r[i,k,j] = 1 for all i,j,k.
    
    # Let's verify one entry to be sure.
    # i=0, k=0, j=0.
    # We need sum_r U[0,0,r] * V[0,0,r] * W[0,0,r] == 1.
    # W index: j*4 + i = 0*4 + 0 = 0.
    
    err = 0.0
    for i in range(4):
        for k in range(4):
            for j in range(4):
                s = 0.0
                u_val = U49[i*4+k, :]
                v_val = V49[k*4+j, :]
                w_val = W49[j*4+i, :]
                s = np.dot(np.dot(u_val, v_val), w_val)
                err += abs(s - 1.0)
                
    if err > 1e-9:
        # Fallback to schoolbook if construction failed (should not happen)
        rank_f = 64
        U_f = np.zeros((16, rank_f), dtype=np.float64)
        V_f = np.zeros((16, rank_f), dtype=np.float64)
        W_f = np.zeros((16, rank_f), dtype=np.float64)
        idx = 0
        for i in range(4):
            for k in range(4):
                for j in range(4):
                    U_f[i*4+k, idx] = 1.0
                    V_f[k*4+j, idx] = 1.0
                    W_f[j*4+i, idx] = 1.0
                    idx += 1
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}

    # --- Step 2: Attempt Rank Reduction to 48 ---
    
    # Check if any term is in the span of the others.
    # Flatten each rank-1 tensor to a 64-dim vector.
    # T_r = U[:,r] (16) x V[:,r] (16) x W[:,r] (16) -> 64 dim.
    # We need to check if vec(T_elim) is in the column space of [vec(T_0), ..., vec(T_48)] excluding elim.
    
    tensors = []
    for r in range(49):
        # Flatten U[:,r], V[:,r], W[:,r] into a 64-vector
        # Order: i, k, j? 
        # U is indexed i*4+k. V is k*4+j. W is j*4+i.
        # The tensor T[i,k,j] = U[i,k,r] * V[k,j,r] * W[j,i,r].
        # We need a consistent flattening.
        # Let's create a 4x4x4 array and flatten it.
        T_r = np.zeros((4, 4, 4), dtype=np.float64)
        for i in range(4):
            for k in range(4):
                for j in range(4):
                    T_r[i, k, j] = U49[i*4+k, r] * V49[k*4+j, r] * W49[j*4+i, r]
        tensors.append(T_r.flatten())
    
    tensors = np.array(tensors) # Shape (49, 64)
    
    best_rank = 49
    best_U = U49
    best_V = V49
    best_W = W49
    
    # Try eliminating each term
    for elim in range(49):
        if time.time() - t0 > deadline - 10:
            break
            
        others = [r for r in range(49) if r != elim]
        M_others = tensors[others, :] # (48, 64)
        target = tensors[elim, :]     # (64,)
        
        # Check if target is in the column space of M_others.
        # M_others.T is (64, 48).
        # We want to solve M_others.T * x = target.
        # If a solution exists, target is in the span.
        
        # Use least squares to check residual
        try:
            # x is 48x1
            x, residuals, rank, sv = np.linalg.lstsq(M_others.T, target, rcond=None)
            # Check residual
            if residuals is not None and len(residuals) > 0:
                res = residuals[0]
            else:
                res = np.linalg.norm(M_others.T @ x - target)
            
            if res < 1e-9:
                # The term is redundant. 
                # We can construct a rank 48 decomposition.
                # T_total = sum_{r=0}^{48} T_r = T_elim + sum_{r != elim} T_r
                # T_elim = sum_{s != elim} c_s T_s  (where x contains c_s)
                # T_total = sum_{s != elim} (1 + c_s) T_s
                # This is a valid rank 48 decomposition using the SAME vectors U_s, V_s, W_s 
                # but scaled by (1+c_s).
                
                # Construct new U, V, W
                U48 = np.zeros((16, 48), dtype=np.float64)
                V48 = np.zeros((16, 48), dtype=np.float64)
                W48 = np.zeros((16, 48), dtype=np.float64)
                
                idx_new = 0
                for s in others:
                    scale = 1.0 + x[idx_new]
                    U48[:, idx_new] = U49[:, s] * scale
                    V48[:, idx_new] = V49[:, s] * scale
                    W48[:, idx_new] = W49[:, s] * scale
                    idx_new += 1
                
                # Verify the new decomposition
                err48 = 0.0
                for i in range(4):
                    for k in range(4):
                        for j in range(4):
                            s_val = 0.0
                            u_val = U48[i*4+k, :]
                            v_val = V48[k*4+j, :]
                            w_val = W48[j*4+i, :]
                            s_val = np.dot(np.dot(u_val, v_val), w_val)
                            err48 += abs(s_val - 1.0)
                            
                if err48 < 1e-9:
                    best_rank = 48
                    best_U = U48
                    best_V = V48
                    best_W = W48
                    break # Found a valid rank 48, no need to check others
                    
        except Exception as e:
            continue

    # Return the best found
    if best_rank == 48:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
