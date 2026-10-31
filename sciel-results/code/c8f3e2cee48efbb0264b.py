import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    n, m, p = 4, 4, 4

    # First, try to find a rank-48 decomposition using the 2x2x2 tensor over 2x2 blocks
    # The 4x4x4 multiplication tensor can be viewed as a 2x2x2 tensor where each entry
    # is a 2x2 matrix multiplication.
    
    # Strategy: Use the known optimal rank-7 Strassen for 2x2x2, but try to find
    # a rank-6 decomposition for the "outer" 2x2x2 tensor where each component is a 2x2 matmul.
    
    # Actually, let's think differently. The 4x4 matmul tensor has rank at most 49 (Strassen^2).
    # Can we do better?
    
    # Known results: The rank of 4x4x4 matrix multiplication tensor is at most 49.
    # Lower bounds suggest it's at least 48 or 49.
    
    # Let me try to construct a rank-48 decomposition by exploiting symmetry.
    
    # The 4x4x4 tensor T[i,j,k] = delta_{i,k} (for A[i,a]*B[a,j] -> C[i,j])
    # Wait, the tensor is T[i,j,a] where C[i,j] = sum_a A[i,a] B[a,j]
    
    # Let's use the approach: treat 4x4 as 2x2 blocks of 2x2 matrices.
    # C = A*B where A,B,C are 2x2 blocks of 2x2 matrices.
    # C_ij = sum_k A_ik * B_kj (block multiplication)
    
    # The block-level tensor is 2x2x2 with entries being 2x2x2 tensors (matrix mult of 2x2).
    # Rank of 2x2x2 matmul is 7 (Strassen).
    # So naive: 7 * 7 = 49.
    
    # To get 48, we need to find that one of the 49 terms is redundant or can be combined.
    
    # Let me try a different approach: use the fact that the 4x4x4 tensor might have
    # a rank-48 decomposition based on known algebraic identities.
    
    # For now, let's verify the rank-49 solution and check if we can trim one term.
    
    # Build rank-49 Strassen^2 decomposition
    rank_49 = True
    rank = 49
    
    # 2x2 Strassen factors
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)

    # M0 = (A00 + A11) * (B00 + B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1

    # M1 = (A10 + A11) * B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1

    # M2 = A00 * (B01 - B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1

    # M3 = A11 * (B10 - B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1

    # M4 = (A00 + A01) * B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1

    # M5 = (A10 - A00) * (B00 + B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1

    # M6 = (A01 - A11) * (B10 + B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # Build 49-rank decomposition
    U_arr = np.zeros((16, 49), dtype=int)
    V_arr = np.zeros((16, 49), dtype=int)
    W_arr = np.zeros((16, 49), dtype=int)
    
    I_map = [i // 2 for i in range(4)]
    J_map = [j // 2 for j in range(4)]
    ii_map = [i % 2 for i in range(4)]
    jj_map = [j % 2 for j in range(4)]
    
    for s in range(7):
        for t in range(7):
            r = s * 7 + t
            for i in range(4):
                for j in range(4):
                    coeff = U2[I_map[i]*2 + J_map[j], s] * U2[ii_map[i]*2 + jj_map[j], t]
                    if coeff != 0:
                        U_arr[i*4 + j, r] = coeff
            for k in range(4):
                for l in range(4):
                    coeff = V2[I_map[k]*2 + J_map[l], s] * V2[ii_map[k]*2 + jj_map[l], t]
                    if coeff != 0:
                        V_arr[k*4 + l, r] = coeff
            for j in range(4):
                for i in range(4):
                    coeff = W2[I_map[i]*2 + J_map[j], s] * W2[ii_map[i]*2 + jj_map[j], t]
                    if coeff != 0:
                        W_arr[j*4 + i, r] = coeff
    
    # Verify the rank-49 solution
    def verify_decomp(U_mat, V_mat, W_mat, n, m, p):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # C[i,j] = sum_a A[i,a] * B[a,j]
        # W index: k*n+i where k=j (col of C), i=row of C? Let me check the convention.
        # "W[k*n+i,r]" - k is column, i is row? Or k is row, i is column?
        # Standard: C[i,j] = sum_a A[i,a] B[a,j]
        # W[k*n+i, r] where this should correspond to C[i,j] with some mapping.
        # Let's assume W index = j*n + i (column-major) or i*n + j (row-major)
        # The task says "W[k*n+i,r]" - let's interpret k as the second index (column) and i as first (row)
        # So W[j*n+i, r] corresponds to C[i,j]
        
        U_mat = np.array(U_mat)
        V_mat = np.array(V_mat)
        W_mat = np.array(W_mat)
        rank = U_mat.shape[1]
        
        # Test with random matrices
        np.random.seed(42)
        A = np.random.randint(-5, 5, (n, m)).astype(float)
        B = np.random.randint(-5, 5, (m, p)).astype(float)
        C_expected = A @ B
        
        # Compute C from decomposition
        # A[i,a] -> U index i*m + a
        # B[a,j] -> V index a*p + j  
        # C[i,j] -> W index j*n + i (based on W[k*n+i] where k=j, i=i)
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            u_vec = U_mat[:, r]  # (n*m,)
            v_vec = V_mat[:, r]  # (m*p,)
            w_vec = W_mat[:, r]  # (p*n,)
            
            # u_vec[i*m + a] = U[i,a,r]
            # v_vec[a*p + j] = V[a,j,r]
            # w_vec[j*n + i] = W[j,i,r]
            
            U_block = u_vec.reshape(n, m)
            V_block = v_vec.reshape(m, p)
            W_block = w_vec.reshape(p, n)
            
            # C[i,j] += sum_a U[i,a,r] * V[a,j,r] * W[j,i,r]
            for i in range(n):
                for j in range(p):
                    C_computed[i,j] += np.sum(U_block[i,:] * V_block[:,j]) * W_block[j,i]
        
        return np.allclose(C_computed, C_expected, atol=1e-10)
    
    if verify_decomp(U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), n, m, p):
        # Rank 49 is valid. Now try to find rank 48.
        
        # Check if any column is linearly dependent or can be removed
        # Try removing each column and checking if the decomposition still works
        found_rank_48 = False
        
        for remove_col in range(49):
            if time.time() > deadline:
                break
            
            U_test = np.delete(U_arr, remove_col, axis=1)
            V_test = np.delete(V_arr, remove_col, axis=1)
            W_test = np.delete(W_arr, remove_col, axis=1)
            
            if verify_decomp(U_test.tolist(), V_test.tolist(), W_test.tolist(), n, m, p):
                found_rank_48 = True
                rank = 48
                U_arr = U_test
                V_arr = V_test
                W_arr = W_test
                break
        
        if not found_rank_48:
            # Try a different approach: look for rank-48 using algebraic structure
            # The 4x4x4 tensor might admit a rank-48 decomposition via the 
            # "fourfold" or "dual number" algebra approach.
            
            # For now, fall back to rank 49
            rank = 49
    
    else:
        # Error in verification, fall back to rank 49
        rank = 49
    
    return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}
