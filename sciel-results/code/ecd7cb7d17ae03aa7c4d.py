import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 3, 4, 8
    
    # Build target tensor T[i*m+j, j*p+k, k*n+i] = 1
    target = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Build rank-88 decomposition
    # Part 1: schoolbook for i=0 (rank 32)
    # Part 2: Strassen for i=1,2 in 2x2 blocks (rank 56)
    
    rank0 = 88
    U = np.zeros((n*m, rank0))
    V = np.zeros((m*p, rank0))
    W = np.zeros((p*n, rank0))
    
    col = 0
    # Part 1: i=0, schoolbook
    for j in range(m):
        for k in range(p):
            U[j, col] = 1
            V[j*p+k, col] = 1
            W[k*n, col] = 1
            col += 1
    
    # Part 2: Strassen for i=1,2
    # Strassen 2x2: 7 products
    # We split j into 2 blocks of 2, k into 4 blocks of 2
    # For each (j_block, k_block), use Strassen on 2x2 submatrix
    
    s_U = [
        [1, 0, 0, 1], 
        [0, 0, 1, 1], 
        [1, 0, 0, 0], 
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [-1, 0, 1, 0], 
        [0, 1, 0, -1]  
    ]
    s_V = [
        [1, 0, 0, 1], 
        [1, 0, 0, 0], 
        [0, 1, 0, -1], 
        [-1, 0, 1, 0], 
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [0, 0, 1, 1]  
    ]
    s_W = [
        [1, 0, 0, 1], 
        [0, 0, 1, -1], 
        [0, 1, 0, 1], 
        [1, 0, 1, 0], 
        [-1, 1, 0, 0], 
        [0, 0, 0, 1], 
        [1, 0, 0, 0]  
    ]
    
    for j_block in range(2):
        j_start = j_block * 2
        for k_block in range(4):
            k_start = k_block * 2
            for t in range(7):
                # s_U, s_V, s_W have 4 elements each representing (li, lj) or (lj, lk) or (li, lk)
                for l in range(4):
                    if s_U[t][l] != 0:
                        li = l // 2
                        lj = l % 2
                        gi = 1 + li
                        gj = j_start + lj
                        U[gi*m + gj, col] += s_U[t][l]
                    
                    if s_V[t][l] != 0:
                        lj = l // 2
                        lk = l % 2
                        gj = j_start + lj
                        gk = k_start + lk
                        V[gj*p + gk, col] += s_V[t][l]
                    
                    if s_W[t][l] != 0:
                        li = l // 2
                        lk = l % 2
                        gi = 1 + li
                        gk = k_start + lk
                        W[gk*n + gi, col] += s_W[t][l]
                col += 1
    
    # Verify
    def verify(U_mat, V_mat, W_mat):
        recon = np.zeros((n*m, m*p, p*n), dtype=float)
        for r in range(U_mat.shape[1]):
            recon += np.einsum('a,b,c->abc', U_mat[:,r], V_mat[:,r], W_mat[:,r])
        return np.allclose(recon, target, atol=1e-8)
    
    if not verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = rank0
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Trilinear dependency search:
    # Flatten each rank-1 tensor into a vector
    # T_r = U_r ⊗ V_r ⊗ W_r, flatten to (n*m*m*p*p*n,) vector
    # Check if any T_r is in the span of others
    
    def flatten_tensor(U_mat, V_mat, W_mat, r):
        return np.kron(np.kron(U_mat[:,r], V_mat[:,r]), W_mat[:,r])
    
    # Try to remove columns one at a time, greedily
    U_cur, V_cur, W_cur = U.copy(), V.copy(), W.copy()
    current_rank = rank0
    
    max_attempts = 50
    attempts = 0
    
    while current_rank > 50 and attempts < max_attempts and time.time() < deadline - 20:
        attempts += 1
        
        # Build flattened matrix of all current rank-1 tensors
        num_cols = current_rank
        flat_dim = n*m * m*p * p*n
        
        # Instead of full flatten (too large: 12*32*24=9216, times 88 columns),
        # use a smarter approach: for each candidate column r, check if 
        # U_r, V_r, W_r can be "absorbed" into other columns
        
        # Method: Try to find α, β, γ such that for all (a,b,c):
        # U[a,r]*V[b,r]*W[c,r] = sum_s α_s*β_s*γ_s * U[a,s]*V[b,s]*W[c,s]
        # This is hard in general. 
        # 
        # Simpler: check linear dependence of the set of vectors {U_r ⊗ V_r} 
        # against {U_s ⊗ V_s} with W as a separate factor.
        #
        # Alternative: use the fact that if we fix W_r, we need 
        # U_r ⊗ V_r ⊗ W_r in span of {U_s ⊗ V_s ⊗ W_s}
        #
        # Practical approach: randomly project the 3D tensor to 2D and check rank
        # Or: use iterative removal with numerical linear algebra
        
        # Try each column as a candidate for removal
        removed = False
        for r in range(current_rank):
            if time.time() > deadline - 20:
                break
            
            # Build the "signature" vectors for all other columns
            # For column r to be removable, we need coefficients c_s such that:
            # For all a,b: U[a,r]*V[b,r] = sum_s c_s * U[a,s]*V[b,s] * (W[c,r]/W[c,s])
            # This is inconsistent unless W ratios are constant.
            
            # Better approach: check if the 4D tensor formed by 
            # (U_r, V_r) is in the span when weighted by W_r
            #
            # Use SVD on the matrix M where M[a,b,s] = U[a,s]*V[b,s]
            # and check if M[:, :, r] is in the span of M[:, :, s≠r]
            
            # Construct matrix: rows = (a,b) pairs, columns = s
            # M[(a,b), s] = U[a,s] * V[b,s]
            ab_dim = n*m * m*p
            M = np.zeros((ab_dim, current_rank))
            for a in range(n*m):
                for b in range(m*p):
                    for s in range(current_rank):
                        M[a*m*p + b, s] = U_cur[a, s] * V_cur[b, s]
            
            # Check if column r of M (weighted by W) is in span of others
            # Actually, we need: U_r ⊗ V_r ⊗ W_r = sum_s c_s (U_s ⊗ V_s ⊗ W_s)
            # Divide by W_r (where nonzero): 
            # (U_r[a]/?) ... this gets messy.
            
            # Simpler numerical check: form the full tensor matrix
            # T[s, (a,b,c)] = U[a,s]*V[b,s]*W[c,s]
            # Check rank of [T[0:r], T[r+1:]] vs [T]
            
            # Build full tensor matrix (current_rank x ab_dim * p*n)
            # This is 88 x 9216 = manageable
            full_dim = ab_dim * p*n
            T_mat = np.zeros((current_rank, full_dim))
            idx = 0
            for a in range(n*m):
                for b in range(m*p):
                    for c in range(p*n):
                        for s in range(current_rank):
                            T_mat[s, idx] = U_cur[a,s] * V_cur[b,s] * W_cur[c,s]
                        idx += 1
            
            # Check if row r is in the span of other rows
            others = np.delete(T_mat, r, axis=0)
            
            # Use SVD to check if row r is in the column space of others
            # Transpose: we want to know if T_mat[r] is in span of {T_mat[s] for s≠r}
            # This is equivalent to: does adding row r increase the rank?
            
            # Compute rank of others and rank of all
            # Use np.linalg.matrix_rank
            rank_others = np.linalg.matrix_rank(others, tol=1e-10)
            rank_all = np.linalg.matrix_rank(T_mat, tol=1e-10)
            
            if rank_all == rank_others:
                # Row r is in the span - try to remove it
                new_U = np.delete(U_cur, r, axis=1)
                new_V = np.delete(V_cur, r, axis=1)
                new_W = np.delete(W_cur, r, axis=1)
                
                if verify(new_U, new_V, new_W):
                    U_cur, V_cur, W_cur = new_U, new_V, new_W
                    current_rank -= 1
                    if current_rank < best_rank:
                        best_rank = current_rank
                        best_U, best_V, best_W = U_cur.copy(), V_cur.copy(), W_cur.copy()
                    removed = True
                    break
        
        if not removed:
            break
    
    if best_rank < rank0:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
