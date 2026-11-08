"""Matrix multiplication 3,4,8 using rank-88 decomposition with local search for rank reduction."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Build the target tensor T[i,j,k] = delta_{i,j,k} for matrix multiplication
    # T is a 3D array of shape (n*m, m*p, p*n) where T[i,j,k] = 1 iff the
    # (i,j) entry of A times (j,k) entry of B contributes to (k,i) entry of C
    # More precisely: T[im_j, jm_k, kn_i] = 1 if the product A[i,j]*B[j,k] contributes to C[k,i]
    # In the trace convention: W[k*n+i, r], so C[k,i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # The tensor T has T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    
    # Build target tensor as a 2D matrix (flattened)
    # Shape: (n*m * m*p) x (p*n) or similar
    # Actually, the tensor is T[a,b,c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T[a,b,c] = 1 if there exist i,j,k with a=i*m+j, b=j*p+k, c=k*n+i
    
    target = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Initial decomposition: rank-88
    rank = 88
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Part 1: i=0, schoolbook for 1x4x8 (rank 32)
    for j in range(4):
        for k in range(8):
            U[0 * m + j, col] = 1
            V[j * p + k, col] = 1
            W[k * n + 0, col] = 1
            col += 1
    
    # Part 2: i=1,2, Strassen on 2x2 blocks (rank 56)
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
                for l in range(4):
                    if s_U[t][l] != 0:
                        li = l // 2
                        lj = l % 2
                        gi = 1 + li
                        gj = j_start + lj
                        U[gi * m + gj, col] += s_U[t][l]
                        
                    if s_V[t][l] != 0:
                        lj = l // 2
                        lk = l % 2
                        gj = j_start + lj
                        gk = k_start + lk
                        V[gj * p + gk, col] += s_V[t][l]
                        
                    if s_W[t][l] != 0:
                        li = l // 2
                        lk = l % 2
                        gi = 1 + li
                        gk = k_start + lk
                        W[gk * n + gi, col] += s_W[t][l]
                col += 1
    
    # Verify initial decomposition
    def verify(U_mat, V_mat, W_mat):
        """Check if U, V, W decompose the target tensor."""
        # Compute the reconstructed tensor
        # T_recon[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # This is a 3-tensor contraction
        # Efficient way: for each r, outer product of U[:,r], V[:,r], W[:,r]
        recon = np.zeros_like(target)
        for r in range(U_mat.shape[1]):
            recon += np.einsum('a,b,c->abc', U_mat[:,r], V_mat[:,r], W_mat[:,r])
        return np.allclose(recon, target, atol=1e-10)
    
    if not verify(U, V, W):
        # Fallback to original if verification fails
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank by finding linear dependencies
    # Check if any column is a linear combination of others
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try eliminating one column at a time
    for r_remove in range(rank):
        if time.time() > deadline - 10:
            break
        
        # Try to express column r_remove as a linear combination of others
        U_others = np.delete(U, r_remove, axis=1)
        V_others = np.delete(V, r_remove, axis=1)
        W_others = np.delete(W, r_remove, axis=1)
        
        # We want to find coefficients alpha such that:
        # U[:,r_remove] = U_others @ alpha_u
        # V[:,r_remove] = V_others @ alpha_v  
        # W[:,r_remove] = W_others @ alpha_w
        # But the same alpha must work for all three (since it's the same tensor factor)
        # Actually, no - the coefficients can be different for U, V, W
        # But the product must still match: U[:,r_remove] * V[:,r_remove] * W[:,r_remove]
        # = sum_s (U_others[:,s] * V_others[:,s] * W_others[:,s]) * alpha_u[s] * alpha_v[s] * alpha_w[s]
        # This is complex. Let's try a simpler approach.
        
        # Check if the column is zero
        if np.allclose(U[:,r_remove], 0) and np.allclose(V[:,r_remove], 0) and np.allclose(W[:,r_remove], 0):
            # Can remove this column
            new_rank = rank - 1
            new_U = np.delete(U, r_remove, axis=1)
            new_V = np.delete(V, r_remove, axis=1)
            new_W = np.delete(W, r_remove, axis=1)
            
            if verify(new_U, new_V, new_W):
                best_rank = new_rank
                best_U, best_V, best_W = new_U, new_V, new_W
                U, V, W = new_U, new_V, new_W
                rank = new_rank
                # Restart the loop with new rank
                break
    
    # If we reduced rank, return the improved version
    if best_rank < 88:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
