import numpy as np
from itertools import product

def construct():
    n, m, p = 3, 4, 4
    num_a = n * m  # 12
    num_b = m * p  # 16
    num_c = p * n  # 12
    
    # Build the tensor: T[a,b,c] = 1 iff a = i*m+k, b = k*p+j, c = j*n+i
    # i in [0,n), j in [0,p), k in [0,m)
    tensor = np.zeros((num_a, num_b, num_c))
    for i in range(n):
        for j in range(p):
            for k in range(m):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                tensor[a, b, c] = 1.0
    
    # Start with a high-rank decomposition and iteratively prune
    # Initial rank: use 36 terms
    R_start = 36
    
    # Structured initialization: create terms that approximately cover the tensor
    # Each term r has U[:,r], V[:,r], W[:,r]
    # Initialize with random sparse vectors
    
    np.random.seed(12345)
    
    def random_sparse_vec(size, nnz, scale=1.0):
        v = np.zeros(size)
        idx = np.random.choice(size, nnz, replace=False)
        v[idx] = np.random.choice([-1, 1], nnz) * scale
        return v
    
    # Create initial U, V, W
    U = np.zeros((num_a, R_start))
    V = np.zeros((num_b, R_start))
    W = np.zeros((num_c, R_start))
    
    for r in range(R_start):
        nnz_u = np.random.choice([2, 3, 4], p=[0.3, 0.4, 0.3])
        nnz_v = np.random.choice([2, 3, 4], p=[0.3, 0.4, 0.3])
        nnz_w = np.random.choice([2, 3, 4], p=[0.3, 0.4, 0.3])
        U[:, r] = random_sparse_vec(num_a, nnz_u)
        V[:, r] = random_sparse_vec(num_b, nnz_v)
        W[:, r] = random_sparse_vec(num_c, nnz_w)
    
    # Compute decomposition error: sum_r U[:,r] outer V[:,r] outer W[:,r] - tensor
    def compute_decomp(U_mat, V_mat, W_mat):
        result = np.zeros_like(tensor)
        for r in range(U_mat.shape[1]):
            result += np.outer(np.outer(U_mat[:, r], V_mat[:, r]), W_mat[:, r])
        return result
    
    def compute_error(U_mat, V_mat, W_mat):
        decomp = compute_decomp(U_mat, V_mat, W_mat)
        return np.sum((decomp - tensor) ** 2)
    
    # ALS optimization
    def als_step(U_mat, V_mat, W_mat, n_iters=200, lr=0.01):
        """Alternating Least Squares to minimize ||sum U_r V_r W_r - tensor||^2"""
        R = U_mat.shape[1]
        U_cur = U_mat.copy()
        V_cur = V_mat.copy()
        W_cur = W_mat.copy()
        
        for it in range(n_iters):
            # Update U
            for r in range(R):
                # Residual for term r
                resid = tensor - compute_decomp(U_cur, V_cur, W_cur) + np.outer(np.outer(U_cur[:, r], V_cur[:, r]), W_cur[:, r])
                # Optimize U[:, r] to minimize ||resid - U[:,r] outer V[:,r] outer W[:,r]||^2
                # = argmin_u ||tensor - sum_s U_s V_s W_s + u V_r W_r||^2
                # Gradient: -2 * sum_{a,b} (tensor[a,b,:] - ...)[a,b] * V[b,r] * W[?,r]
                # Simpler: use gradient descent
                grad = np.zeros(num_a)
                for a in range(num_a):
                    for b in range(num_b):
                        for c in range(num_c):
                            diff = tensor[a, b, c] - np.sum(U_cur[:, s] * V_cur[b, s] * W_cur[c, s] for s in range(R))
                            grad[a] += 2 * diff * V_cur[b, r] * W_cur[c, r]
                U_cur[a, r] -= lr * grad[a] if False else 0  # placeholder
                # Actually, let's use a simpler gradient approach
                # The error is E = sum_{a,b,c} (T[a,b,c] - sum_r U[a,r]V[b,r]W[c,r])^2
                # dE/dU[a,r] = -2 * sum_{b,c} (T[a,b,c] - sum_s U[a,s]V[b,s]W[c,s]) * V[b,r] * W[c,r]
                diff_ac = tensor[a, :, :] - np.einsum('rs,bs,cs->bc', U_cur[:, r], V_cur[:, r], W_cur[:, r]) if False else None
                # This is getting complex. Let me use a different approach.
            break  # Skip full ALS for now
        
        return U_cur, V_cur, W_cur
    
    # Better approach: use a greedy term selection with exact verification
    # Start with empty decomposition, add terms one by one
    
    def verify_decomposition(U_list, V_list, W_list):
        """Verify that the list of terms exactly decomposes the tensor."""
        if not U_list:
            return False
        R = len(U_list)
        # Check all (a,b,c) triples
        decomp = np.zeros((num_a, num_b, num_c))
        for r in range(R):
            decomp += np.outer(np.outer(U_list[r], V_list[r]), W_list[r])
        # Check against tensor
        if not np.allclose(decomp, tensor, atol=1e-10):
            return False
        return True
    
    # Build a valid decomposition by adding terms
    # Start with the trivial decomposition (one term per tensor entry)
    # Then try to merge/prune
    
    # Trivial: for each (i,j,k), add term with U[i*m+k]=1, V[k*p+j]=1, W[j*n+i]=1
    U_terms = []
    V_terms = []
    W_terms = []
    
    for i in range(n):
        for j in range(p):
            for k in range(m):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                u = np.zeros(num_a)
                u[a] = 1.0
                v = np.zeros(num_b)
                v[b] = 1.0
                w = np.zeros(num_c)
                w[c] = 1.0
                U_terms.append(u)
                V_terms.append(v)
                W_terms.append(w)
    
    # Now we have 48 terms. Try to reduce by merging.
    # Two terms (u1,v1,w1) and (u2,v2,w2) can be merged if we can find
    # u, v, w such that u outer v outer w = u1 outer v1 outer w1 + u2 outer v2 outer w2
    
    # This is generally not possible, but we can try to find linear combinations.
    
    # Alternative: use a rank-reduction algorithm
    # For each pair of terms, try to replace them with a single term
    
    # Actually, let me try a completely different strategy:
    # Use the fact that we can factor the tensor using block structure
    
    # The tensor has a natural block structure. Let me exploit it.
    
    # C = A * B where A is 3x4, B is 4x4, C is 3x4
    # C[i,j] = sum_k A[i,k] * B[k,j]
    
    # I'll use a decomposition based on computing partial products and combining.
    
    # Strategy: compute 4 rank-1 contributions for each (i,j) pair, but share computations.
    
    # Let me try a systematic construction with rank 32.
    
    # For each column j of B, compute C[:,j] = A * B[:,j]
    # C[:,j] is a 3-vector. We need 3*4=12 multiplications for this column.
    # But we can share across columns.
    
    # Let me use the following: for each k, compute the outer product A[:,k] outer B[k,:]
    # This gives a 3x4 matrix. Sum over k.
    # Each outer product can be represented as 3*4=12 rank-1 terms in the tensor,
    # but actually it's just one "multiplication" A[i,k]*B[k,j] for each (i,j).
    
    # In the tensor decomposition framework, each scalar multiplication A[i,k]*B[k,j]
    # contributes to C[i,j] with weight 1.
    
    # So the trivial decomposition has 3*4*4 = 48 terms.
    
    # To reduce, I need to use linear combinations in U, V, or W.
    
    # Let me try: for fixed k, the contribution is sum_{i,j} A[i,k] * B[k,j] * delta_{C[i,j]}
    # This is a rank-1 term in the (A_k, B_k) space, but rank 12 in the full space.
    
    # I'll use a block diagonal approach with Strassen-like savings.
    
    # Split k into two groups: k=0,1 and k=2,3
    # For each group, use a 3x2x4 sub-decomposition
    
    # For 3x2x4: C = A_sub * B_sub where A_sub is 3x2, B_sub is 2x4
    # C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j]
    
    # This can be decomposed as:
    # For each i,j: two terms. Total 3*4*2 = 24 terms per group, 48 total.
    
    # To save, use Strassen on the 2x2 blocks within the 2x4.
    
    # Let me just implement a clean rank-40 decomposition using a different structure.
    
    # I'll use 5 blocks of 8 terms each, where each block handles a subset of (i,k,j) combinations.
    
    # Actually, let me try a rank-36 construction.
    
    R_target = 36
    U_mat = np.zeros((num_a, R_target))
    V_mat = np.zeros((num_b, R_target))
    W_mat = np.zeros((num_c, R_target))
    
    # For each k in range(4), for each "Strassen-like" combination
    # Use 9 terms per k-group (2 groups of 9 each = 18), plus 18 more for sharing
    
    # Let me try a different approach: use random search with verification
    
    best_rank = 48
    best_U = None
    best_V = None
    best_W = None
    
    # Try to find a decomposition with rank < 48
    for R in range(32, 48):
        found = False
        for trial in range(100):
            np.random.seed(trial + R * 1000)
            U_try = np.zeros((num_a, R))
            V_try = np.zeros((num_b, R))
            W_try = np.zeros((num_c, R))
            
            for r in range(R):
                nnz_u = np.random.randint(1, 4)
                nnz_v = np.random.randint(1, 4)
                nnz_w = np.random.randint(1, 4)
                u_idx = np.random.choice(num_a, nnz_u, replace=False)
                v_idx = np.random.choice(num_b, nnz_v, replace=False)
                w_idx = np.random.choice(num_c, nnz_w, replace=False)
                U_try[u_idx, r] = np.random.choice([-1, 1], nnz_u)
                V_try[v_idx, r] = np.random.choice([-1, 1], nnz_v)
                W_try[w_idx, r] = np.random.choice([-1, 1], nnz_w)
            
            # Verify
            decomp = np.zeros((num_a, num_b, num_c))
            for r in range(R):
                decomp += np.outer(np.outer(U_try[:, r], V_try[:, r]), W_try[:, r])
            
            if np.allclose(decomp, tensor, atol=1e-10):
                best_rank = R
                best_U = U_try
                best_V = V_try
                best_W = W_try
                found = True
                break
        
        if found:
            break
    
    if best_U is not None:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    
    # Fallback: return the trivial 48-term decomposition
    U_trivial = np.zeros((num_a, 48))
    V_trivial = np.zeros((num_b, 48))
    W_trivial = np.zeros((num_c, 48))
    
    r_idx = 0
    for i in range(n):
        for j in range(p):
            for k in range(m):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                U_trivial[a, r_idx] = 1.0
                V_trivial[b, r_idx] = 1.0
                W_trivial[c, r_idx] = 1.0
                r_idx += 1
    
    return {
        "U": U_trivial.tolist(),
        "V": V_trivial.tolist(),
        "W": W_trivial.tolist()
    }
