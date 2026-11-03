"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    """
    n, m, p = 3, 3, 7
    # Schoolbook rank is n*m*p = 63.
    # We can try to find a decomposition with lower rank.
    # For 3x3 by 3x3, the minimal rank is 23 (Strassen).
    # For 3x3 by 3x7, we need to think carefully.
    
    # Actually, let's think about this differently.
    # We're computing C = A @ B where A is 3x3 and B is 3x7.
    # The tensor has size 3*3 * 3*7 * 7*3 = 9 * 21 * 21
    # Wait, that's not right. The tensor T_{ijk} where i in [0,n), j in [0,m), k in [0,p)
    # represents the (i,j) entry of C = sum_k A_{ik} B_{kj}.
    
    # Actually, the bilinear complexity of multiplying an n x m matrix by an m x p matrix
    # is the rank of the tensor. For n=3, m=3, p=7, the schoolbook rank is 3*3*7 = 63.
    
    # Let me reconsider. The bilinear complexity of C = A*B where A is 3x3 and B is 3x7:
    # C_{i,j} = sum_{k=0}^{2} A_{i,k} * B_{k,j} for i in [0,3), j in [0,7)
    # This requires 3*3*7 = 63 scalar multiplications in schoolbook.
    
    # Can we do better? Strassen shows that 2x2 by 2x2 can be done in 7 mults.
    # For 3x3 by 3x3, the best known is 23 (Lott's algorithm or similar).
    # But 3x3 by 3x7 is different.
    
    # Let's try to use a tensor decomposition approach with random initialization
    # and see if we can find a lower rank decomposition.
    
    import time
    start_time = time.time()
    deadline = start_time + 120
    
    # We'll return the schoolbook decomposition as a fallback.
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    best_rank = rank
    best_result = {"U": U, "V": V, "W": W}
    
    # Try random tensor decomposition with lower ranks
    import random
    import numpy as np
    
    def random_decomposition(rank_to_try, num_iters, seed=42):
        rng = np.random.RandomState(seed)
        for _ in range(num_iters):
            U_rand = rng.randn(n * m, rank_to_try)
            V_rand = rng.randn(m * p, rank_to_try)
            W_rand = rng.randn(p * n, rank_to_try)
            
            # Check if the decomposition is exact
            # The (i, j, k) entry of the multiplication tensor should be:
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{k,0} * ... 
            # Wait, I need to be more careful.
            
            # For matrix multiplication C = A * B where A is n x m and B is m x p:
            # C[i, j] = sum_k A[i, k] * B[k, j]
            # The tensor T has entries T[i, j, k] = 1 if this corresponds to A[i,k]*B[k,j]
            # So T[i*m+j, k*n+i] for the U*V part... 
            # Actually, the standard form is:
            # C[i,j] = sum_c U[i*m+j, c] * (A*B decomposition)
            
            # Let me re-derive. We want:
            # For all i in [0,n), j in [0,m), k in [0,p):
            # The product A[i,k] * B[k,j] contributes to C[i,j]
            # So the tensor is T[i*m+k, j*p+k... no.
            
            # Standard bilinear form: 
            # (A * B)[i,j] = sum_{c} U[i*m+j, c] * sum_{k} (A[i,k] * something)
            # This is getting complicated. Let me just verify with the schoolbook.
            
            # Schoolbook: For each (i,j,k) triple, we have one term A[i,k]*B[k,j]
            # U[i*m+k... no, U[i*m+j... 
            # In the seed program: U[i*m+j], V[j*p+k], W[k*n+i]
            # So the decomposition is: for c corresponding to (i,j,k):
            # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
            # And the product U*V*W gives 1 at (i*m+j, j*p+k, k*n+i) for each term.
            
            # The multiplication tensor in this coordinate system:
            # T[i*m+j, j*p+k, k*n+i] = 1 for valid (i,j,k) where i<m (rows of A), j<m (cols of A = rows of B), k<p (cols of B)
            # Wait, that's not quite right either.
            
            # Let me re-examine: A is 3x3 (n=3 rows, m=3 cols), B is 3x7 (m=3 rows, p=7 cols)
            # C = A @ B is 3x7
            # C[i,j] = sum_{k=0}^{2} A[i,k] * B[k,j] where i in [0,3), k in [0,3), j in [0,7)
            
            # In the seed: i in range(n)=3, j in range(m)=3, k in range(p)=7
            # U[i*m+j] -> row index for A[i,j] part... but A only has 3 cols, so j<3 makes sense for A
            # V[j*p+k] -> B[j,k] where j<3 (rows of B) and k<7 (cols of B)
            # W[k*n+i] -> output position... C has 3 rows and 7 cols, so k should be col index of C, i should be row?
            # C[i,j] where i in [0,3), j in [0,7). W[k*n+i] with k in [0,7), i in [0,3) -> k*n+i = k*3+i
            # So W index = k*3 + i, which means the output is indexed as [col * n + row] = [j * 3 + i]
            # So W[j*3+i] = 1 corresponds to C[i,j].
            
            # So the tensor is: for (i, j_A, j_C) where i < m=3 (inner dim), j_A < m=3 (A cols), j_C < p=7 (B cols / C cols)
            # U[i_A * m + i_inner, c] where i_A < n=3, i_inner < m=3
            # V[i_inner * p + j_C, c] where i_inner < m=3, j_C < p=7
            # W[j_C * n + i_A, c] where j_C < p=7, i_A < n=3
            
            # And the tensor entry is T[U_idx, V_idx, W_idx] = 1 for each (i_A, i_inner, j_C) triple.
            # This is correct: the product U * V * W at those indices equals 1, and this corresponds to
            # A[i_A, i_inner] * B[i_inner, j_C] contributing to C[i_A, j_C].
            
            pass
    
    # Actually, let's just try to search for a valid decomposition using a simple approach.
    # We'll try ranks from some minimum up to schoolbook and check validity.
    
    def check_decomposition(U_arr, V_arr, W_arr, n=3, m=3, p=7):
        """Check if the decomposition is exact."""
        # The (i*m+j, j*p+k, k*n+i) entry should be 1 if i<m, j<m, k<p (valid indices)
        # Wait, i is row of A (0..n-1), j is col of A / row of B (0..m-1), k is col of B (0..p-1)
        # U index: i*m + j, range [0, n*m)
        # V index: j*p + k, range [0, m*p)
        # W index: k*n + i, range [0, p*n)
        
        rank = U_arr.shape[1]
        # Build the tensor from decomposition
        # T[u, v, w] = sum_c U[u,c] * V[v,c] * W[w,c]
        
        # We can compute this efficiently:
        # For each c, add outer product of U[:,c], V[:,c], W[:,c]
        # But this might be slow. Let's just check specific entries.
        
        # Actually, for exactness, we need T[u,v,w] = 1 for all valid (i,j,k) and 0 for others?
        # No, that's not right. The tensor T represents the multiplication map.
        # T[u, v, w] should be 1 iff (u, v, w) corresponds to a valid (i, j, k) triple.
        
        # Let's compute T and check.
        T = np.zeros((n*m, m*p, p*n))
        for c in range(rank):
            T += np.einsum('a,b,c->abc', U_arr[:,c], V_arr[:,c], W_arr[:,c])
        
        # Now check: for each i in [0,n), j in [0,m), k in [0,p):
        # T[i*m+j, j*p+k, k*n+i] should be 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(T[i*m+j, j*p+k, k*n+i] - 1) > 1e-6:
                        return False
        
        # And for invalid entries, T should be 0.
        # A valid (u,v,w) means: u = i*m+j, v = j'*p+k, w = k'*n+i' 
        # where the j in v must equal the j in u, and the i in w must equal the i in u, etc.
        # Actually, the tensor should be zero at (u,v,w) unless u = i*m+j, v = j*p+k, w = k*n+i
        # for some consistent (i,j,k). But "consistent" means the j in u matches the j in v,
        # and the i in u matches the i in w (with k in v matching k in w).
        
        # So for (u, v, w) to be a valid non-zero entry, we need:
        # u = i*m + j_a, v = j_b*p + k, w = k'*n + i'
        # with j_a == j_b, i == i', and k == k' (i.e., u, v, w all refer to the same (i,j,k) triple)
        
        # Let's check that T is zero at all invalid positions.
        valid_positions = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    valid_positions.add((i*m+j, j*p+k, k*n+i))
        
        # Check a few invalid positions
        # Actually, let's just verify the full tensor matches.
        expected_T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected_T[i*m+j, j*p+k, k*n+i] = 1.0
        
        if not np.allclose(T, expected_T, atol=1e-6):
            return False
        
        return True
    
    # Try to find a decomposition with rank < 63 using random search + verification
    # This is very hard in general, but let's try a few ranks.
    
    # Actually, for small problems, we can try to use known results or search.
    # For now, let's just return the schoolbook decomposition.
    
    if time.time() > deadline - 5:
        return best_result
    
    # Try some random decompositions with rank < 63
    ranks_to_try = range(21, 63)  # Try various ranks
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
        # Try a few random decompositions
        for trial in range(20):
            if time.time() > deadline - 10:
                break
            np.random.seed(r * 1000 + trial)
            U_r = np.random.randn(n * m, r) * 0.1
            V_r = np.random.randn(m * p, r) * 0.1
            W_r = np.random.randn(p * n, r) * 0.1
            
            if check_decomposition(U_r, V_r, W_r):
                # Found a valid decomposition with rank r!
                best_rank = r
                best_result = {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
                break
    
    return best_result
# EVOLVE-BLOCK-END
