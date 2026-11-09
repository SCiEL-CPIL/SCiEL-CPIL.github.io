import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n, m, p = 2, 4, 8
    dim_a = n * m  # 8
    dim_b = m * p  # 32
    dim_c = p * n  # 16

    # The matrix multiplication tensor T[i,j,k] = 1 if j is the "connecting" index
    # T is a 3-tensor of shape (n*m, m*p, p*n) but really it's:
    # T[a_idx, b_idx, c_idx] where:
    #   a_idx = i*m + j  (i in 0..n-1, j in 0..m-1)
    #   b_idx = j*p + k  (j in 0..m-1, k in 0..p-1)
    #   c_idx = k*n + i  (k in 0..p-1, i in 0..n-1)
    # T[a_idx, b_idx, c_idx] = 1 if the j in a_idx equals the j in b_idx AND the i in a_idx equals the i in c_idx AND the k in b_idx equals the k in c_idx
    # Actually, T[i,j; j,k; k,i] = 1, else 0.
    # So T is a 3D array where T[a,b,c] = 1 iff:
    #   a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
    #   i = a // m, j = a % m
    #   j' = b // p, k = b % p
    #   k' = c // n, i' = c % n
    #   T=1 iff i==i' and j==j' and k==k'

    # Build the tensor as a sparse representation for verification
    # T[a,b,c] = 1 iff (a//m == c%n) and (a%m == b//p) and (b%p == c//n)

    def verify(U, V, W, rank):
        """Verify that sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all a,b,c"""
        # U: (dim_a, rank), V: (dim_b, rank), W: (dim_c, rank)
        # Compute for all a,b,c: sum_r U[a,r]*V[b,r]*W[c,r]
        # This is (U @ V.T) gives (dim_a, dim_b) intermediate, then contract with W
        # Actually: result[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # = (U @ V.T)[a,b] weighted... no.
        # result[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # For each r: U[:,r] * V[:,r] * W[:,r] contributes to outer product
        # So result = sum_r np.outer(np.outer(U[:,r], V[:,r]), W[:,r])
        # But that's expensive for large rank. Let's do it in chunks.
        
        # Efficient: for each r, add U[:,r] @ (V[:,r] * W[:,r]).T ... no
        # result[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # = sum_r U[a,r] * (V[b,r] * W[c,r])
        # For fixed a: result[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Let's compute M = V @ W.T which is (dim_b, dim_c) where M[b,c] = sum_r V[b,r]*W[c,r]
        # No, that's not right either.
        
        # result[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # For each a: result[a,:,:] = sum_r U[a,r] * (V[:,r] * W[:,r]).T ... 
        # V[:,r] is (dim_b,), W[:,r] is (dim_c,), V[:,r]*W[:,r] doesn't make sense (different sizes)
        # V[:,r] * W[:,r] element-wise only if dim_b == dim_c, which is not the case.
        
        # Correct: result[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # This is a 3-tensor contraction. Let's compute it as:
        # For each r: term = np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
        # Sum over r.
        
        # More efficient: 
        # result = np.einsum('ar,br,cr->abc', U, V, W)
        # But that creates a huge array (8*32*16 = 4096) which is fine.
        
        result = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Build expected tensor
        expected = np.zeros((dim_a, dim_b, dim_c))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    expected[a, b, c] = 1.0
        
        # Check if they match (allowing for integer/half-integer, so use tolerance)
        if not np.allclose(result, expected, atol=1e-9):
            return False
        return True

    # Strategy: Try to find a low-rank decomposition using a search approach.
    # The trivial decomposition has rank n*m*p = 64 (one term per (i,j,k) triple).
    # Strassen-based gives 56. Can we do better?
    
    # First, let's establish the trivial rank-64 solution as a fallback
    U_trivial = np.zeros((dim_a, n*m*p), dtype=float)
    V_trivial = np.zeros((dim_b, n*m*p), dtype=float)
    W_trivial = np.zeros((dim_c, n*m*p), dtype=float)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_trivial[a, r] = 1.0
                V_trivial[b, r] = 1.0
                W_trivial[c, r] = 1.0
                r += 1
    
    # Verify trivial solution
    assert verify(U_trivial, V_trivial, W_trivial, n*m*p), "Trivial solution should be valid"
    
    best_rank = n * m * p
    best_U = U_trivial.copy()
    best_V = V_trivial.copy()
    best_W = W_trivial.copy()
    
    # Try Strassen-based approach (rank 56)
    # But first, let's try to find better decompositions using optimization
    
    # Approach: Use a greedy/iterative method to find low-rank decompositions
    # We'll try ranks from 48 upward and see if we can find a valid one
    
    # Actually, let's first try to construct a Strassen-like decomposition properly
    # and verify it, then try to optimize.
    
    # The 2x4 * 4x8 problem:
    # A is 2x4, B is 4x8, C is 2x8
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    # Let's try a different structural approach:
    # Partition A into two 2x2 blocks: A = [A_L, A_R]
    # Partition B into four 2x2 blocks in a 2x4 grid: B = [[B_00, B_01, B_02, B_03],
    #                                                          [B_10, B_11, B_12, B_13]]
    # where B_jk is 2x2
    # C = [A_L * B_0k + A_R * B_1k for k=0..3]
    
    # Each A_L * B_0k and A_R * B_1k is a 2x2 * 2x2 product
    # Using Strassen: 7 multiplications each
    # Total: 4 columns * 2 (L,R) * 7 = 56
    
    # But maybe we can share some computations between the 4 column blocks?
    # For example, A_L is the same for all 4 column blocks of B_top.
    # So we can precompute the 7 Strassen factors of A_L once, and similarly for A_R.
    # Then for each column block, we only need the B-side factors.
    
    # This is already what the rank-56 solution does implicitly.
    
    # Let's try to find a rank < 56 solution using a randomized search or optimization.
    
    # Alternative: Use the fact that the tensor has a specific structure and try
    # to find a decomposition with fewer terms using integer programming or search.
    
    # Let's try a simple approach: randomly initialize U, V, W with small integers
    # and try to adjust them to satisfy the tensor equation.
    
    # Actually, let's try a more systematic approach:
    # The tensor T has entries that are 0 or 1. We need:
    # sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    
    # This is a system of equations. For rank R, we have R*3 unknowns (entries of U,V,W)
    # and n*m*p*p*n = 8*32*16 = 4096 equations.
    
    # For R=48: 144 unknowns, 4096 equations. Overdetermined.
    # For R=56: 168 unknowns, 4096 equations.
    
    # Let's try to use a randomized approach to find a valid decomposition.
    
    def try_random_decomposition(rank, num_trials=1000, seed=None):
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.default_rng()
        
        for trial in range(num_trials):
            # Initialize with random small integers
            U = rng.randint(-2, 3, size=(dim_a, rank))
            V = rng.randint(-2, 3, size=(dim_b, rank))
            W = rng.randint(-2, 3, size=(dim_c, rank))
            
            # Convert to float for computation
            U_f = U.astype(float)
            V_f = V.astype(float)
            W_f = W.astype(float)
            
            result = np.einsum('ar,br,cr->abc', U_f, V_f, W_f)
            
            # Build expected
            expected = np.zeros((dim_a, dim_b, dim_c))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        expected[a, b, c] = 1.0
            
            if np.allclose(result, expected, atol=1e-9):
                return U, V, W
            elif time.time() > deadline - 10:
                break
        
        return None
    
    # Try ranks from 48 to 55
    for rank in range(48, min(56, best_rank)):
        if time.time() > deadline - 20:
            break
        
        result = try_random_decomposition(rank, num_trials=500)
        if result is not None:
            U, V, W = result
            if verify(U, V, W, rank):
                best_rank = rank
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                break
    
    # If we haven't found anything better than 56, use the Strassen-based solution
    if best_rank >= 56:
        # Build the Strassen-based rank-56 solution
        x_forms = np.array([
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1]
        ])
        
        y_forms = np.array([
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [-1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ])
        
        z_forms = np.array([
            [1, 0, 0, 1, -1, 0, 1],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [1, -1, 1, 0, 0, 1, 0]
        ])
        
        num_bcol_blocks = 4
        num_brow_blocks = 2
        num_strassen = 7
        
        rank56 = num_bcol_blocks * num_brow_blocks * num_strassen
        U56 = np.zeros((dim_a, rank56), dtype=float)
        V56 = np.zeros((dim_b, rank56), dtype=float)
        W56 = np.zeros((dim_c, rank56), dtype=float)
        
        r = 0
        for g in range(num_bcol_blocks):
            for b in range(num_brow_blocks):
                for s in range(num_strassen):
                    for i in range(n):
                        for jp in range(2):
                            a_idx = i * m + b * 2 + jp
                            U56[a_idx, r] = x_forms[s, i * 2 + jp]
                    
                    for jp in range(2):
                        for kp in range(2):
                            b_idx = (b * 2 + jp) * p + (2 * g + kp)
                            V56[b_idx, r] = y_forms[s, jp * 2 + kp]
                    
                    for i in range(n):
                        for kp in range(2):
                            k = 2 * g + kp
                            c_idx = k * n + i
                            W56[c_idx, r] = z_forms[i * 2 + kp, s]
                    
                    r += 1
        
        # Verify
        if verify(U56, V56, W56, rank56):
            best_rank = rank56
            best_U = U56.copy()
            best_V = V56.copy()
            best_W = W56.copy()
        else:
            # Fall back to trivial
            best_rank = n * m * p
            best_U = U_trivial.copy()
            best_V = V_trivial.copy()
            best_W = W_trivial.copy()
    
    # Final verification
    if not verify(best_U, best_V, best_W, best_rank):
        # Fall back to trivial
        best_rank = n * m * p
        best_U = U_trivial.copy()
        best_V = V_trivial.copy()
        best_W = W_trivial.copy()
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
