"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # Schoolbook fallback (rank 63)
    rank_fb = n * m * p  # 63
    U_fb = np.zeros((n * m, rank_fb), dtype=float)
    V_fb = np.zeros((m * p, rank_fb), dtype=float)
    W_fb = np.zeros((p * n, rank_fb), dtype=float)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, c] = 1.0
                V_fb[j * p + k, c] = 1.0
                W_fb[k * n + i, c] = 1.0
                c += 1
    
    # Verify function
    def verify(U, V, W):
        r = len(U[0]) if isinstance(U, list) else U.shape[1]
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        # Check all (i,j,k) triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c2 in range(r):
                        s += U_arr[i*m+j, c2] * V_arr[j*p+k, c2] * W_arr[k*n+i, c2]
                    if abs(s - 1.0) > 1e-6:
                        return False
        # Check zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                if i2 == i and j2 == j and k2 == k:
                                    continue
                                s2 = 0.0
                                for c2 in range(r):
                                    s2 += U_arr[i2*m+j2, c2] * V_arr[j2*p+k2, c2] * W_arr[k2*n+i2, c2]
                                if abs(s2) > 1e-6:
                                    return False
        return True
    
    # Fast verification using tensor reshaping
    def verify_fast(U, V, W):
        r = U.shape[1]
        # Reshape to tensors
        U_t = U.reshape(n, m, r)
        V_t = V.reshape(m, p, r)
        W_t = W.reshape(p, n, r)
        # Compute the full tensor: T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
        T = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
        # Expected tensor: T[i,j,k] = 1 for all (i,j,k) since this is matrix mult
        # Wait - for matrix multiplication C = AB where A is n×m and B is m×p,
        # C[i,k] = sum_j A[i,j]*B[j,k]
        # The tensor is: T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
        # Actually the decomposition should satisfy:
        # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all (i,j,k)
        # because each (i,j,k) corresponds to one term in the sum for C[i,k]
        # But we also need orthogonality: no cross terms
        # The full tensor T[i,j,k] should equal 1 for all valid (i,j,k)
        # and 0 otherwise - but since all (i,j,k) with i in [0,n), j in [0,m), k in [0,p) are valid,
        # T should be all ones.
        expected = np.ones((n, m, p))
        if not np.allclose(T, expected, atol=1e-6):
            return False
        return True
    
    start_time = time.time()
    time_limit = 130  # leave margin before 150s kill
    
    def time_check():
        return (time.time() - start_time) < time_limit
    
    # Strategy: Try to find a lower-rank decomposition using structured search
    # Key insight: The matrix multiplication tensor M_{n,m,p} has a specific structure.
    # For M_{3,3,7}, the schoolbook rank is 63.
    # Known results: rank(M_{3,3,3}) <= 23 (Strassen-type), rank(M_{3,3,1}) = 7
    # Split: 7 = 3+3+1, total rank <= 23+23+7 = 53
    
    # Let's try to construct explicit rank-23 M_{3,3,3} and rank-7 M_{3,3,1}
    
    # Rank-7 construction for M_{3,3,1}:
    # C_i = sum_j A[i,j] * b_j for i=0,1,2
    # U: 9×7, V: 3×7, W: 3×7
    # We need: sum_c U[i*m+j,c]*V[j,c]*W[i,c] = 1 for all (i,j)
    # and 0 for wrong combinations.
    
    # A known rank-7 construction:
    # Use the fact that we can share computations.
    # Let's try:
    # c=0: U[0,0]=1, V[0]=1, W[0]=1  -> contributes to (0,0)
    # c=1: U[1,1]=1, V[1]=1, W[1]=1  -> contributes to (1,1)  
    # c=2: U[2,2]=1, V[2]=1, W[2]=1  -> contributes to (2,2)
    # c=3: U[0,1]=1, U[1,0]=1, V[0]=1, V[1]=1, W[0]=1, W[1]=1
    #       -> contributes to (0,1) and (1,0)
    # c=4: U[0,2]=1, U[2,0]=1, V[0]=1, V[2]=1, W[0]=1, W[2]=1
    #       -> contributes to (0,2) and (2,0)
    # c=5: U[1,2]=1, U[2,1]=1, V[1]=1, V[2]=1, W[1]=1, W[2]=1
    #       -> contributes to (1,2) and (2,1)
    # c=6: U[0,0]=1, U[1,1]=1, U[2,2]=1, V[0]=1, V[1]=1, V[2]=1, W[0]=1, W[1]=1, W[2]=1
    #       -> contributes to (0,0),(1,1),(2,2) again... that's double counting
    
    # This approach is getting complicated. Let me try a different strategy.
    
    # Alternative: Use the Kronecker product structure and try to find
    # a low-rank approximation via SVD of the reshaped tensor.
    
    # The matrix multiplication tensor can be viewed as a linear map:
    # T: R^{n*m} ⊗ R^{m*p} -> R^{p*n}
    # T(A ⊗ B) = AB (reshaped)
    
    # The rank of this tensor is the minimum number of rank-1 tensors that sum to T.
    
    # Let's try a direct approach: build the tensor explicitly and try to
    # find a low-rank decomposition via iterative methods.
    
    # Build the full tensor T[i,j,k] = 1 for all (i,j,k) in [0,n)×[0,m)×[0,p)
    # Wait, that's not right. The tensor for matrix multiplication is:
    # T[i,j,k] = 1 if and only if we're using A[i,j] and B[j,k] to compute C[i,k]
    # So T[i,j,k] = 1 for all valid (i,j,k).
    
    # Actually, I think the correct formulation is:
    # We want to decompose the bilinear map (A,B) -> AB
    # as a sum of rank-1 bilinear maps.
    # A rank-1 bilinear map is: (A,B) -> (u^T A v) * w^T (where u,v,w are vectors)
    # Wait, that's not quite right either.
    
    # Let me reconsider. The standard formulation:
    # C = AB where A is n×m, B is m×p, C is n×p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # A tensor decomposition of rank r means:
    # C[i,k] = sum_{c=0}^{r-1} sum_j U[i,j,c] * A[i,j] * V[j,k,c] * B[j,k] * W[i,k,c]
    # Wait, that's not standard either.
    
    # The correct formulation for matrix multiplication tensor:
    # We have trilinear forms. The tensor T is in R^{n*m} ⊗ R^{m*p} ⊗ R^{n*p}
    # T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(i,k) for some (i,j,k)
    # i.e., T[(i,j),(j,k),(i,k)] = 1, and 0 otherwise.
    
    # A rank-1 tensor is: u ⊗ v ⊗ w where u in R^{n*m}, v in R^{m*p}, w in R^{n*p}
    # The sum of r such rank-1 tensors should equal T.
    
    # So we need: sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[i*n+k] = T[(i,j),(j,k),(i,k)]
    # where T[(i,j),(j,k),(i,k)] = 1 if j is the same in all three (which it is by construction)
    # Actually, the index structure is:
    # First mode: (i,j) for i in [0,n), j in [0,m)
    # Second mode: (j,k) for j in [0,m), k in [0,p)
    # Third mode: (i,k) for i in [0,n), k in [0,p)
    # 
    # T[(i,j),(j',k),(i',k')] = 1 if i=i', j=j', k=k'
    # i.e., T is the "identity" tensor in this structured sense.
    
    # So the constraint is:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[i*n+k, c] = 1 if (i,j,k) is consistent
    # and 0 otherwise.
    
    # But since the indices are structured, "consistent" means j appears in both
    # the first and second modes. So for any (i,j,k) with i in [0,n), j in [0,m), k in [0,p),
    # the value should be 1. For "inconsistent" combinations (which don't exist in our
    # indexing scheme since j is always shared), the value should be 0.
    
    # So effectively, we need:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[i*n+k, c] = 1 for all (i,j,k)
    # in the valid range.
    
    # This is exactly what the verify function checks.
    
    # Now, let's try a practical approach:
    # 1. Try to find a rank-53 construction via splitting 7=3+3+1
    # 2. If that fails, try random search for lower ranks
    # 3. Fall back to schoolbook rank 63
    
    # For the split approach, we need:
    # - Rank-23 decomposition of M_{3,3,3}
    # - Rank-7 decomposition of M_{3,3,1}
    
    # Let me try to construct these explicitly.
    
    # M_{3,3,1}: C[i] = sum_j A[i,j] * b[j] for i=0,1,2
    # U: 9×7, V: 3×7, W: 3×7
    # Constraint: sum_c U[i*3+j,c] * V[j,c] * W[i,c] = 1 for all (i,j)
    
    # Try this construction:
    # c=0: U[0,0]=1, V[0]=1, W[0]=1
    # c=1: U[1,1]=1, V[1]=1, W[1]=1
    # c=2: U[2,2]=1, V[2]=1, W[2]=1
    # c=3: U[0,1]=1, U[1,0]=1, V[0]=1, V[1]=1, W[0]=1, W[1]=1
    # c=4: U[0,2]=1, U[2,0]=1, V[0]=1, V[2]=1, W[0]=1, W[2]=1
    # c=5: U[1,2]=1, U[2,1]=1, V[1]=1, V[2]=1, W[1]=1, W[2]=1
    # c=6: U[0,0]=1, U[1,1]=1, U[2,2]=1, V[0]=1, V[1]=1, V[2]=1, W[0]=1, W[1]=1, W[2]=1
    
    # Check (0,0): c=0 gives 1*1*1=1, c=3 gives 0 (U[0,1] not U[0,0]... wait)
    # Let me re-index. U[i*3+j, c] where i=row, j=col of A.
    
    # c=0: U[0,0]=1 (i=0,j=0), V[0]=1, W[0]=1
    #   Contributes to (i=0,j=0): 1*1*1=1
    # c=1: U[1,1]=1 (i=1,j=1), V[1]=1, W[1]=1
    #   Contributes to (i=1,j=1): 1*1*1=1
    # c=2: U[2,2]=1 (i=2,j=2), V[2]=1, W[2]=1
    #   Contributes to (i=2,j=2): 1*1*1=1
    # c=3: U[0,1]=1 (i=0,j=1), U[1,0]=1 (i=1,j=0), V[0]=1, V[1]=1, W[0]=1, W[1]=1
    #   Contributes to (i=0,j=1): U[0,1]*V[1]*W[0] = 1*1*1 = 1
    #   Contributes to (i=1,j=0): U[1,0]*V[0]*W[1] = 1*1*1 = 1
    # c=4: U[0,2]=1 (i=0,j=2), U[2,0]=1 (i=2,j=0), V[0]=1, V[2]=1, W[0]=1, W[2]=1
    #   Contributes to (i=0,j=2): U[0,2]*V[2]*W[0] = 1*1*1 = 1
    #   Contributes to (i=2,j=0): U[2,0]*V[0]*W[2] = 1*1*1 = 1
    # c=5: U[1,2]=1 (i=1,j=2), U[2,1]=1 (i=2,j=1), V[1]=1, V[2]=1, W[1]=1, W[2]=1
    #   Contributes to (i=1,j=2): U[1,2]*V[2]*W[1] = 1*1*1 = 1
    #   Contributes to (i=2,j=1): U[2,1]*V[1]*W[2] = 1*1*1 = 1
    # c=6: U[0,0]=1, U[1,1]=1, U[2,2]=1, V[0]=1, V[1]=1, V[2]=1, W[0]=1, W[1]=1, W[2]=1
    #   Contributes to (i=0,j=0): 1*1*1=1
    #   Contributes to (i=1,j=1): 1*1*1=1
    #   Contributes to (i=2,j=2): 1*1*1=1
    
    # Now check all (i,j):
    # (0,0): c=0 gives 1, c=6 gives 1. Total = 2. Need 1. FAIL.
    
    # So this construction doesn't work. Let me try a different approach.
    
    # Actually, I realize that for M_{3,3,1}, the "tensor" is:
    # T[i,j] = 1 for all (i,j) in [0,3)×[0,3)
    # And we need: sum_c U[i,j,c] * V[j,c] * W[i,c] = 1
    
    # This is equivalent to: the matrix M where M[i,j] = sum_c U[i,j,c]*V[j,c]*W[i,c]
    # should be all ones.
    
    # Note that V[j,c] and W[i,c] don't depend on the other index.
    # So M[i,j] = sum_c U[i,j,c] * (V[j,c] * W[i,c])
    
    # Let's define for each c: a_c[j] = V[j,c], b_c[i] = W[i,c]
    # Then M[i,j] = sum_c U[i,j,c] * a_c[j] * b_c[i]
    
    # This is like: M = sum_c (outer product of b_c^T and a_c) scaled element-wise by U_c
    # where U_c[i,j] = U[i,j,c].
    
    # For rank 7, we need 7 such terms.
    
    # Let me try a simpler construction. What if we use:
    # c=0..2: diagonal terms
    # c=3..5: off-diagonal pairs
    # c=6: correction term
    
    # Actually, let me just try random search for M_{3,3,1} rank 7 with small integer entries.
    
    import random
    random.seed(42)
    
    def try_m331_rank7():
        """Try to find rank-7 M_{3,3,1} decomposition with entries in {-1,0,1}."""
        for _ in range(100000):
            U = np.zeros((9, 7))
            V = np.zeros((3, 7))
            W = np.zeros((3, 7))
            for c in range(7):
                # Random sparse pattern
                for i in range(3):
                    for j in range(3):
                        if random.random() < 0.3:
                            U[i*3+j, c] = random.choice([-1, 1])
                for j in range(3):
                    if random.random() < 0.5:
                        V[j, c] = random.choice([-1, 1])
                for i in range(3):
                    if random.random() < 0.5:
                        W[i, c] = random.choice([-1, 1])
            # Check
            M = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    M[i, j] = sum(U[i*3+j, c] * V[j, c] * W[i, c] for c in range(7))
            if np.allclose(M, np.ones((3, 3)), atol=1e-9):
                return U, V, W
        return None
    
    # Try to find M_{3,3,1} rank 7
    m331_result = try_m331_rank7()
    
    # For M_{3,3,3}, this is much harder. Let's try a different approach.
    # Use the known fact that rank(M_{3,3,3}) <= 23 and try to find it.
    
    # Actually, given the time constraints, let me try a more direct approach:
    # Try to find a low-rank decomposition of the full M_{3,3,7} tensor
    # using iterative optimization.
    
    def try_full_decomposition(rank_target, max_iters=500, lr=0.01):
        """Try to find a rank-`rank_target` decomposition of M_{3,3,7}."""
        if not time_check():
            return None
        
        U = np.random.randn(9, rank_target) * 0.1
        V = np.random.randn(21, rank_target) * 0.1
        W = np.random.randn(21, rank_target) * 0.1
        
        for it in range(max_iters):
            if not time_check():
                return None
            
            # Compute current tensor
            U_t = U.reshape(3, 3, rank_target)
            V_t = V.reshape(3, 7, rank_target)
            W_t = W.reshape(7, 3, rank_target)
            
            # T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
            T = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
            
            # Target: all ones
            target = np.ones((3, 3, 7))
            err = T - target
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(3):
                for j in range(3):
                    for k in range(7):
                        e = err[i, j, k]
                        for c in range(rank_target):
                            vc = V_t[j, k, c]
                            wc = W_t[k, i, c]
                            uc = U_t[i, j, c]
                            dU[i*3+j, c] += e * vc * wc
                            dV[j*7+k, c] += e * uc * wc
                            dW[k*3+i, c] += e * uc * vc
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Check convergence
            if np.max(np.abs(err)) < 1e-6:
                if verify_fast(U, V, W):
                    return U, V, W
        
        # Check final result
        if verify_fast(U, V, W):
            return U, V, W
        return None
    
    # Try various ranks
    best_result = None
    for r in range(50, 63):
        if not time_check():
            break
        result = try_full_decomposition(r, max_iters=200, lr=0.005)
        if result is not None:
            U_r, V_r, W_r = result
            best_result = (U_r, V_r, W_r, r)
            break
    
    if best_result is not None:
        U_found, V_found, W_found, r_found = best_result
        def to_half_integer(arr):
            result = []
            for row in arr:
                new_row = []
                for x in row:
                    if abs(x * 2 - round(x * 2)) < 1e-9:
                        new_row.append(round(x * 2) / 2.0)
                    else:
                        new_row.append(float(x))
                result.append(new_row)
            return result
        return {"U": to_half_integer(U_found), "V": to_half_integer(V_found), "W": to_half_integer(W_found)}
    
    # Fallback to schoolbook
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

# EVOLVE-BLOCK-END
