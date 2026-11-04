"""Rank-minimal exact tensor decomposition for 3x3 by 3x4 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 4
    deadline = time.time() + 130  # safety margin before 150s kill

    # Try to find a rank-25 decomposition via Strassen-style block decomposition
    # Split A (3x3) into blocks: A11 (2x2), A12 (2x1), A21 (1x2), A22 (1x1)
    # Split B (3x4) into blocks: B11 (2x2), B12 (2x2), B21 (1x2), B22 (1x2)
    # Output C (3x4) has blocks: C11 (2x2), C12 (2x2), C21 (1x2), C22 (1x2)
    #
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    #
    # Using Strassen on the 2x2 block part (A11, B11 -> C11):
    # M1 = (A11 + A22) * (B11 + B22)  -- but this doesn't directly work for our block structure
    #
    # Let's use a more direct approach. The full multiplication can be viewed as:
    # We need to compute 3*3*4 = 36 scalar products naively.
    # 
    # A rank-25 decomposition would save 11 multiplications.
    # 
    # Let's try an explicit construction. The tensor T has indices (i, j, k) with
    # i in 0..2 (rows of A), j in 0..2 (cols of A / rows of B), k in 0..3 (cols of B).
    # T[i,j,k] = e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i} in the appropriate spaces.
    #
    # Actually, let me think about this differently. The matrix multiplication tensor
    # M_{n,m,p} has entries M[i,j,k] where i indexes rows of first matrix, j indexes
    # cols of first/rows of second, k indexes cols of second. The decomposition is:
    # M[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    #
    # For rank 25, we need 25 triples (U_r, V_r, W_r) such that this holds exactly.
    #
    # Let me try a known result: the rank of 3x3x3 matrix multiplication is at most 23
    # (Landsberg 1987 showed rank <= 23, and it's known to be at least 19).
    # For 3x3x4, the rank is at most... let me think.
    #
    # Actually, let me just try to construct a rank-25 decomposition explicitly.
    # 
    # One approach: use the fact that we can write the 3x3x4 multiplication as
    # a combination of smaller multiplications.
    #
    # Let's try: treat the 3x3x4 as (2x1)x(2x2)x(1x2) plus corrections.
    #
    # Actually, let me just use a direct algebraic approach. I'll set up the system
    # of equations and solve it.
    #
    # For rank r, we need to find U (9 x r), V (12 x r), W (12 x r) such that
    # for all i in 0..2, j in 0..2, k in 0..3:
    #   sum_{r=0}^{r-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # and for all other (i,j,k) combinations not corresponding to valid multiplications,
    # the sum should be 0.
    #
    # Wait, actually the tensor is defined such that T[i,j,k] = 1 if the indices
    # correspond to a valid multiplication entry, which they all do for i in [0,n),
    # j in [0,m), k in [0,p). So all 36 entries are 1.
    #
    # No wait, that's not right. The matrix multiplication tensor has T[i,j,k] = 1
    # for all valid (i,j,k) because each output entry C[i,k] = sum_j A[i,j]*B[j,k].
    # So the tensor is the sum over j of e_i ⊗ e_j ⊗ e_k for each (i,k) pair... 
    # 
    # Let me reconsider. The bilinear map is: (A, B) -> A*B
    # where A is n x m, B is m x p, result is n x p.
    # The tensor T in (R^{n*m})^* ⊗ (R^{m*p})^* ⊗ R^{n*p} has
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = i*p+k for some i,j,k, else 0.
    # 
    # So T has 1s at positions (i*m+j, j*p+k, i*p+k) for all i in [0,n), j in [0,m), k in [0,p).
    # That's n*m*p = 36 ones.
    #
    # A rank-r decomposition: T = sum_{r=1}^{r} u_r ⊗ v_r ⊗ w_r
    # where u_r is in R^{n*m}, v_r in R^{m*p}, w_r in R^{n*p}.
    #
    # So we need: for each (a, b, c), T[a,b,c] = sum_{r=1}^{R} U[a,r] * V[b,r] * W[c,r]
    #
    # Let me try rank 25 with a structured approach.
    #
    # I'll use the following block decomposition:
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    # B = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x2, B21 is 1x2, B22 is 1x2
    # C = [[C11, C12], [C21, C22]] where C11 is 2x2, C12 is 2x2, C21 is 1x2, C22 is 1x2
    #
    # C11 = A11*B11 + A12*B21  (2x2 = 2x2*2x2 + 2x1*1x2)
    # C12 = A11*B12 + A12*B22  (2x2 = 2x2*2x2 + 2x1*1x2)
    # C21 = A21*B11 + A22*B21  (1x2 = 1x2*2x2 + 1x1*1x2)
    # C22 = A21*B12 + A22*B22  (1x2 = 1x2*2x2 + 1x1*1x2)
    #
    # Naive cost: 4*4 + 2*2 + 2*2 + 1*2 + 1*2 + 1*2 + 1*2 = 16+4+4+2+2+2+2 = 32
    # That's better than 36!
    #
    # Now apply Strassen to the 2x2*2x2 -> 2x2 multiplications:
    # A11*B11: rank 7 (Strassen)
    # A11*B12: rank 7
    # A21*B11: rank 7
    # A21*B12: rank 7
    # 
    # And the smaller ones:
    # A12*B21: 2x1 * 1x2 -> rank 2 (just 2 multiplications)
    # A12*B22: 2x1 * 1x2 -> rank 2
    # A22*B21: 1x1 * 1x2 -> rank 2
    # A22*B22: 1x1 * 1x2 -> rank 2
    #
    # Total: 7*4 + 2*4 = 28 + 8 = 36. Hmm, same as naive.
    #
    # Let me try a different splitting. What if I use:
    # A11 (2x2), A12 (2x1), A21 (1x2), A22 (1x1)
    # B11 (2x2), B12 (2x2), B21 (1x2), B22 (1x2)
    #
    # But combine some terms:
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    #
    # Note that C11 and C12 both involve A11. Let me define:
    # M1 = A11 * (B11 + B12)  -- 2x2 * 2x4 -> 2x4, rank 7 (Strassen on the 2x2 part, times 2)
    # Actually, 2x2 * 2x4 can be done with rank 7*2 = 14? No...
    #
    # Let me think more carefully. A 2x2 times 2x4 multiplication:
    # A (2x2) * B (2x4) = C (2x4)
    # This has 2*2*4 = 16 scalar multiplications naively.
    # Using Strassen on the 2x2 block: we can view B as two 2x2 blocks B[:,0:2] and B[:,2:4].
    # A * B[:,0:2] uses rank 7, A * B[:,2:4] uses rank 7. Total rank 14.
    #
    # Similarly, 1x2 * 2x4: A (1x2) * B (2x4) = C (1x4), rank 2*4 = 8 naively, or...
    # Actually this is just a dot product for each column, so rank 2 (one multiplication per column pair).
    # Wait no, it's 1x2 times 2x4, so for each of the 4 output columns, we do 2 multiplications.
    # Total rank = 2*4 = 8? No, the rank of the bilinear map is the minimum number of scalar
    # multiplications. For 1x2 * 2x4, it's just 2*4 = 8 scalar multiplications, and the rank
    # is 8 (you can't do better since each output entry depends on distinct input pairs... 
    # actually no, the rank of M_{1,2,4} is 2*4 = 8? Let me think... 
    # M_{1,m,p} has rank m*p because each output entry C[0,k] = sum_j A[0,j]*B[j,k],
    # and these are all independent bilinear forms. So rank = m*p = 8.
    #
    # OK let me try a completely different approach. Let me just set up the optimization
    # problem and try to solve it numerically, then round to integers/half-integers.
    #
    # For rank 25, I'll use a randomized initialization and gradient descent on the
    # non-convex problem, then check if the solution is close to integer/half-integer.
    
    def try_rank(r):
        """Try to find a rank-r decomposition."""
        if time.time() > deadline:
            return None
        
        n, m, p = 3, 3, 4
        nm = n * m  # 9
        mp = m * p  # 12
        np_ = p * n  # 12
        
        # Build the tensor T as a 3D array
        T = np.zeros((nm, mp, np_))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        
        # Try random initialization with scaling
        np.random.seed(42)
        best_err = float('inf')
        best_solution = None
        
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random values, scaled to encourage integer-like solutions
            U = np.random.randn(nm, r) * 0.5
            V = np.random.randn(mp, r) * 0.5
            W = np.random.randn(np_, r) * 0.5
            
            # Use a simple iterative method: alternating least squares
            for it in range(2000):
                if time.time() > deadline - 5:
                    break
                
                # Update U: for each column r, minimize ||T - sum_r U[:,r] V[:,r]^T W[:,r]^T||
                # This is tricky because it's a 3-way product. Let me use a different approach.
                
                # Actually, let me use the fact that for fixed V and W, the optimal U is:
                # U[:,r] = argmin ||T - sum_r U[:,r] (V[:,r] ⊗ W[:,r])||_F
                # where V[:,r] ⊗ W[:,r] is the outer product vectorized.
                # This is a linear least squares problem.
                
                # Build the design matrix: for each r, the "coefficient" vector is V[:,r] ⊗ W[:,r]
                # T reshaped is (nm*mp*np_), and each U[:,r] is multiplied by (V[:,r] ⊗ W[:,r])
                # So T_flat = sum_r (V[:,r] ⊗ W[:,r]) ⊗ U[:,r] ... this is getting complicated.
                
                # Let me use a simpler approach: gradient descent with a custom gradient.
                # Loss = ||T - sum_r u_r ⊗ v_r ⊗ w_r||_F^2
                # dL/dU[a,r] = -2 * sum_{b,c} T[a,b,c] V[b,r] W[c,r] + 2 * sum_{b,c} (sum_s U[a,s]V[b,s]W[c,s]) V[b,r] W[c,r]
                #            = -2 * (T * (V[:,r] ⊗ W[:,r]))[a] ... where * means element-wise product summed over b,c
                
                # Compute the current reconstruction
                recon = np.einsum('ar,br,cr->abc', U, V, W)
                residual = T - recon
                
                # Gradient for U
                grad_U = -2 * np.einsum('abc,br,cr->ar', residual, V, W)
                # Gradient for V
                grad_V = -2 * np.einsum('abc,ar,cr->br', residual, U, W)
                # Gradient for W
                grad_W = -2 * np.einsum('abc,ar,br->cr', residual, U, V)
                
                # Simple gradient descent with adaptive step size
                lr = 0.01
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Check convergence
                err = np.linalg.norm(residual)
                if err < 1e-10:
                    break
                
                if err < best_err:
                    best_err = err
                    best_solution = (U.copy(), V.copy(), W.copy())
            
            if best_err < 1e-6:
                break
        
        if best_err > 1e-4:
            return None
        
        # Try to round to integers or half-integers
        U_r, V_r, W_r = best_solution
        
        # Scale to make entries close to integers
        # Find a common scale
        all_entries = np.concatenate([U_r.flatten(), V_r.flatten(), W_r.flatten()])
        nonzero = all_entries[np.abs(all_entries) > 1e-8]
        if len(nonzero) == 0:
            return None
        
        # Try different scales
        for scale in [0.5, 1.0, 2.0, 0.25, 4.0]:
            U_scaled = U_r / scale
            V_scaled = V_r / scale
            W_scaled = W_r / scale
            
            # Round to nearest half-integer
            U_rounded = np.round(U_scaled * 2) / 2
            V_rounded = np.round(V_scaled * 2) / 2
            W_rounded = np.round(W_scaled * 2) / 2
            
            # Check if rounded solution is exact
            recon_rounded = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
            err_rounded = np.linalg.norm(recon_rounded - T)
            
            if err_rounded < 1e-8:
                # Convert to lists
                return {
                    "U": U_rounded.tolist(),
                    "V": V_rounded.tolist(),
                    "W": W_rounded.tolist()
                }
        
        return None
    
    # Try rank 25
    result = try_rank(25)
    if result is not None:
        return result
    
    # Fall back to schoolbook if we can't find rank 25
    n, m, p = 3, 3, 4
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
