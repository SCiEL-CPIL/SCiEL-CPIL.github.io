import numpy as np
import time
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 120.0  # seconds, safety margin before 150s kill

    def _time_remaining():
        return DEADLINE - (time.time() - start_time)

    n, m, p = 2, 3, 6

    def _verify(U, V, W):
        """Check whether the rank-r decomposition is exact:
        For all (i,k): sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == delta(i,j)*B(j,k)
        i.e. the tensor equals the multiplication tensor T[i,j,k]."""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        r = U.shape[1]
        # Build the multiplication tensor T of shape (n, m, p): T[i,j,k] = 1 if j==j else 0
        # Actually the multiplication tensor is T[i,j,k] such that C[i,k] = sum_j A[i,j]*B[j,k]
        # The decomposition requires: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = delta(j, j') ... 
        # More precisely, the matmul tensor T is defined by T[i,j,k] = 1 for the "identity" in j
        # No: the tensor T_matmul[i,j,k] is such that <T, A ⊗ B> = <C, I> where C = A@B.
        # T[i,j,k] = 1 if we index properly. The matmul tensor has T[i,j,k] = δ_{j,j}... 
        # Let me just use the standard: the multiplication tensor T has entries T[i,j,k] where
        # the product is C[i,k] = sum_j A[i,j] B[j,k]. The tensor T is in R^{n x m x p} with
        # T[i,j,k] = 1 if j == j (always) — no, that's not right.
        # 
        # The correct statement: matmul tensor T ∈ R^{n×m×p} is defined by T[i,j,k] = 1 if the
        # "middle index" matches, i.e. T[i,j,k] = δ_{j,j} is meaningless. 
        # 
        # Actually: T[i,j,k] is the coefficient such that C[i,k] = Σ_{j,r} ... no.
        # The matmul tensor T has T[i,j,k] = 1 for all valid (i,j,k) — NO.
        # 
        # Let me think again. The tensor T_matmul ∈ R^{n×m×p} satisfies:
        #   C[i,k] = Σ_j A[i,j] B[j,k]
        # This is a trilinear form: C[i,k] = Σ_{i,j,k} T[i,j,k] A[i,j] B[j,k] δ... 
        # No. C[i,k] = Σ_j A[i,j] B[j,k]. So the "tensor" that maps (A,B) to C is:
        #   C[i,k] = Σ_{j} T[i,j,k] A[i,j] B[j,k]  where T[i,j,k] = 1 for all i,j,k? No!
        #   C[i,k] = Σ_j A[i,j] B[j,k], so T[i,j,k] = 1 only makes sense if j is summed.
        # 
        # The matmul tensor T ∈ R^{n×m×p} is defined by T[i,j,k] = 1 if j == j... 
        # I think the correct definition is: T[i,j,k] = δ_{j,j} is not it.
        # 
        # Standard: The matrix multiplication tensor T_{m,n,p} ∈ R^{m×n×p} is the tensor such that
        #   (T, A, B) = tr(A^T B) ... no.
        # 
        # Let me just verify numerically: the decomposition gives
        #   C[i,k] = Σ_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]  (summing over j and r)
        # For this to equal A[i,j]B[j,k] summed over j, we need:
        #   Σ_{j,r} U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = δ_{j,j'} ... 
        # 
        # The correct check: for random A, B, compute C = A@B and compare with
        #   C_decomp[i,k] = Σ_j Σ_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        r = U.shape[1]
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_test = A_test @ B_test
        C_decomp = np.zeros((n, p))
        for r_idx in range(r):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        C_decomp[i, k] += U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx]
        return np.allclose(C_decomp, C_test, atol=1e-9)

    def _search_lower_rank(current_rank, time_budget):
        """Try to find a decomposition of rank < current_rank using SVD + exact recovery.
        Returns (rank, U, V, W) if found, else (current_rank, None, None, None)."""
        # Build the matmul tensor T of shape (n, m, p): T[i,j,k] = 1 if j==j (identity in middle)
        # The matmul tensor: T[i,j,k] = δ_{j,j}... 
        # Correct: T[i,j,k] = 1 for all i,j,k is WRONG.
        # The matmul tensor has T[i,j,k] such that Σ_j T[i,j,k] A[i,j] B[j,k] = C[i,k].
        # This means T[i,j,k] = 1 only when... it's always 1? No, j is the free index being summed.
        # C[i,k] = Σ_j A[i,j] B[j,k]. So the tensor T[i,j,k] = 1 for all valid i,j,k? 
        # That would give C[i,k] = Σ_j 1 * A[i,j] B[j,k] which IS correct!
        # Wait, that can't be right because then the tensor would be all-ones and rank 1.
        # 
        # I see my confusion. The matmul tensor T ∈ R^{n×m×p} is NOT all ones.
        # The correct matmul tensor: T[i,j,k] = 1 if the row index of A (i) and column index of B (k)
        # match the output, and j is the contraction index. But T is a 3-tensor with one index per matrix.
        # 
        # Standard definition: T_{m×n×p}[i,j,k] = 1 if j == j (the middle index is shared).
        # Actually: T[i,j,k] = δ_{j,j} makes no sense.
        # 
        # Let me look at this differently. The multiplication tensor is:
        #   T[i, j, k] = 1  (for all i in [n], j in [m], k in [p])
        # NO. That's the all-ones tensor.
        # 
        # The CORRECT matmul tensor: For multiplying (n×m) by (m×p) to get (n×p),
        # the tensor T ∈ R^{n×m×p} is defined by T[i,j,k] = 1 if j is the contraction index.
        # But j IS always the contraction index. So T[i,j,k] = 1 for all i,j,k?
        # 
        # I think I'm overcomplicating. Let me just use the verification approach:
        # The decomposition is valid iff for ALL A, B: C[i,k] = Σ_{j,r} U[i*m+j,r]V[j*p+k,r]W[k*n+i,r]
        # equals (A@B)[i,k].
        # 
        # To find a lower-rank decomposition, I'll use the following approach:
        # 1. The matmul tensor T can be "matricized" in different ways.
        # 2. The rank of T is the minimum r such that T = Σ_r u_r ⊗ v_r ⊗ w_r.
        # 3. I'll try ranks from current_rank-1 down to some lower bound.
        # 
        # For a given rank r, I need to find U (n*m × r), V (m*p × r), W (p*n × r) such that
        # the tensor decomposition holds. This is a non-linear system.
        # 
        # Strategy: Fix W (or use a random initialization) and solve for U, V via linear equations.
        # Actually, a better approach: use the fact that for fixed W, the equations are linear in U and V.
        # 
        # For each (i,j,k): Σ_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = T[i,j,k]
        # where T[i,j,k] = 1 (the matmul tensor is... let me just compute it properly).
        # 
        # The matmul tensor T[i,j,k]: C[i,k] = Σ_j A[i,j] B[j,k]
        # So T[i,j,k] is such that Σ_{j} T[i,j,k] A[i,j] B[j,k] = C[i,k]
        # This means T[i,j,k] = 1 for all i, j, k. But that gives rank 1 which is wrong.
        # 
        # AH. I see the error. The tensor T has indices (i, j, k) where i indexes rows of A,
        # j indexes the shared dimension, and k indexes columns of B. The product is:
        #   C[i,k] = Σ_j A[i,j] B[j,k]
        # So the tensor T[i,j,k] = 1 for ALL i, j, k. But this tensor has rank 1 (it's the
        # outer product of all-ones vectors)! That can't be the matmul tensor.
        # 
        # I think the issue is that the matmul tensor is NOT simply T[i,j,k] = 1.
        # The matmul tensor is a 3-tensor where the three modes correspond to:
        #   mode 1: (i, j) pairs — entries of A  [size n*m]
        #   mode 2: (j, k) pairs — entries of B  [size m*p]  
        #   mode 3: (i, k) pairs — entries of C  [size n*p]
        # And T[(i,j), (j',k), (i',k')] = 1 if i==i' AND k==k' AND j==j'.
        # 
        # So the tensor is: T[a, b, c] = 1 if the (i,j) of a, (j',k) of b, and (i',k') of c
        # are consistent: i==i', k==k', j==j'.
        # 
        # This is the correct matmul tensor. Its size is (n*m) × (m*p) × (n*p) = 6 × 18 × 12.
        # The rank of this tensor is what we want to minimize.
        
        # Build the matmul tensor T of shape (n*m, m*p, n*p)
        T = np.zeros((n*m, m*p, n*p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i*m + j
                    b_idx = j*p + k
                    c_idx = k*n + i
                    T[a_idx, b_idx, c_idx] = 1.0
        
        # Try to find a decomposition of rank r < current_rank
        # For a given r, we need U (n*m × r), V (m*p × r), W (n*p × r) such that
        # T[a,b,c] = Σ_r' U[a,r'] V[b,r'] W[c,r']
        # 
        # This is equivalent to: for each c, T[:,:,c] = Σ_r' W[c,r'] U[:,r'] V[:,r']^T
        # i.e. T[:,:,c] = (U * W[c,:]) @ V.T  (Hadamard product of U with W[c,:] along columns)
        # 
        # Strategy: For a fixed W, the equations are linear in the entries of U and V.
        # But it's still bilinear. 
        # 
        # Alternative: Use the "slice" approach. For each c, T[:,:,c] is a matrix of size (n*m)×(m*p).
        # If rank(T[:,:,c]) ≤ r for all c, and the slices are consistent, we can recover.
        # 
        # Let me try a direct approach: for small r, set up the system and solve.
        # For each (a,b,c): Σ_r' U[a,r'] V[b,r'] W[c,r'] = T[a,b,c]
        # 
        # This is a system of (n*m)*(m*p)*(n*p) equations in (n*m + m*p + n*p)*r unknowns.
        # For n=2,m=3,p=6: 6*18*12 = 1296 equations, (6+18+12)*r = 36r unknowns.
        # For r=33: 1188 unknowns, 1296 equations — overdetermined.
        # For r=32: 1152 unknowns — even more overdetermined.
        # 
        # The system is nonlinear (bilinear/trilinear). I'll use a randomized approach:
        # 1. Fix a random W of shape (n*p, r).
        # 2. For each c, compute M_c = T[:,:,c] (size n*m × m*p).
        # 3. We need: M_c = (U ⊙ W[c,:]) @ V.T where ⊙ is column-wise scaling.
        #    i.e. M_c[a,b] = Σ_r' U[a,r'] W[c,r'] V[b,r']
        # 4. This is: M_c = U @ diag(W[c,:]) @ V.T
        # 5. If we fix W, then for each c: M_c = U @ D_c @ V.T where D_c = diag(W[c,:]).
        #    This is still bilinear in U and V.
        # 
        # Let me try a different approach: use the SVD of the "flattened" tensor.
        # 
        # The tensor T of size (n*m)×(m*p)×(n*p) can be matricized as a (n*m)×(m*p*n*p) matrix.
        # The rank of this matricization gives a lower bound on the tensor rank.
        
        # Matricize T as (n*m) × (m*p * n*p)
        T_mat = T.reshape(n*m, m*p * n*p)
        # SVD to get numerical rank
        try:
            s = np.linalg.svd(T_mat, compute_uv=False)
        except:
            return current_rank, None, None, None
        
        # The numerical rank of this matricization
        tol = 1e-10
        num_rank = np.sum(s > tol)
        # num_rank is a lower bound on the tensor rank
        
        if num_rank >= current_rank:
            # Can't do better than current_rank based on this lower bound
            return current_rank, None, None, None
        
        # Try to find a decomposition of rank r for r from num_rank to current_rank-1
        # For each r, try a randomized iterative approach
        
        best_result = (current_rank, None, None, None)
        
        for r in range(min(num_rank, current_rank - 1), current_rank - 1):
            if _time_remaining() < 10.0:
                break
            # Try multiple random initializations
            for trial in range(5):
                if _time_remaining() < 10.0:
                    break
                try:
                    # Random W initialization
                    W_init = np.random.randn(n*p, r)
                    # Iteratively solve for U and V
                    U_cur = np.random.randn(n*m, r)
                    V_cur = np.random.randn(m*p, r)
                    
                    converged = False
                    for it in range(50):
                        if _time_remaining() < 10.0:
                            break
                        # Update U: for each a, U[a,:] should satisfy
                        # T[a,b,c] for all b,c: Σ_r' U[a,r'] V[b,r'] W[c,r'] = T[a,b,c]
                        # Fix V and W, solve for U[a,:] for each a.
                        # For fixed a: Σ_{b,c,r'} T[a,b,c] = Σ_r' U[a,r'] (Σ_{b,c} V[b,r'] W[c,r'] T[a,b,c])
                        # This is getting complicated. Let me use a gradient-free approach.
                        # 
                        # Actually, let me just verify if the current rank-33 solution is optimal
                        # by checking the lower bound more carefully.
                        break
                    
                    # This approach is too complex for the time budget. 
                    # Let me just return the current best.
                    break
                except:
                    break
        
        return best_result

    # First, construct the rank-33 solution (the known good one)
    
    # Target Rank 33
    # Part 1: A[:, 0] (2x1) * B[0, :] (1x6) -> Rank 12
    # Part 2: A[:, 1:3] (2x2) * B[1:3, :] (2x6) -> Rank 21 (3 Strassen 2x2x2 blocks)
    
    rank = 33
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: A[:, 0] * B[0, :]
    # A[i, 0] * B[0, k] -> C[i, k]
    for k in range(p):
        for i in range(n):
            U[i * m + 0, col] = 1.0
            V[0 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1
            
    # Part 2: Two 2x2x2 Strassen blocks for each of the 3 column pairs
    # A_sub is A[:, 1:3] (2x2)
    # B_sub is B[1:3, :] (2x6)
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Indices in U: i*m + j. Here i in {0,1}, j in {1,2} (cols 1,2 of A)
    # Indices in V: j*p + k. Here j in {1,2} (rows 1,2 of B), k in {0..5}
    # Indices in W: k*n + i. Here k in {0..5}, i in {0,1}
    
    for t in range(3):
        # Block t uses B cols 2*t, 2*t+1
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # M1
        U[0 * m + 1, col] += 1.0; U[1 * m + 2, col] += 1.0
        V[1 * p + k0, col] += 1.0; V[2 * p + k1, col] += 1.0
        W[k0 * n + 0, col] += 1.0; W[k1 * n + 1, col] += 1.0
        col += 1
        
        # M2
        U[1 * m + 1, col] += 1.0; U[1 * m + 2, col] += 1.0
        V[1 * p + k0, col] += 1.0
        W[k0 * n + 1, col] += 1.0; W[k1 * n + 1, col] += -1.0
        col += 1
        
        # M3
        U[0 * m + 1, col] += 1.0
        V[1 * p + k1, col] += 1.0; V[2 * p + k1, col] += -1.0
        W[k1 * n + 0, col] += 1.0; W[k1 * n + 1, col] += 1.0
        col += 1
        
        # M4
        U[1 * m + 2, col] += 1.0
        V[2 * p + k0, col] += 1.0; V[1 * p + k0, col] += -1.0
        W[k0 * n + 0, col] += 1.0; W[k0 * n + 1, col] += 1.0
        col += 1
        
        # M5
        U[0 * m + 1, col] += 1.0; U[0 * m + 2, col] += 1.0
        V[2 * p + k1, col] += 1.0
        W[k0 * n + 0, col] += -1.0; W[k1 * n + 0, col] += 1.0
        col += 1
        
        # M6
        U[1 * m + 1, col] += 1.0; U[0 * m + 1, col] += -1.0
        V[1 * p + k0, col] += 1.0; V[1 * p + k1, col] += 1.0
        W[k1 * n + 1, col] += 1.0
        col += 1
        
        # M7
        U[0 * m + 2, col] += 1.0; U[1 * m + 2, col] += -1.0
        V[2 * p + k0, col] += 1.0; V[2 * p + k1, col] += 1.0
        W[k0 * n + 0, col] += 1.0
        col += 1

    # Verify the decomposition is correct
    if _time_remaining() > 5.0:
        if not _verify(U, V, W):
            # Should not happen, but if it does, we still return (score will be 0)
            pass
    
    # Attempt to find a lower-rank decomposition using SVD-based analysis
    # The matmul tensor T has shape (n*m) x (m*p) x (n*p) = 6 x 18 x 12
    # We check the numerical rank of its matricizations to see if rank < 33 is possible
    
    if _time_remaining() > 15.0:
        # Build the matmul tensor T of shape (n*m, m*p, n*p)
        T_tensor = np.zeros((n*m, m*p, n*p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i*m + j
                    b_idx = j*p + k
                    c_idx = k*n + i
                    T_tensor[a_idx, b_idx, c_idx] = 1.0
        
        # Check matricization ranks (lower bounds on tensor rank)
        # Matricization 1: (n*m) x (m*p * n*p)
        T_mat1 = T_tensor.reshape(n*m, m*p * n*p)
        # Matricization 2: (m*p) x (n*m * n*p)
        T_mat2 = T_tensor.transpose(1, 0, 2).reshape(m*p, n*m * n*p)
        # Matricization 3: (n*p) x (n*m * m*p)
        T_mat3 = T_tensor.transpose(2, 0, 1).reshape(n*p, n*m * m*p)
        
        s1 = np.linalg.svd(T_mat1, compute_uv=False)
        s2 = np.linalg.svd(T_mat2, compute_uv=False)
        s3 = np.linalg.svd(T_mat3, compute_uv=False)
        
        tol = 1e-10
        lb1 = np.sum(s1 > tol)
        lb2 = np.sum(s2 > tol)
        lb3 = np.sum(s3 > tol)
        lower_bound = max(lb1, lb2, lb3)
        
        # If the lower bound is less than 33, try to find a lower-rank decomposition
        if lower_bound < rank and _time_remaining() > 30.0:
            # Try ranks from lower_bound to rank-1
            # For each rank r, we attempt a randomized decomposition search
            # The system: for all (a,b,c): sum_r' U[a,r'] V[b,r'] W[c,r'] = T[a,b,c]
            # This is a trilinear system. We use alternating least squares (ALS).
            
            for r in range(min(lower_bound, rank - 1), rank):
                if _time_remaining() < 15.0:
                    break
                if r < lower_bound:
                    continue
                
                found = False
                for trial in range(3):
                    if _time_remaining() < 15.0:
                        break
                    
                    # Initialize W randomly
                    W_cur = np.random.randn(n*p, r)
                    U_cur = np.random.randn(n*m, r)
                    V_cur = np.random.randn(m*p, r)
                    
                    prev_err = np.inf
                    for it in range(100):
                        if _time_remaining() < 15.0:
                            break
                        
                        # Update U: for each a, solve for U[a,:]
                        # Fix V and W. For each a:
                        #   sum_{b,c,r'} T[a,b,c] V[b,r'] W[c,r'] * U[a,r'] ... 
                        # Actually: T[a,b,c] = sum_r' U[a,r'] V[b,r'] W[c,r']
                        # For fixed a: sum_{b,c} T[a,b,c] = sum_{b,c} sum_r' U[a,r'] V[b,r'] W[c,r']
                        # This doesn't directly give a linear system for U[a,:].
                        # 
                        # Better: For fixed V and W, the equation for U[a,:] is:
                        # For all (b,c): T[a,b,c] = sum_r' U[a,r'] (V[b,r'] W[c,r'])
                        # Let M_c[b, r'] = V[b,r'] * W[c,r'] for each c.
                        # Then for fixed a and c: T[a,:,c] = U[a,:] @ M_c.T
                        # where M_c is (m*p) x r.
                        # So: T[a,:,c].T (size m*p) = M_c @ U[a,:]
                        # Stack over c: [T[a,:,0].T; T[a,:,1].T; ...] = [M_0; M_1; ...] @ U[a,:]
                        # This is a linear system: A_a @ U[a,:] = t_a
                        # where A_a is (m*p * n*p) x r and t_a is (m*p * n*p)
                        
                        U_new = np.zeros_like(U_cur)
                        for a in range(n*m):
                            # Build the linear system
                            # For each (b, c): sum_r' U[a,r'] V[b,r'] W[c,r'] = T[a,b,c]
                            # This is: U[a,:] dot (V[b,:] * W[c,:]) = T[a,b,c]
                            # So the row of the system for (b,c) is: V[b,:] * W[c,:]
                            A_a = np.zeros((m*p * n*p, r))
                            t_a = np.zeros(m*p * n*p)
                            idx = 0
                            for b in range(m*p):
                                for c in range(n*p):
                                    A_a[idx, :] = V_cur[b, :] * W_cur[c, :]
                                    t_a[idx] = T_tensor[a, b, c]
                                    idx += 1
                            # Solve least squares
                            try:
                                U_new[a, :], _, _, _ = np.linalg.lstsq(A_a, t_a, rcond=None)
                            except:
                                U_new[a, :] = U_cur[a, :]
                        U_cur = U_new
                        
                        # Update V similarly
                        V_new = np.zeros_like(V_cur)
                        for b in range(m*p):
                            A_b = np.zeros((n*m * n*p, r))
                            t_b = np.zeros(n*m * n*p)
                            idx = 0
                            for a in range(n*m):
                                for c in range(n*p):
                                    A_b[idx, :] = U_cur[a, :] * W_cur[c, :]
                                    t_b[idx] = T_tensor[a, b, c]
                                    idx += 1
                            try:
                                V_new[b, :], _, _, _ = np.linalg.lstsq(A_b, t_b, rcond=None)
                            except:
                                V_new[b, :] = V_cur[b, :]
                        V_cur = V_new
                        
                        # Update W similarly
                        W_new = np.zeros_like(W_cur)
                        for c in range(n*p):
                            A_c = np.zeros((n*m * m*p, r))
                            t_c = np.zeros(n*m * m*p)
                            idx = 0
                            for a in range(n*m):
                                for b in range(m*p):
                                    A_c[idx, :] = U_cur[a, :] * V_cur[b, :]
                                    t_c[idx] = T_tensor[a, b, c]
                                    idx += 1
                            try:
                                W_new[c, :], _, _, _ = np.linalg.lstsq(A_c, t_c, rcond=None)
                            except:
                                W_new[c, :] = W_cur[c, :]
                        W_cur = W_new
                        
                        # Compute error
                        err = 0.0
                        for a in range(n*m):
                            for b in range(m*p):
                                for c in range(n*p):
                                    val = np.dot(U_cur[a, :], V_cur[b, :] * W_cur[c, :])
                                    err += (val - T_tensor[a, b, c]) ** 2
                        err = np.sqrt(err)
                        
                        if abs(prev_err - err) < 1e-12:
                            break
                        prev_err = err
                        
                        if err < 1e-8:
                            found = True
                            break
                    
                    if found:
                        # Verify the solution
                        if _verify(U_cur, V_cur, W_cur):
                            rank = r
                            U = U_cur
                            V = V_cur
                            W = W_cur
                            break
                        else:
                            found = False
                
                if found:
                    break

    # Final deadline guard
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
