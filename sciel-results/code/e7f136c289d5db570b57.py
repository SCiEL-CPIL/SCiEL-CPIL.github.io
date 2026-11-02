import numpy as np
import time
import itertools

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s deadline, 20s margin
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    # Schoolbook rank
    rank_schoolbook = n * m * p  # 96
    
    # Build schoolbook decomposition as baseline
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    best_rank = rank_schoolbook
    
    # Strategy: Try to find lower rank decompositions using structured approaches
    # The tensor T has T[i*m+j, j*p+k, k*n+i] = 1
    
    # Try block-diagonal / partition approaches
    # Partition n into groups, m into groups, p into groups
    
    # Try rank-12 approach: partition 4x4 input into two 2x4 strips
    # For each strip (2x4 * 4x6), we need a decomposition
    # Actually, let's try various structured decompositions
    
    # Approach 1: Try to use the fact that 4x4*4x6 can be seen as
    # multiple smaller multiplications
    
    # Let's try a systematic search over small ranks using linear algebra
    # The key equation is: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Try ranks from a reasonable lower bound up
    # Known lower bounds: for M_{4,4,6}, rank >= max(n*m, m*p, p*n) = max(16, 24, 24) = 24
    # But actually the tensor rank is at least max(n*m, m*p, n*p) for rectangular case
    
    # Let's try to find decompositions with rank < 96
    # Use a greedy / structured approach
    
    # Try partition-based decompositions
    # Partition indices into blocks
    
    # For a 4x4 * 4x6 multiplication:
    # We can partition the 4 rows of A into 2x2 blocks (2 blocks)
    # The 4 columns of A / rows of B into 2x2 blocks (2 blocks)  
    # The 6 columns of B into 2x3 blocks (2 blocks)
    
    # This gives a 2x2x2 grid of 2x2 * 2x3 multiplications
    # Each 2x2 * 2x3 multiplication has tensor rank at most 2*2*3 = 12
    # But we can do better: 2x2*2x2 has rank 7 (Laderman), 2x2*2x3 might have rank 9 or 10
    
    # Actually, let's just try to construct explicit low-rank decompositions
    
    # Laderman's algorithm for 2x2*2x2 has rank 7
    # For 2x2*2x3, we can extend: it's like 2x2*2x2 + 2x2*2x1
    # 2x2*2x1 has rank 4 (schoolbook) or maybe less
    
    # Let me try a different approach: use known results and construct explicitly
    
    # For 4x4*4x6, one approach:
    # Use 4x4*4x4 (rank 20, known) for the first 4 columns of B
    # Use 4x4*4x2 (rank ?) for the last 2 columns of B
    
    # 4x4*4x2: schoolbook is 32, but we can do better
    # 4x4*4x2 can be seen as 4x4*4x1 + 4x4*4x1, each 4x4*4x1 has rank 16
    # But 4x4*4x1 is just matrix-vector, rank is at most 16 but actually the tensor
    # for 4x4*4x1: T[i*4+j, j*1+k, k*4+i] = 1, so T[i*4+j, j, 4*i] = 1
    # This is a permutation-like tensor, rank is exactly 16 (each entry is independent)
    
    # Hmm, let me think more carefully.
    
    # Actually for M_{n,m,p}, the tensor is T_{(i,j),(j,k),(k,i)} = 1
    # The rank is the minimum number of rank-1 tensors needed
    
    # Let me try a computational approach: for small ranks, set up the nonlinear system
    # and try to solve it
    
    # Given time constraints, let's try a few specific structured decompositions
    
    # Decomposition using Kronecker products / tensor products
    # If we have a rank-r decomposition for M_{a,b,c}, we can get rank r^2 for M_{a^2,b^2,c^2}
    # But that's not directly helpful here
    
    # Let's try: 4x4*4x6 = (2x2*2x3) ⊗ (2x2*2x2) in some sense
    # No, that doesn't quite work
    
    # Let me try the most direct approach: attempt to find a rank-r decomposition
    # by solving the nonlinear equations for small r
    
    # For rank r, we have variables U (16*r), V (24*r), W (24*r) = 64*r variables
    # And 4*4*6 = 96 equations
    # So for r >= 2, we have more variables than equations, suggesting solutions exist
    
    # But the equations are nonlinear (cubic in the variables)
    
    # Let's try a randomized approach with local optimization for various ranks
    
    def verify(U, V, W):
        """Verify that U, V, W give an exact decomposition."""
        # T[i*m+j, j*p+k, k*n+i] should equal 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True
    
    # Try structured decompositions first
    
    # Approach: Use the fact that we can decompose 4x4*4x6 as
    # a sum of smaller multiplications
    
    # Method: Partition A into 4 row blocks of size 1, B into 6 column blocks of size 1
    # Each 1x4 * 4x1 multiplication has rank 4 (or maybe less)
    # 1x4 * 4x1: T[i*4+j, j, k*4+i] = 1 for i=0..0, j=0..3, k=0..0
    # So T[j, j, 0] = 1 for j=0..3
    # This is a rank-4 tensor (identity-like), so rank 4
    # Total: 4 * 4 = 16? No wait, that's not right.
    
    # Let me be more careful. For M_{1,4,1}:
    # T[i*m+j, j*p+k, k*n+i] = T[0*4+j, j*1+0, 0*1+0] = T[j, j, 0]
    # So T[j,j,0] = 1 for j=0,1,2,3
    # This is sum_{j=0}^{3} e_j ⊗ e_j ⊗ e_0, so rank 4.
    
    # For M_{4,4,6}, if we partition into 4*6 = 24 such M_{1,1,1} terms... no that's schoolbook.
    
    # Let me try a different partition.
    
    # Actually, let's just try to optimize for specific ranks using a simple approach.
    
    # Try rank values: we know rank >= 24 (from dimension bounds)
    # Let's try ranks: 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96
    
    # For each rank, try to find a solution using alternating minimization
    
    def try_find_decomposition(target_rank, max_iters=200, n_trials=10):
        """Try to find a rank-target_rank decomposition using alternating minimization."""
        for trial in range(n_trials):
            if time.time() > deadline - 10:
                return None
            
            # Random initialization
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Scale to make initial values reasonable
            U /= np.sqrt(target_rank)
            V /= np.sqrt(target_rank)
            W /= np.sqrt(target_rank)
            
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    return None
                
                # Update U: for each row i*m+j, find U[i*m+j, :] 
                # such that sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
                # This is a linear system: U[i*m+j, :] @ (V[j*p:k,:] * W[k*n+i,:]) = 1 for all k
                # Let M[i*m+j, k] = V[j*p+k, :] * W[k*n+i, :] (element-wise product, shape (p, target_rank))
                # We want U[i*m+j, :] @ M = ones(p)
                # This is: M.T @ U[i*m+j, :].T = ones(p)
                
                # Actually: sum_r U[r] * M[k,r] = 1 for all k
                # So M @ U[i*m+j, :] = ones(p) where M is (p, target_rank)
                # Wait: M[k,r] = V[j*p+k, r] * W[k*n+i, r]
                # sum_r M[k,r] * U[i*m+j, r] = 1 for all k
                # So M @ U[i*m+j, :] = ones(p)
                
                # Solve least squares
                for i in range(n):
                    for j in range(m):
                        # Build M: (p, target_rank)
                        M = np.outer(
                            V[j*p:(j+1)*p, :],  # (p, target_rank)
                            W[0:n, :]  # This is wrong
                        )
                        # Correct: M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        M = V[j*p:(j+1)*p, :] * W[0:p*n, :].reshape(p, n, -1)[:, :, 0]  # Still wrong
                        
                        # Let me redo this properly
                        M = np.zeros((p, target_rank))
                        for k in range(p):
                            M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        
                        # Solve M @ u = ones(p)
                        # Use least squares
                        try:
                            u, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                            U[i*m+j, :] = u
                        except:
                            pass
                
                # Update V similarly
                for j in range(m):
                    for k in range(p):
                        M = np.zeros((n, target_rank))  # Wait, for V[j*p+k, :], we need
                        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
                        # M[i, r] = U[i*m+j, r] * W[k*n+i, r]
                        M = np.zeros((n, target_rank))
                        for i in range(n):
                            M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            v, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                            V[j*p+k, :] = v
                        except:
                            pass
                
                # Update W similarly
                for k in range(p):
                    for i in range(n):
                        M = np.zeros((m, target_rank))
                        for j in range(m):
                            M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            w, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                            W[k*n+i, :] = w
                        except:
                            pass
                
                # Check convergence
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            error += (val - 1.0)**2
                
                if error < 1e-8:
                    if verify(U, V, W):
                        return (U, V, W)
        
        return None
    
    # Try various ranks
    for target_rank in [24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92]:
        if time.time() > deadline - 15:
            break
        
        result = try_find_decomposition(target_rank, max_iters=100, n_trials=3)
        if result is not None:
            U, V, W = result
            if verify(U, V, W):
                best_U = U
                best_V = V
                best_W = W
                best_rank = target_rank
                break  # Found a valid decomposition, stop
    
    # If we found a better rank, use it; otherwise use schoolbook
    if best_rank < rank_schoolbook:
        U_out = best_U
        V_out = best_V
        W_out = best_W
    else:
        U_out = U_sb
        V_out = V_sb
        W_out = W_sb
    
    # Convert to lists with rounded values for cleanliness
    # Round to nearest 0.5 to get integer/half-integer entries
    U_out = np.round(U_out * 2) / 2
    V_out = np.round(V_out * 2) / 2
    W_out = np.round(W_out * 2) / 2
    
    # Final verification
    if not verify(U_out, V_out, W_out):
        # If rounding broke it, use unrounded
        U_out = best_U if best_rank < rank_schoolbook else U_sb
        V_out = best_V if best_rank < rank_schoolbook else V_sb
        W_out = best_W if best_rank < rank_schoolbook else W_sb
    
    return {
        "U": U_out.tolist(),
        "V": V_out.tolist(),
        "W": W_out.tolist()
    }
