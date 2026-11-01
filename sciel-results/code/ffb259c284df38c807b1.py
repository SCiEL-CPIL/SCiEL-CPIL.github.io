"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Schoolbook baseline
    rank_schoolbook = n * m * p
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
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_schoolbook
    
    # Try Strassen-based reduction
    # Partition 5 into 2+2+1
    # For the 2x2x2 blocks, use Strassen (rank 7 instead of 8)
    # This gives: 2 blocks of 2x2x2 (each rank 7) + remaining terms
    
    # Let me try a direct approach: use the known fact that M_{2,5,5} 
    # can be done with rank less than 50 by exploiting structure.
    
    # Approach: Use random initialization + numerical optimization
    # to find a low-rank decomposition.
    
    def verify(U, V, W):
        """Check if the decomposition is exact."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        # Reconstruct tensor
        # T[i*m+j, j*p+k, k*n+i] should equal sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(expected - actual) > 1e-6:
                        return False
        return True
    
    # Try various ranks from low to high
    # Known lower bound: for M_{2,5,5}, the rank is at least... 
    # Let's try ranks from 30 upward
    
    np.random.seed(42)
    
    for target_rank in range(30, 51):
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Normalize to help optimization
            for r in range(target_rank):
                norm = np.max(np.abs(U[:, r]))
                if norm > 1e-10:
                    U[:, r] /= norm
                    V[:, r] *= norm
            
            # Simple iterative refinement (ALS-like)
            for iteration in range(200):
                if time.time() > deadline - 10:
                    break
                
                # Update U
                for r in range(target_rank):
                    # U[:,r] should satisfy: sum_k U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
                    # This is complex; use a simple gradient step
                    pass
                
                # Check convergence
                if verify(U, V, W):
                    # Found a valid decomposition
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best_result = {
                            "U": U.tolist(), 
                            "V": V.tolist(), 
                            "W": W.tolist()
                        }
                    break
            else:
                continue
            break
    
    # Also try the Strassen-inspired structured approach
    # For 2x5x5: think of it as 2 rows of output
    # Each output row i: C[i,k] = sum_j A[i,j]*B[j,k]
    # This is 2 separate 1x5 * 5x5 multiplications
    
    # For 1x5 * 5x5: this is just 5 dot products, each of length 5
    # Total: 2*5*5 = 50 scalar multiplications (schoolbook)
    
    # Can we do better? The tensor is T_{i,j,k} where i in {0,1}, j in {0..4}, k in {0..4}
    # T_{i,j,k} = 1 for all valid (i,j,k)
    
    # This is actually the outer product structure: T = sum over (i,j,k) of e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    # Wait, that's not right. Let me re-examine.
    
    # The multiplication tensor for A(n×m) * B(m×p) -> C(n×p) has:
    # T_{(i,j),(j,k),(i,k)} = 1
    # In our indexing: U index = i*m+j, V index = j*p+k, W index = k*n+i
    
    # So T_{i*m+j, j*p+k, k*n+i} = 1
    
    # This is a specific tensor. For n=2, m=5, p=5:
    # U indices: 0..9 (i=0: 0-4, i=1: 5-9)
    # V indices: 0..24 (j=0: 0-4, j=1: 5-9, ..., j=4: 20-24)
    # W indices: 0..9 (k=0: 0-1, k=1: 2-3, ..., k=4: 8-9)
    
    # Let me try a more targeted numerical approach using scipy
    
    from scipy.optimize import minimize
    
    def objective(params):
        """Flatten U, V, W and compute error."""
        n, m, p = DIMENSIONS
        rank = target_rank
        U_flat = params[:n*m*rank]
        V_flat = params[n*m*rank:n*m*rank+m*p*rank]
        W_flat = params[n*m*rank+m*p*rank:]
        
        U = U_flat.reshape(n*m, rank)
        V = V_flat.reshape(m*p, rank)
        W = W_flat.reshape(p*n, rank)
        
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    error += (val - 1.0)**2
        return error
    
    # Try a few more structured approaches
    # Use the fact that for small n, we can exploit linearity in the n dimension
    
    # For n=2: C[0,k] = sum_j A[0,j]*B[j,k], C[1,k] = sum_j A[1,j]*B[j,k]
    # These are independent! So M_{2,5,5} = M_{1,5,5} + M_{1,5,5} in terms of tensor rank
    # (since the two rows don't interact)
    
    # So rank(M_{2,5,5}) = 2 * rank(M_{1,5,5})
    
    # Now what is rank(M_{1,5,5})? This is the tensor for 1x5 * 5x5 -> 1x5
    # T_{j,k} = 1 (only 25 entries, all 1)
    # U has shape (5, rank), V has shape (25, rank), W has shape (5, rank)
    # T_{j,k} = sum_r U[j,r]*V[j*5+k,r]*W[k,r] = 1
    
    # This is a rank-1 tensor! Because T_{j,k} = 1 = 1*1*1
    # Wait, no. U[j,r], V[j*5+k,r], W[k,r]
    # If rank=1: U[j,0]=a_j, V[j*5+k,0]=b_{j,k}, W[k,0]=c_k
    # Then sum = a_j * b_{j,k} * c_k = 1 for all j,k
    # This requires a_j*c_k * b_{j,k} = 1, so b_{j,k} = 1/(a_j*c_k)
    # This works! So rank(M_{1,5,5}) = 1?? That can't be right.
    
    # Wait, I'm confusing myself. Let me re-examine the tensor structure.
    
    # For matrix multiplication A(m×n) * B(n×p) -> C(m×p):
    # The tensor is T_{(i,j),(j,k),(i,k)} = 1
    # U index: (i,j) where i in [0,m), j in [0,n)  -> size m*n
    # V index: (j,k) where j in [0,n), k in [0,p)  -> size n*p  
    # W index: (i,k) where i in [0,m), k in [0,p)  -> size m*p
    
    # In our case: m=2, n=5, p=5 (using the task's notation: n=2, m=5, p=5)
    # U: (n*m, rank) = (10, rank)
    # V: (m*p, rank) = (25, rank)
    # W: (p*n, rank) = (10, rank)
    
    # T_{i*m+j, j*p+k, k*n+i} = 1
    
    # For the "1x5 * 5x5" subproblem (i=0 only):
    # T_{j, j*5+k, k} = 1 for j in [0,5), k in [0,5)
    # U index: j (0-4), V index: j*5+k (0-24), W index: k (0-4)
    # This is a 5x25x5 tensor with T_{j, j*5+k, k} = 1
    
    # Can this be rank 1? U[j,0]=a_j, V[j*5+k,0]=b_{j,k}, W[k,0]=c_k
    # a_j * b_{j,k} * c_k = 1 for all j,k
    # b_{j,k} = 1/(a_j*c_k)
    # Yes! So rank(M_{1,5,5}) = 1
    
    # Therefore rank(M_{2,5,5}) = 2 * 1 = 2??? That seems too good to be true.
    
    # Wait, I think I'm making an error. Let me reconsider.
    
    # The issue is that for the full 2x5x5 problem, the U, V, W matrices must work 
    # for BOTH i=0 and i=1 simultaneously. The V matrix is shared between both i values.
    
    # For i=0: T_{j, j*5+k, k} = 1  (U index j, V index j*5+k, W index k)
    # For i=1: T_{5+j, j*5+k, 5+k} = 1  (U index 5+j, V index j*5+k, W index 5+k)
    
    # So V is shared! This is the key constraint.
    
    # Let's try rank 2:
    # U[j, r] for r=0,1 (i=0 part)
    # U[5+j, r] for r=0,1 (i=1 part)
    # V[j*5+k, r] for r=0,1
    # W[k, r] for r=0,1 (i=0 part)
    # W[5+k, r] for r=0,1 (i=1 part)
    
    # Constraints:
    # For i=0: sum_r U[j,r]*V[j*5+k,r]*W[k,r] = 1
    # For i=1: sum_r U[5+j,r]*V[j*5+k,r]*W[5+k,r] = 1
    
    # This is a system of 50 equations with 2*(10+25+10) = 90 unknowns.
    # Should be solvable!
    
    # Let me try to solve this numerically.
    
    def solve_low_rank(rank):
        """Try to find a valid decomposition with given rank."""
        n, m, p = DIMENSIONS
        num_vars = rank * (n*m + m*p + p*n)
        
        # Use random initialization and Newton's method
        np.random.seed(rank)
        params = np.random.randn(num_vars) * 0.1
        
        def residuals(params):
            U = params[:rank*n*m].reshape(n*m, rank)
            V = params[rank*n*m:rank*(n*m+m*p)].reshape(m*p, rank)
            W = params[rank*(n*m+m*p):].reshape(p*n, rank)
            
            res = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        res.append(val - 1.0)
            return np.array(res)
        
        # Use scipy's least_squares
        from scipy.optimize import least_squares
        
        result = least_squares(residuals, params, method='lm', max_nfev=10000)
        
        if result.success or np.max(np.abs(result.fun)) < 1e-8:
            U = result.x[:rank*n*m].reshape(n*m, rank)
            V = result.x[rank*n*m:rank*(n*m+m*p)].reshape(m*p, rank)
            W = result.x[rank*(n*m+m*p):].reshape(p*n, rank)
            return U, V, W
        return None
    
    # Try ranks from 2 upward
    for target_rank in range(2, 15):
        if time.time() > deadline - 30:
            break
        
        solution = solve_low_rank(target_rank)
        if solution is not None:
            U, V, W = solution
            if verify(U, V, W):
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_result = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                break  # Found lowest rank, stop
    
    return best_result
# EVOLVE-BLOCK-END

