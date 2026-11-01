"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s safety margin
    
    n, m, p = DIMENSIONS
    rank_target = n * m * p  # Start with schoolbook rank
    
    # Generate all rank-1 terms for the tensor
    # Term (i, j, k) has:
    #   U[i*m+j, col] = 1
    #   V[j*p+k, col] = 1
    #   W[k*n+i, col] = 1
    
    all_terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                all_terms.append((i, j, k))
    
    num_terms = len(all_terms)
    
    # Build the tensor as a matrix where each column is a rank-1 term
    # The tensor T[i*m+j, j*p+k, k*n+i] = sum over terms of U*V*W
    # We can represent each term as a vector in R^(n*m * m*p * p*n) but that's huge
    # Instead, use the fact that the tensor is determined by its slices
    
    # Better approach: Use the structure of matrix multiplication
    # T = sum_c U_c ⊗ V_c ⊗ W_c
    # where U_c is in R^(n*m), V_c in R^(m*p), W_c in R^(p*n)
    
    # For rank-1 term (i,j,k):
    #   u = e_{i*m+j} in R^(n*m)
    #   v = e_{j*p+k} in R^(m*p)  
    #   w = e_{k*n+i} in R^(p*n)
    
    # The tensor equation is: for all (a,b,c) indices,
    # T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
    
    # This is equivalent to: the tensor T as a 3D array equals sum of outer products
    
    # Let's use a different approach: random linear combinations
    # Generate random vectors and check if we can express the tensor with fewer terms
    
    # First, let's get the schoolbook solution as a fallback
    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    schoolbook = get_schoolbook()
    
    # Now try to find a lower rank decomposition using randomization
    # Key idea: The tensor for matrix multiplication has structure
    # For 2x2x10, we can think of it as 5 blocks of 2x2x2
    
    # Each 2x2x2 block has rank at most 7 (Strassen)
    # But the blocks share structure
    
    # Let's try a direct approach: use SVD-like method on the tensor
    
    # Represent the tensor as a matrix by flattening two indices
    # T[i*m+j, j*p+k, k*n+i]
    # Flatten first two: T_flat[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    
    # Build the tensor as a 3D numpy array
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Now we want to find rank-1 decompositions: T = sum_t u_t ⊗ v_t ⊗ w_t
    # This is equivalent to finding the tensor rank
    
    # Use a random subspace approach:
    # 1. Generate random vectors r1 in R^(n*m), r2 in R^(m*p), r3 in R^(p*n)
    # 2. The contraction T[r1, r2, r3] = sum_{a,b,c} T[a,b,c] * r1[a] * r2[b] * r3[c]
    #    = sum_t (r1·u_t) * (r2·v_t) * (r3·w_t)
    
    # For a given rank-r decomposition, this contraction is a degree-3 polynomial
    # in the random vectors that factors as sum of r products
    
    # Alternative: Use the fact that for matrix multiplication,
    # we can use the block structure
    
    # Let's try: express 2x2x10 as a sum of 2x2x2 blocks
    # Block b (b=0..4): A[i,j] * B[j, b*2+k] for k=0,1
    # Each block is a 2x2x2 tensor
    
    # The full tensor is: T = sum_{b=0}^{4} T_b
    # where T_b[i*m+j, j*p+b*2+k, (b*2+k)*n+i] = 1
    
    # If we can find rank-7 decompositions for each T_b, total rank <= 35
    # But we want to exploit shared structure
    
    # Let's try a different approach: use integer linear algebra
    # Generate random integer combinations and use Gaussian elimination
    
    # Actually, let's use a simpler but effective approach:
    # Try to find linear dependencies among the 40 rank-1 terms
    
    # Each term t = (i,j,k) defines a rank-1 tensor
    # We want to find if sum_{t in S} c_t * T_t = 0 for some non-trivial coefficients
    
    # This is equivalent to: the 40 vectors (in the tensor space) are linearly dependent
    
    # The tensor space has dimension (n*m)*(m*p)*(p*n) = 4*20*20 = 1600
    # So 40 vectors in 1600-dim space are almost certainly independent
    # But the tensor has special structure!
    
    # Key insight: The tensor T satisfies certain algebraic identities
    # For matrix multiplication, there are known rank bounds
    
    # Let's just try to verify if rank 34 or lower is possible by random search
    
    # Approach: Assume rank r, generate random U, V, W with r columns
    # Check if U ⊗ V ⊗ W can represent T
    
    # This is a system of equations: for all (a,b,c),
    # T[a,b,c] = sum_{t=1}^r U[a,t] * V[b,t] * W[c,t]
    
    # This is a polynomial system in the entries of U, V, W
    
    # For small r, we can try random initialization and gradient descent
    
    # But first, let's check: what's the known rank for 2x2x10?
    # It's at most 35 (5 blocks of 2x2x2, each rank 7)
    # It's at least 2*2*10 = 40? No, that's the trivial bound
    # Actually the trivial bound is n*m*p = 40
    
    # Wait, the schoolbook rank is n*m*p = 40, but we claimed 35 above
    # Let me re-examine
    
    # Schoolbook: for each (i,j,k), one term. That's n*m*p = 40 terms
    # So schoolbook rank is 40, not 35
    
    # The current program uses rank = n*m*p = 40
    
    # So we're starting from rank 40 and want to go lower
    
    # Known results: 
    # - 2x2x2 has rank 7 (Strassen)
    # - 2x2x10 could potentially have rank less than 40
    
    # Let's try to use the block structure more cleverly
    
    # The 2x2x10 tensor can be seen as 5 copies of 2x2x2
    # But they share the A matrix indices
    
    # Let's try: use Strassen for each block, but look for shared terms
    
    # Actually, let's just do a random search for lower rank
    
    def try_random_rank(r, num_trials=100):
        """Try to find a rank-r decomposition using random initialization"""
        for trial in range(num_trials):
            if time.time() > deadline - 10:
                break
            
            # Random initialization with small integers
            U = np.random.randint(-2, 3, size=(n*m, r))
            V = np.random.randint(-2, 3, size=(m*p, r))
            W = np.random.randint(-2, 3, size=(p*n, r))
            
            # Compute the tensor from this decomposition
            T_recon = np.zeros((n*m, m*p, p*n))
            for t in range(r):
                T_recon += np.outer(np.outer(U[:, t], V[:, t]), W[:, t])
            
            # Check if T_recon matches T
            if np.allclose(T_recon, T):
                # Found a valid decomposition!
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from 35 down to 30
    for r in range(35, 29, -1):
        if time.time() > deadline - 20:
            break
        
        result = try_random_rank(r, num_trials=50)
        if result is not None:
            return result
    
    # If we couldn't find a lower rank, return schoolbook
    return schoolbook
# EVOLVE-BLOCK-END
