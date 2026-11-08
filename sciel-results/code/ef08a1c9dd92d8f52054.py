import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    N, M, P = n, m, p
    target_rank = 56
    
    # Try to find a lower rank decomposition
    # Start with a known lower bound or a guess
    # For 2x4x7, rank 32 is a plausible target
    ranks_to_try = [32, 36, 40, 44, 48, 52, 56]
    
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin
    
    for R in ranks_to_try:
        if time.time() > deadline:
            break
            
        # Try to find a decomposition of rank R
        # We use a randomized approach: generate random rank-1 tensors and solve for coefficients
        # The multiplication tensor T has entries T[i, j, k] = 1 if i < n, j < m, k < p and the multiplication (A * B)[i, k] = sum_j A[i, j] * B[j, k]
        # The tensor is defined by T[i, j, k] = 1 if the term A[i, j] * B[j, k] contributes to C[i, k]
        # Actually, the tensor is T[i, j, k] = delta_{i, i'} delta_{j, j'} delta_{k, k'} ... no.
        # The multiplication tensor is T such that (A * B)[i, k] = sum_{j} A[i, j] B[j, k] = sum_{i', j', k'} T[i', j', k'] A[i', j'] B[j', k']
        # So T[i, j, k] = 1 if i=i', j=j', k=k' and the indices match.
        # Specifically, T is a tensor of shape (n*p, m, n*p)? No.
        # The tensor is in the space of (n*m) x (m*p) x (p*n) ? No.
        # The standard tensor for matrix multiplication is T in U^* \otimes V^* \otimes W where U is the space of A, V is the space of B, W is the space of C.
        # dim(U) = n*m = 8, dim(V) = m*p = 28, dim(W) = p*n = 14.
        # T[i, j, k] = 1 if i corresponds to (i_A, j_A), j corresponds to (j_B, k_B), k corresponds to (i_C, k_C) and i_A = i_C, j_A = j_B, k_B = k_C.
        
        # Let's construct the target tensor T of shape (n*m, m*p, p*n)
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_A = i*m + j
                    idx_B = j*p + k
                    idx_C = k*n + i
                    T[idx_A, idx_B, idx_C] = 1
        
        # We want to find U, V, W of shapes (n*m, R), (m*p, R), (p*n, R) such that
        # T = sum_{r=1}^R U[:, r] \otimes V[:, r] \otimes W[:, r]
        # This is equivalent to finding a rank-R decomposition of the tensor T.
        
        # Randomized search:
        # Generate random matrices U, V, W and solve for the coefficients?
        # Or generate random rank-1 tensors and check if they span the tensor space?
        
        # A better approach: use a known algorithm for small tensor rank.
        # Since the dimensions are small, we can use a greedy algorithm or a random search with integer coefficients.
        
        # Let's try a random search with small integer coefficients.
        # We will try to find a set of R rank-1 tensors that sum to T.
        # This is a hard problem, but for small R and small dimensions, it might be feasible.
        
        # Alternatively, we can use the fact that the rank is at most the product of the dimensions.
        # We already have rank 56. Can we do better?
        
        # Let's try to find a decomposition of rank 32 using a random basis.
        # We will generate R random vectors u_r, v_r, w_r and solve for the coefficients c_r such that
        # sum c_r u_r \otimes v_r \otimes w_r = T
        # This is a linear system of size (n*m)*(m*p)*(p*n) = 8*28*14 = 3136 equations and R unknowns.
        # If R < 3136, the system is underdetermined. We can find a solution if T is in the span of the random tensors.
        # But we want an exact solution with small coefficients.
        
        # A simpler approach: use the schoolbook algorithm and try to combine terms.
        # For example, if we have two terms that differ only in one index, we might be able to combine them.
        # But this is not straightforward.
        
        # Given the time limit, we will return the schoolbook algorithm if we don't find a better one quickly.
        # However, we can try a few random attempts.
        
        # Let's try a random attempt with R=32.
        np.random.seed(42)
        for _ in range(100):  # Try 100 random attempts
            if time.time() > deadline:
                break
                
            U_rand = np.random.randint(-1, 2, size=(n*m, R))
            V_rand = np.random.randint(-1, 2, size=(m*p, R))
            W_rand = np.random.randint(-1, 2, size=(p*n, R))
            
            # Construct the tensor from the random decomposition
            T_rand = np.zeros((n*m, m*p, p*n))
            for r in range(R):
                T_rand += np.einsum('i,j,k->ijk', U_rand[:, r], V_rand[:, r], W_rand[:, r])
            
            # Check if T_rand == T
            if np.allclose(T_rand, T):
                # We found a decomposition
                return {
                    "U": U_rand.tolist(),
                    "V": V_rand.tolist(),
                    "W": W_rand.tolist()
                }
        
        # If no random decomposition found, try the next rank
        continue
    
    # If no better decomposition found, return the schoolbook algorithm
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
