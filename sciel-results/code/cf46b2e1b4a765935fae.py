"""Find a rank-minimal exact tensor decomposition for 5x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = start_time + 115.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 125
    
    # Build the schoolbook decomposition
    U = np.zeros((n * m, rank_schoolbook), dtype=np.float64)
    V = np.zeros((m * p, rank_schoolbook), dtype=np.float64)
    W = np.zeros((p * n, rank_schoolbook), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_schoolbook
    
    # Try to find a lower rank decomposition using structured random search
    # The key insight: for the all-ones tensor T[i,j,k]=1, we can try to find
    # decompositions with rank < 125 by exploiting the bilinear structure.
    
    # Strategy: Use the fact that T[i,j,k] = 1 for all i,j,k.
    # A rank-r decomposition satisfies:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1  for all i,j,k
    #
    # We can try:
    # 1. Random initialization with constraints
    # 2. Gradient descent on the residual
    # 3. Structured patterns (e.g., using Hadamard-like matrices)
    
    np.random.seed(42)
    
    # Try several candidate ranks
    # For 5x5x5, known upper bounds:
    # - Schoolbook: 125
    # - Winograd-style: potentially lower
    # - Best known: around 70-80 (but hard to find automatically)
    
    # Let's try a structured approach: use the fact that the tensor is all-ones
    # and try to find a low-rank approximation that is exact.
    
    # Approach: Try rank r and see if we can find U, V, W such that the
    # decomposition holds exactly. We'll use a simple iterative method.
    
    def try_rank(r, max_iter=500):
        """Try to find a rank-r decomposition using iterative refinement."""
        if time.time() > DEADLINE - 20:
            return None
        
        # Initialize with random values
        U_r = np.random.randn(n * m, r)
        V_r = np.random.randn(m * p, r)
        W_r = np.random.randn(p * n, r)
        
        # Normalize to avoid numerical issues
        U_r /= np.linalg.norm(U_r) + 1e-10
        V_r /= np.linalg.norm(V_r) + 1e-10
        W_r /= np.linalg.norm(W_r) + 1e-10
        
        # We want: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # This is a tensor equation. We can try to minimize the residual.
        
        # For efficiency, let's use a simpler approach:
        # If we fix U and V, we can solve for W (or vice versa).
        
        # Actually, let's try a different approach: use the structure of the problem.
        # For the all-ones tensor, one known approach is to use:
        #   U[i*m+j, c] = a_i * b_j
        #   V[j*p+k, c] = b_j * c_k  
        #   W[k*n+i, c] = c_k * a_i
        # but this doesn't quite work for the product to be 1.
        
        # Let's try a more direct approach: assume a structured form.
        # For rank r, let's try:
        #   U[i*m+j, c] = u_i * v_j * w_c  (separable)
        #   V[j*p+k, c] = v_j * x_k * w_c
        #   W[k*n+i, c] = x_k * u_i * w_c
        # Then: sum_c u_i * v_j * w_c * v_j * x_k * w_c * x_k * u_i * w_c
        #      = u_i^2 * x_k^2 * sum_c (v_j * w_c)^3
        # This needs to equal 1 for all i, j, k, which is very restrictive.
        
        # Let's try a different structured approach.
        # Use the fact that we can write 1 = sum_c f_c(i,j) * g_c(j,k) * h_c(k,i)
        
        # For now, let's just try random search with a few iterations.
        # This is unlikely to find a good solution quickly, but it's worth trying.
        
        for _ in range(max_iter):
            if time.time() > DEADLINE - 20:
                return None
            
            # Compute the residual
            residual = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_r[i * m + j, :], V_r[j * p + k, :])
                        total = np.dot(val, W_r[k * n + i, :])
                        residual += (total - 1.0) ** 2
            
            if residual < 1e-10:
                return U_r, V_r, W_r
            
            # Gradient step (simplified)
            # This is a complex optimization, so we'll just do random perturbations
            U_r += 0.1 * np.random.randn(n * m, r)
            V_r += 0.1 * np.random.randn(m * p, r)
            W_r += 0.1 * np.random.randn(p * n, r)
        
        return None
    
    # Try a few ranks
    for target_rank in [124, 120, 115, 110, 105, 100, 95, 90, 85, 80]:
        if time.time() > DEADLINE - 30:
            break
        
        result = try_rank(target_rank)
        if result is not None:
            U_r, V_r, W_r = result
            best_U, best_V, best_W = U_r, V_r, W_r
            best_rank = target_rank
            break
    
    # Convert to list-of-lists format
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
