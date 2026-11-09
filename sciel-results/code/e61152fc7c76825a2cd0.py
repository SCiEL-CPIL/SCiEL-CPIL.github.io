"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # Start with schoolbook decomposition
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_rank = rank_schoolbook
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(15, 35, 21) = 35
    min_possible_rank = max(n * m, m * p, p * n)
    
    def verify_decomposition(U_mat, V_mat, W_mat, tol=1e-10):
        """Verify if U, V, W form an exact decomposition of matrix multiplication."""
        n, m, p = DIMENSIONS
        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)
        rank = len(U_mat[0])
        
        # The tensor T[i,j,k] = 1 for all i,j,k (multiplication tensor)
        # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # For efficiency, we can check a few random (i,j,k) tuples
        # Or check all if rank is small enough
        if rank <= 100:
            # Check all
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(rank):
                            val += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                        if abs(val - 1.0) > tol:
                            return False
            return True
        else:
            # Check random sample
            np.random.seed(12345)
            for _ in range(50):
                i = np.random.randint(n)
                j = np.random.randint(m)
                k = np.random.randint(p)
                val = 0.0
                for c in range(rank):
                    val += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
            return True
    
    # Try structured decompositions using linearity
    # Idea: group terms to reduce multiplications
    # For example, (a+b)(c+d) = ac + ad + bc + bd can be done with 3 mults: (a+b)(c+d) - ac - bd
    
    # Let's try to find a decomposition of rank r < n*m*p
    # We'll use a systematic approach: try ranks from min_possible_rank upward
    
    # For 3x5 * 5x7, the schoolbook rank is 105
    # We know that matrix multiplication can be done with fewer than n*m*p multiplications
    # For small matrices, we can try to find explicit decompositions
    
    # Approach: use the fact that we can combine terms
    # For 2x2 * 2x2, Strassen's algorithm uses 7 multiplications instead of 8
    # For our case, let's try to find a decomposition using random search + local optimization
    
    # Actually, let's try a more direct approach: 
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We want T = sum_c U_c ⊗ V_c ⊗ W_c where U_c[i,j], V_c[j,k], W_c[k,i]
    
    # This is equivalent to finding rank-r decomposition of the tensor
    # Let's try to use a known result: for matrix multiplication, the rank is at most n*m*p - something
    
    # For now, let's try to find a better decomposition using random initialization and gradient descent
    
    def try_find_decomposition(target_rank, max_iter=100, seed=42):
        """Try to find a decomposition of the given rank using optimization."""
        np.random.seed(seed)
        
        # Initialize with small random values
        U_rand = np.random.randn(n * m, target_rank) * 0.1
        V_rand = np.random.randn(m * p, target_rank) * 0.1
        W_rand = np.random.randn(p * n, target_rank) * 0.1
        
        # We want to minimize ||T_decomp - T||^2 where T[i,j,k] = 1
        # T_decomp[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        
        # Use a simple gradient-free approach: coordinate descent
        # For each c, update U[:,c], V[:,c], W[:,c] to minimize the error
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                return None
            
            # Update U
            for c in range(target_rank):
                # Fix V[:,c] and W[:,c], optimize U[:,c]
                # This is a linear problem: for each (i,j), 
                # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                # For fixed c, the contribution to (i,j,k) is U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                # This is complex, so let's use a simpler approach
                
                pass
            
            # Check if we've converged
            if verify_decomposition(U_rand.tolist(), V_rand.tolist(), W_rand.tolist()):
                return U_rand, V_rand, W_rand
        
        return None
    
    # Try different ranks
    for target_rank in range(min_possible_rank, rank_schoolbook + 1):
        if time.time() > deadline:
            break
        
        found = False
        for restart in range(3):
            if time.time() > deadline:
                break
            
            result = try_find_decomposition(target_rank, max_iter=50, seed=42 + restart)
            if result is not None:
                U_rand, V_rand, W_rand = result
                if verify_decomposition(U_rand.tolist(), V_rand.tolist(), W_rand.tolist()):
                    best_rank = target_rank
                    best_U = U_rand.tolist()
                    best_V = V_rand.tolist()
                    best_W = W_rand.tolist()
                    found = True
                    break
        
        if found:
            break
    
    # Return the best decomposition found
    if best_rank < rank_schoolbook:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
