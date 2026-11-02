"""Program for 3x3 x 3x5 matrix multiplication tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 5]
DEADLINE = 150.0
SAFETY_MARGIN = 10.0  # seconds before deadline to stop

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    # The multiplication tensor T[i,j,k] = delta(i,j,k) where
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor index: (i, j, k) with i in [0,n), j in [0,m), k in [0,p)
    # Factorization: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # W uses trace convention: W[k*n + i, r]
    
    def verify(U, V, W, rank):
        """Verify that the decomposition exactly reconstructs the multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # T[i,j,k] should be 1 if this is a valid multiplication term
                    # Actually T[i,j,k] = 1 for all (i,j,k) in the multiplication
                    # The tensor is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    # This should equal 1 for all i,j,k (since C[i,k] = sum_j A[i,j]B[j,k])
                    val = 0
                    for r in range(rank):
                        val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(val - 1) > 1e-9:
                        return False
        return True
    
    def make_schoolbook():
        """Return the schoolbook decomposition of rank n*m*p = 45."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = make_schoolbook()
    best_rank = n * m * p  # 45
    
    # Try to find a lower rank decomposition
    # Known minimal rank for 3x3x5 is 43
    # We'll try ranks from 43 upward
    
    # Strategy: Use a structured search with random perturbations
    # The tensor has a specific structure we can exploit
    
    # First, try a known construction if we can recall one
    # For now, use randomized search with verification
    
    # The multiplication tensor for (n,m,p) has entries T[i,j,k] = 1
    # We need U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r to give 1
    
    # Let's try a more systematic approach:
    # Represent the decomposition and solve for entries
    
    # For rank R, we have R unknowns per row of U, V, W
    # Total unknowns: R*(n*m + m*p + p*n) = R*(9 + 15 + 15) = R*39
    # Constraints: n*m*p = 45 equations (one for each (i,j,k))
    
    # For R >= 45, schoolbook works. For R = 43 or 44, we need to search.
    
    # Use a randomized approach: try different random initializations
    # and use a solver to find exact solutions
    
    def try_rank(R, max_attempts=100):
        """Try to find a rank-R decomposition."""
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            return None
        
        # The system is: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # This is a system of 45 cubic equations in R*39 variables.
        
        # Strategy: Fix W to be a random matrix, then the system becomes:
        # For each (i,j,k): sum_r W[k*n+i,r] * (U[i*m+j,r] * V[j*p+k,r]) = 1
        # Let X_r = U[i*m+j,r] * V[j*p+k,r] for each r
        # Then: sum_r W[k*n+i,r] * X_r = 1
        # But X_r depends on both i,j,k through U and V, so this is coupled.
        
        # Alternative: Use the fact that we can set some entries to 0 (sparsity)
        # and solve a linear system for the rest.
        
        # Let's try: set W to have a specific structure
        # For each r, W[:,r] is a vector of length p*n = 15
        
        # Try random W with many zeros, then solve for U and V
        
        for attempt in range(max_attempts):
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                return None
            
            # Random W: each entry is 0, 1, or -1 with some probability
            W_mat = np.zeros((p*n, R))
            for col in range(R):
                # Randomly choose which entries are non-zero
                nnz = random.randint(1, 5)
                idx = random.sample(range(p*n), min(nnz, p*n))
                for idx_val in idx:
                    W_mat[idx_val, col] = random.choice([-1, 1])
            
            # Now for each (i,j,k), we need:
            # sum_r W_mat[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1
            
            # This is still nonlinear. Let's try a different approach.
            
            # Use a greedy construction:
            # Start with schoolbook and try to merge terms.
            
            break  # This approach is too complex for the time limit
        
        return None
    
    # Given the complexity, let's use a known result:
    # The minimal rank for 3x3x5 matrix multiplication is 43.
    # Let me try to construct it directly using a known pattern.
    
    # Actually, let me try a simpler approach: use the fact that we can
    # use half-integer entries and try to find a valid rank-43 or rank-44 solution.
    
    # For now, let's try a randomized search with a linear solver approach.
    
    def solve_linear_system(A, b):
        """Solve Ax = b, return None if no solution."""
        try:
            x = np.linalg.solve(A, b)
            # Check if solution is valid
            if np.allclose(A @ x, b, atol=1e-8):
                return x
            return None
        except np.linalg.LinAlgError:
            return None
    
    # Try rank 43 and 44 with structured search
    for target_rank in [43, 44]:
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        found = False
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
            
            # Strategy: Use a sparse pattern for U, V, W
            # Initialize with schoolbook-like structure but with some merged terms
            
            # Create random sparse U, V, W
            U_mat = np.zeros((n*m, target_rank))
            V_mat = np.zeros((m*p, target_rank))
            W_mat = np.zeros((p*n, target_rank))
            
            # For each column r, assign random non-zero entries
            for r in range(target_rank):
                # U: choose 1-3 rows to be non-zero
                u_rows = random.sample(range(n*m), random.randint(1, 3))
                for row in u_rows:
                    U_mat[row, r] = random.choice([-1, 1])
                
                # V: choose 1-3 rows to be non-zero
                v_rows = random.sample(range(m*p), random.randint(1, 3))
                for row in v_rows:
                    V_mat[row, r] = random.choice([-1, 1])
                
                # W: choose 1-3 rows to be non-zero
                w_rows = random.sample(range(p*n), random.randint(1, 3))
                for row in w_rows:
                    W_mat[row, r] = random.choice([-1, 1])
            
            # Verify
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r in range(target_rank):
                            val += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                        if abs(val - 1) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                found = True
                best_rank = target_rank
                best_result = {
                    "U": U_mat.tolist(),
                    "V": V_mat.tolist(),
                    "W": W_mat.tolist()
                }
                break
        
        if found:
            break
    
    # Check deadline
    if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
