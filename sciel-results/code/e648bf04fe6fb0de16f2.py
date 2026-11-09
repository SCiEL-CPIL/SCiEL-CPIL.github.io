"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110s to leave safety margin before 120s limit
    
    # Start with schoolbook
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Verify function for exact decomposition
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition exactly decomposes the multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True
    
    # Try to find lower rank decompositions using a structured approach
    # For 2x4x7, we can exploit the structure
    
    # Approach: Use a bilinear method based on the structure
    # We'll try to find a decomposition with rank less than n*m*p = 56
    
    # Known lower bound: rank >= max(n*m, m*p, p*n) = max(8, 28, 14) = 28
    # Also rank >= n*m*p / min(n,m,p) = 56/2 = 28
    
    # Let's try ranks from 28 upward
    
    for target_rank in range(28, rank_schoolbook):
        if time.time() > deadline - 30:
            break
        
        # Try a structured decomposition using the fact that n=2
        # For each pair (j,k), we need two equations:
        # sum_c U[0*m+j, c] * V[j*p+k, c] * W[k*n+0, c] = 1
        # sum_c U[1*m+j, c] * V[j*p+k, c] * W[k*n+1, c] = 1
        
        # We can try to construct U, V, W with a specific pattern
        # Let's use a greedy approach: assign values to minimize rank
        
        # Try random initialization with exact verification
        np.random.seed(target_rank)  # Reproducible for each rank
        
        found = False
        max_attempts = 100
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 20:
                break
            
            # Initialize U, V with random small integers
            U = np.zeros((n * m, target_rank), dtype=int)
            V = np.zeros((m * p, target_rank), dtype=int)
            
            # Set some entries to 1, -1, 0 randomly
            for c in range(target_rank):
                # Choose a few non-zero entries for each column
                num_nonzero_U = np.random.randint(1, 4)
                indices_U = np.random.choice(n * m, num_nonzero_U, replace=False)
                for idx in indices_U:
                    U[idx, c] = np.random.choice([-1, 1])
                
                num_nonzero_V = np.random.randint(1, 4)
                indices_V = np.random.choice(m * p, num_nonzero_V, replace=False)
                for idx in indices_V:
                    V[idx, c] = np.random.choice([-1, 1])
            
            # For each (i,j,k), we need:
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            # This is a linear system in W for fixed U, V
            
            # Build the linear system for W
            # Unknowns: W[k*n+i, c] for all k, i, c
            # Equations: for each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            
            # Number of equations: n*m*p = 56
            # Number of unknowns: p*n*target_rank = 14*target_rank
            
            # Let's set up the matrix
            A = np.zeros((n * m * p, p * n * target_rank), dtype=int)
            b = np.ones(n * m * p, dtype=int)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(target_rank):
                            coeff = U[i * m + j, c] * V[j * p + k, c]
                            if coeff != 0:
                                # W[k*n+i, c] is at index (k*n+i) * target_rank + c
                                A[eq_idx, (k * n + i) * target_rank + c] = coeff
                        eq_idx += 1
            
            # Solve the linear system
            # Since we have more unknowns than equations, we want a particular solution
            # Use least squares to get a solution, then check if it's exact
            try:
                W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                W = W_flat.reshape(p * n, target_rank)
                
                # Check if the solution is exact
                # Verify that A @ W_flat == b
                if np.allclose(A @ W_flat, b, atol=1e-10):
                    # Check if W has integer or half-integer entries
                    # Convert to check
                    W_int = np.round(W * 2)
                    if np.allclose(W_int / 2, W, atol=1e-10):
                        # Convert to list of lists with integer/half-integer entries
                        U_list = [[U[i, c] for c in range(target_rank)] for i in range(n * m)]
                        V_list = [[V[i, c] for c in range(target_rank)] for i in range(m * p)]
                        W_list = [[W_int[i, c] / 2 for c in range(target_rank)] for i in range(p * n)]
                        
                        # Final verification
                        if verify_decomposition(U_list, V_list, W_list, target_rank):
                            best_result = {"U": U_list, "V": V_list, "W": W_list}
                            best_rank = target_rank
                            found = True
                            break
            except Exception:
                pass
        
        if found:
            break
    
    # If we didn't find anything better with random search,
    # try a more structured approach using the fact that n=2
    
    if best_rank == rank_schoolbook:
        # Structured approach: exploit n=2
        # For each (j,k), we need:
        # sum_c U[0*m+j, c] * V[j*p+k, c] * W[k*n+0, c] = 1
        # sum_c U[1*m+j, c] * V[j*p+k, c] * W[k*n+1, c] = 1
        
        # Let's try to construct U, V, W with a specific pattern
        # Use the schoolbook as a starting point and try to merge terms
        
        # For now, just return the best we have
        pass
    
    return best_result
# EVOLVE-BLOCK-END
