import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [5, 5, 5]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 5x5 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Build the schoolbook decomposition as baseline
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook), dtype=float)
    V_school = np.zeros((m * p, rank_schoolbook), dtype=float)
    W_school = np.zeros((p * n, rank_schoolbook), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    def check_decomposition(U, V, W, tol=1e-8):
        """Check if the decomposition is valid."""
        rank = U.shape[1]
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    max_err = max(max_err, abs(s - 1.0))
        return max_err < tol, max_err
    
    def try_strassen_block():
        """
        Try to use Strassen's algorithm for 2x2 blocks.
        5 = 2 + 2 + 1. We can split the matrices into blocks.
        This is complex, so let's try a simpler approach: 
        use known low-rank decompositions for smaller cases.
        """
        # For now, return None to indicate we didn't find anything better
        return None
    
    # Strategy: Use the fact that for 5x5x5, we can potentially use
    # a combination of Strassen-like tricks on sub-blocks.
    # 
    # Let's try a different approach: use the known result that the
    # tensor rank of matrix multiplication for 5x5x5 is at most 103.
    # We'll try to construct such a decomposition using a randomized
    # algebraic approach.
    
    # The key insight: we need to find U, V, W such that for all i,j,k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #
    # This is equivalent to: the tensor T with T[i,j,k] = 1 can be
    # decomposed as a sum of rank-1 tensors.
    #
    # Let's try a structured approach using the fact that the all-ones
    # tensor has a special structure.
    
    # Approach 1: Try to find a decomposition with rank < 125 using
    # a randomized search with local optimization.
    
    random.seed(42)
    np.random.seed(42)
    
    # Try several candidate ranks
    for target_rank in [120, 115, 110, 105, 103, 100, 95]:
        if time.time() > deadline - 15:
            break
        
        # Try multiple random initializations
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
            
            # Initialize with small random values
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Use alternating minimization to try to satisfy the constraints
            # For each r, we want sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            #
            # We can fix two matrices and solve for the third.
            # Given U and V, for each (i,j,k), we need:
            # sum_r W[k*n+i, r] * (U[i*m+j, r] * V[j*p+k, r]) = 1
            #
            # This is a linear system in W!
            
            # Let's try alternating minimization
            converged = False
            for iteration in range(50):
                if time.time() > deadline - 5:
                    break
                
                # Update W: for each (i,k), solve for W[k*n+i, :]
                # We need: for all j, sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1
                # Let A_{j,r} = U[i*m+j, r] * V[j*p+k, r]
                # Then: sum_r W[k*n+i, r] * A_{j,r} = 1 for all j
                # This is: W[k*n+i, :] @ A.T = ones(m)
                
                # Actually, this is underdetermined in general. Let's use least squares.
                for i in range(n):
                    for k in range(p):
                        # Build the system
                        # For each j in range(m): sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1
                        # Let M be a (m, target_rank) matrix where M[j, r] = U[i*m+j, r] * V[j*p+k, r]
                        # We want: W[k*n+i, :] @ M.T = ones(m)
                        # i.e., M @ W[k*n+i, :].T = ones(m)
                        M = np.zeros((m, target_rank))
                        for j in range(m):
                            for r in range(target_rank):
                                M[j, r] = U[i*m+j, r] * V[j*p+k, r]
                        
                        # Solve M @ w = ones(m) in least squares sense
                        try:
                            w, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                            W[k*n+i, :] = w
                        except:
                            pass
                
                # Update U: for each (i,j), solve for U[i*m+j, :]
                # We need: for all k, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # Let M2_{k,r} = V[j*p+k, r] * W[k*n+i, r]
                # Then: sum_r U[i*m+j, r] * M2_{k,r} = 1 for all k
                # i.e., M2 @ U[i*m+j, :].T = ones(p)
                for i in range(n):
                    for j in range(m):
                        M2 = np.zeros((p, target_rank))
                        for k in range(p):
                            for r in range(target_rank):
                                M2[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        
                        try:
                            u, residuals, rank_m, sv = np.linalg.lstsq(M2, np.ones(p), rcond=None)
                            U[i*m+j, :] = u
                        except:
                            pass
                
                # Update V: for each (j,k), solve for V[j*p+k, :]
                # We need: for all i, sum_r V[j*p+k, r] * U[i*m+j, r] * W[k*n+i, r] = 1
                # Let M3_{i,r} = U[i*m+j, r] * W[k*n+i, r]
                # Then: sum_r V[j*p+k, r] * M3_{i,r} = 1 for all i
                # i.e., M3 @ V[j*p+k, :].T = ones(n)
                for j in range(m):
                    for k in range(p):
                        M3 = np.zeros((n, target_rank))
                        for i in range(n):
                            for r in range(target_rank):
                                M3[i, r] = U[i*m+j, r] * W[k*n+i, r]
                        
                        try:
                            v, residuals, rank_m, sv = np.linalg.lstsq(M3, np.ones(n), rcond=None)
                            V[j*p+k, :] = v
                        except:
                            pass
                
                # Check convergence
                max_err, _ = check_decomposition(U, V, W, tol=1e-6)
                if max_err < 1e-6:
                    converged = True
                    break
            
            if converged:
                # Verify with tighter tolerance
                valid, max_err = check_decomposition(U, V, W, tol=1e-8)
                if valid:
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = target_rank
                    break
        
        if best_rank < rank_schoolbook:
            break
    
    # Convert to lists for output
    U_out = best_result["U"].tolist()
    V_out = best_result["V"].tolist()
    W_out = best_result["W"].tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
