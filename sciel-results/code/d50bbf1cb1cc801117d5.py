"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_school = n * m * p  # 100
    
    # Return schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    deadline = time.time() + 100  # safety margin
    
    # Try to find a lower-rank decomposition using random search
    # The tensor T_{i,j,k} = 1 for all i,j,k (matrix multiplication tensor)
    # We need U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r = 1
    
    # For matrix multiplication, the tensor is T[i,j,k] = 1
    # A rank-r decomposition means: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Let's try small ranks. Known lower bounds: for 4x5x5, rank >= ?
    # The trivial upper bound is 100. Let's try to find something better.
    
    # Strategy: Use random initialization and gradient-free search
    # Try ranks from a reasonable lower bound upward
    
    # Lower bound: max(n*m, m*p, p*n) = max(20, 25, 20) = 25
    # But actual lower bound for matrix multiplication is higher.
    # Let's try ranks in range [25, 100) and see if we can find exact solutions.
    
    best_result = schoolbook()
    best_rank = rank_school
    
    # For exact decomposition with integer/half-integer entries,
    # we can try specific structured decompositions.
    
    # One approach: if we can find matrices A (n x m), B (m x p), C (p x n)
    # such that certain conditions hold, we can reduce rank.
    
    # Actually, for the standard matrix multiplication tensor, a rank-r
    # decomposition corresponds to expressing the product as a sum of r
    # "rank-1" tensor terms.
    
    # Let's try a simple approach: check if rank < 100 is possible
    # by attempting random searches with small ranks.
    
    # Given the time constraint and the difficulty of finding exact
    # low-rank decompositions, let's try a few structured approaches.
    
    # Approach 1: Try to use the fact that for square matrices, there
    # might be symmetries. For 4x5x5, it's not square.
    
    # Approach 2: Random search for small ranks
    for target_rank in range(26, min(60, rank_school)):
        if time.time() > deadline - 20:
            break
        if target_rank >= best_rank:
            break
        
        # Try a few random initializations
        for attempt in range(50):
            if time.time() > deadline - 20:
                break
            
            # Initialize with small random values
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Simple iterative refinement: try to make the tensor match
            # This is a non-convex problem, so we'll use a simple approach
            # Check current error
            # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            # We want T[i,j,k] = 1 for all i,j,k
            
            # Compute error
            err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for r in range(target_rank):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        err += (s - 1)**2
            
            if err < 1e-10:
                # Found an exact solution!
                # Round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Verify
                err2 = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for r in range(target_rank):
                                s += U_r[i*m+j, r] * V_r[j*p+k, r] * W_r[k*n+i, r]
                            err2 += (s - 1)**2
                
                if err2 < 1e-10:
                    best_rank = target_rank
                    best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    break
        
        if best_rank == target_rank:
            break
    
    # Also try the schoolbook as a baseline and return the best
    if best_rank < rank_school:
        return best_result
    else:
        return schoolbook()
# EVOLVE-BLOCK-END
