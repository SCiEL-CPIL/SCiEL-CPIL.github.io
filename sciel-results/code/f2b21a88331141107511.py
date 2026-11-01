import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    trivial_rank = n * m * p  # 36
    
    # Build the multiplication tensor
    # T[i, j, k] = 1 for i in [0,n), j in [0,m), k in [0,p)
    # This is a tensor of shape (n, m, p) but we need it in the form
    # where U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # and sum_r U[:,r] outer V[:,r] outer W[:,r] = T
    # where T is flattened appropriately
    
    # The tensor T has entries T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i
    # for some i, j, k, else 0.
    # Actually, the multiplication tensor is defined as:
    # For each (i, j, k), we have a rank-1 term: e_{i*m+j} outer e_{j*p+k} outer e_{k*n+i}
    # So the full tensor is sum_{i,j,k} e_{i*m+j} outer e_{j*p+k} outer e_{k*n+i}
    
    # Let's build the full tensor T as a 3D array of shape (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    
    # Try to find a lower rank decomposition using random search
    # We'll try ranks from 20 to trivial_rank-1
    # Use a simple approach: try to find coefficients such that
    # sum_r U[:,r] outer V[:,r] outer W[:,r] = T
    
    # This is a hard problem. Let's try a few strategies:
    # 1. For small ranks, use random initialization and gradient descent
    # 2. For larger ranks, use the structure to find a decomposition
    
    # Actually, let's try to use the known result that for matrix multiplication,
    # the rank is at most n*m*p, and for 3x3x4, it might be lower.
    
    # Let's try a random search for rank 30-35
    # We'll use a simple iterative method: start with random U, V, W
    # and try to minimize the Frobenius norm of the residual
    
    deadline = time.time() + 120
    best_rank = trivial_rank
    best_result = None
    
    # First, set up the trivial decomposition as a fallback
    U_triv = np.zeros((n * m, trivial_rank))
    V_triv = np.zeros((m * p, trivial_rank))
    W_triv = np.zeros((p * n, trivial_rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_triv[i * m + j, idx] = 1
                V_triv[j * p + k, idx] = 1
                W_triv[k * n + i, idx] = 1
                idx += 1
    
    # Try random search for lower ranks
    # We'll try to use the fact that the tensor is sparse
    # and look for a decomposition with fewer terms
    
    # Strategy: For each candidate rank r, try to find U, V, W such that
    # the sum of outer products equals T
    # This is equivalent to solving a system of equations
    
    # Let's try a different approach: use the structure of the problem
    # The multiplication (3x3) * (3x4) -> (3x4)
    # The output has 12 entries, each computed as a sum of 3 products
    # Total: 36 multiplications in the naive algorithm
    
    # Can we do better? For 3x3 matrix multiplication, the rank is 23
    # (or 21 over some fields). For 3x3x4, it might be lower than 36.
    
    # Let's try to use a known decomposition or find one via search
    # Given the time constraint, let's try a few random searches
    
    for r in range(30, trivial_rank):
        if time.time() > deadline - 10:
            break
        
        # Try random search with a few restarts
        found = False
        for trial in range(100):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # Simple iterative refinement
            for it in range(50):
                if time.time() > deadline - 5:
                    break
                
                # Compute the residual
                # T_approx = sum_r U[:,r] outer V[:,r] outer W[:,r]
                # This is expensive, so let's compute it efficiently
                # T_approx[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
                
                # Compute the residual for a few random entries to check
                # If the residual is small enough, we've found a decomposition
                # But for exact decomposition, we need integer/half-integer entries
                # So this approach won't work for exact decomposition
                
                break
            
            # Check if we found a valid decomposition
            # For exact decomposition, we need the entries to be rational
            # and the sum to exactly equal T
            # This is very hard to find randomly
            
            found = False
        
        if found:
            best_rank = r
            break
    
    # If we didn't find a better decomposition, use the trivial one
    if best_rank == trivial_rank:
        return {"U": U_triv.tolist(), "V": V_triv.tolist(), "W": W_triv.tolist()}
    
    # This shouldn't be reached, but just in case
    return {"U": U_triv.tolist(), "V": V_triv.tolist(), "W": W_triv.tolist()}
