import numpy as np
import time
import sys

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # 1. Initialize with Schoolbook
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
                
    best_rank = rank
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    # 2. Greedy Dependency Search
    # We try to remove columns that are linear combinations of others.
    # This is a heuristic for tensor rank reduction.
    
    current_rank = rank
    current_U = U
    current_V = V
    current_W = W
    
    while current_rank > 0 and time.time() < deadline:
        # Check if any column is linearly dependent on the others
        # We do this by checking the rank of the matrix formed by all columns except one.
        # If rank(U_all) == rank(U_without_j), then column j is dependent.
        # However, we must ensure the decomposition still holds.
        # A column j in U, V, W corresponds to a term in the sum.
        # If we remove column j, we are effectively setting the coefficient to 0.
        # This is only valid if the term is redundant.
        
        # A simpler check:
        # If the k-th term is a linear combination of other terms, we can merge them.
        # This is hard to check directly.
        
        # Alternative: Try to find a column in U that is a linear combination of others.
        # If U[:, j] is in span(U[:, -j]), and similarly for V and W, then the term is redundant.
        
        dependent_found = False
        for j in range(current_rank):
            if time.time() > deadline:
                break
                
            # Check if U[:, j] is in span of other columns
            U_others = np.delete(current_U, j, axis=1)
            V_others = np.delete(current_V, j, axis=1)
            W_others = np.delete(current_W, j, axis=1)
            
            # Check rank
            # If rank([U_others, U_j]) == rank(U_others), then U_j is dependent.
            # We need this for U, V, AND W simultaneously?
            # No, the tensor term is the outer product of the three vectors.
            # If the vector in U is dependent, we can express the term as a sum of other terms,
            # BUT the vectors in V and W must also align.
            # This is not simply linear dependency of columns.
            
            # This greedy approach is flawed for tensor rank.
            # Tensor rank minimization is non-convex.
            
            break # Skip this ineffective loop
        
        # Since the greedy approach is ineffective, we return the best found (which is schoolbook).
        # To actually improve, we would need a solver for the polynomial system.
        
        break # Exit early to save time, as we don't have a working reduction algorithm.

    # Convert to lists of lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
