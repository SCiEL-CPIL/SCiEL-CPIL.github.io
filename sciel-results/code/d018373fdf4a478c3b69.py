"""
Strategy: Systematic Search for Low-Rank Integer Decompositions using Linear Algebra.
Instead of gradient descent, we treat the tensor decomposition as a system of linear equations.
We fix U and V to be sparse integer matrices and solve for W.
If the solution for W is close to integers, we accept it.
"""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the factor matrices exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Must equal 1 for all i, j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_integer_decomposition(n, m, p, rank, max_trials=500, seed=0):
    """
    Try to find an integer decomposition of the given rank.
    1. Randomly generate sparse integer U and V.
    2. Solve the linear system for W.
    3. Check if W is close to integers.
    """
    np.random.seed(seed)
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    for trial in range(max_trials):
        # Generate random sparse integer U and V
        # Entries in {-1, 0, 1}
        U = np.random.randint(-1, 2, size=(n_m, rank))
        V = np.random.randint(-1, 2, size=(m_p, rank))
        
        # Construct the linear system for W
        # Equation for each (i, j, k):
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # Let x be the vector of W flattened: x[k*n+i, r]
        # The coefficient for x[k*n+i, r] in equation (i,j,k) is U[i*m+j, r] * V[j*p+k, r]
        
        # We have n*m*p equations.
        # We have p*n*rank unknowns.
        
        A = np.zeros((n * m * p, p * n * rank))
        b = np.ones(n * m * p)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        # Unknown index: W[k*n+i, r] -> flattened index (k*n+i)*rank + r
                        unk_idx = (k * n + i) * rank + r
                        # Coefficient
                        coeff = U[i * m + j, r] * V[j * p + k, r]
                        A[eq_idx, unk_idx] = coeff
                    eq_idx += 1
        
        # Solve A x = b
        # Since it's underdetermined (usually), we use least squares to get a solution.
        # Then we check if the solution is close to integers.
        try:
            x, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            continue
            
        # Reshape x to W
        W = x.reshape((p * n, rank))
        
        # Check if W is close to integers
        # We allow some tolerance, but since we want exact integer decompositions,
        # we check distance to nearest integer.
        W_int = np.round(W)
        if np.allclose(W, W_int, atol=1e-6):
            W_int = W_int.astype(float) # Keep as float for interface consistency
            
            # Verify the solution
            if _verify_decomposition(U, V, W_int, n, m, p):
                return {"U": U, "V": V, "W": W_int}
    
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition with rank = n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Start with schoolbook
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Lower bound for rank is max(n*m, m*p, p*n)
    # For 3x5x6: max(15, 30, 18) = 30
    lower_bound = max(n * m, m * p, p * n)
    
    # Try ranks from lower_bound up to best_rank - 1
    # We prioritize lower ranks
    for rank in range(lower_bound, best_rank):
        if time.time() > deadline - 5:
            break
            
        # Try to find an integer decomposition for this rank
        # We use a few different seeds to explore different random U, V structures
        found = False
        for seed in range(10):
            if time.time() > deadline - 5:
                break
            result = _try_integer_decomposition(n, m, p, rank, max_trials=20, seed=rank * 100 + seed)
            if result is not None:
                best_result = result
                best_rank = rank
                found = True
                break
                
        if found:
            # If we found a solution for this rank, we can try to go lower, 
            # but since we are iterating upwards, this is the first found for this rank.
            # We should continue to check if a lower rank was possible? 
            # No, we iterate from lower_bound upwards. So the first found is the minimum 
            # in this search space. 
            # However, it's possible that rank=30 fails but rank=31 succeeds. 
            # We break out of the inner loop, but continue the outer loop?
            # No, if we find a solution for rank R, it's the best so far.
            # We should check if we can go lower? We started at lower_bound.
            # So if we find a solution at R, it is the minimal rank found in this search.
            # We can break the outer loop if we assume the search is exhaustive enough?
            # No, we should just record it and break, because we are iterating from low to high.
            # If we find one at R, it's better than any R+1.
            # But we might have missed R-1? We started at lower_bound.
            # So if we find one at R, it is the lowest possible in our search order.
            break

    return {"U": best_result["U"].tolist(), "V": best_result["V"].tolist(), "W": best_result["W"].tolist()}
