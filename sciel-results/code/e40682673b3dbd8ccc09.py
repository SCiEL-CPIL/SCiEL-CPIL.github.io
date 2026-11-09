"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]
_DEADLINE = time.time() + 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def _verify(U, V, W):
    """Check if U,V,W exactly decompose the 2x3 x 3x6 matmul tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Tensor T[i,j,k] = 1 iff the matmul has entry at (i,k) from (j)
    # U shape: (n*m, r), V shape: (m*p, r), W shape: (p*n, r)
    # For each column c: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should sum to 1
    # if (i,j,k) is a valid triple, else 0.
    r = U.shape[1]
    for c in range(r):
        u_col = U[:, c]
        v_col = V[:, c]
        w_col = W[:, c]
        # u_col has n*m entries, v_col has m*p entries, w_col has p*n entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = u_col[i * m + j] * v_col[j * p + k] * w_col[k * n + i]
                    if abs(actual - expected) > 1e-9:
                        return False
    return True

def _try_rank_reduction(n, m, p, base_rank, deadline):
    """
    Try to find a decomposition with rank < base_rank.
    Strategy: start from schoolbook, then try to merge columns where
    the pattern allows cancellation or superposition.
    
    For 2x3x6, known bounds: 
    - Schoolbook: 36
    - Winograd-like: potentially lower
    
    We try random perturbations and local search.
    """
    # Build schoolbook as numpy arrays
    U = np.zeros((n * m, base_rank))
    V = np.zeros((m * p, base_rank))
    W = np.zeros((p * n, base_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Try to reduce rank by finding linear dependencies
    # The key insight: if two columns have the same support pattern
    # up to scaling, we might combine them.
    
    # Alternative: use the structure that n=2 is small.
    # For fixed j,k, the i-index only has 2 values.
    # U[i*m+j, c] for i=0,1 and W[k*n+i, c] for i=0,1
    # This means for each (j,k), we have a 2x2 sub-problem.
    
    # Let's try a smarter construction:
    # Group by (j,k). For each (j,k), we need:
    #   sum_c U[0*m+j,c] * V[j*p+k,c] * W[k*n+0,c] = 1
    #   sum_c U[1*m+j,c] * V[j*p+k,c] * W[k*n+1,c] = 1
    #   And cross-terms must vanish:
    #   sum_c U[0*m+j,c] * V[j'*p+k',c] * W[k'*n+0,c] = 0 for (j',k') != (j,k)
    #   sum_c U[1*m+j,c] * V[j'*p+k',c] * W[k'*n+1,c] = 0 for (j',k') != (j,k)
    
    # This is a system of equations. Let's try to solve it with rank < 36.
    
    # For now, try random rank-r decompositions with r < 36
    # and check if they satisfy the constraints.
    
    best_rank = base_rank
    rng = np.random.RandomState(42)
    
    # Try a few candidate ranks below 36
    for target_rank in [24, 20, 18, 16, 15, 14, 12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]:
        if time.time() > deadline:
            break
        
        # Random initialization
        U_r = rng.randn(n * m, target_rank) * 0.1
        V_r = rng.randn(m * p, target_rank) * 0.1
        W_r = rng.randn(p * n, target_rank) * 0.1
        
        # Build the constraint matrix
        # For each (i,j,k) triple, we need:
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # For wrong triples (i,j',k) with j' != j, etc., we need 0.
        
        # This is a polynomial system. Let's use a different approach:
        # Fix U and V, solve for W linearly.
        
        # For each (k,i) pair in W, the constraint is:
        # sum_c W[k*n+i,c] * (sum_j U[i*m+j,c] * V[j*p+k,c] for the correct j) ... 
        # Actually, let me think more carefully.
        
        # The tensor T[i,j,k] = 1 for all valid (i,j,k).
        # We need: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        
        # For fixed (i,j,k), this is one equation.
        # There are n*m*p = 36 equations.
        # Unknowns: r entries in each of U (n*m rows), V (m*p rows), W (p*n rows)
        # Total unknowns: r*(n*m + m*p + p*n) = r*(6+18+12) = 36r
        # With 36 equations, we have many more unknowns than equations for any r >= 1.
        # But the equations are nonlinear (products of three variables).
        
        # Let's try a structured approach:
        # Assume U and V are rank-1 per group.
        
        # For the 2x3x6 case, let's try:
        # U[i*m+j, c] = a[i,j,c]
        # V[j*p+k, c] = b[j,k,c]  
        # W[k*n+i, c] = w[k,i,c]
        
        # We need: sum_c a[i,j,c] * b[j,k,c] * w[k,i,c] = 1 for all i,j,k
        
        # This is a 3-tensor factorization problem.
        
        # Try rank-1 per (j) slice:
        # For each j, let a[i,j,c] = alpha[i,j] * gamma[j,c]
        #                    b[j,k,c] = beta[k,j] * gamma[j,c]  (same gamma)
        # Then sum_c alpha[i,j]*gamma[j,c]*beta[k,j]*gamma[j,c]*w[k,i,c]
        #        = sum_c alpha[i,j]*beta[k,j]*gamma[j,c]^2 * w[k,i,c]
        # This doesn't simplify nicely.
        
        # Let's just try random search with a few iterations.
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            U_r = rng.randn(n * m, target_rank)
            V_r = rng.randn(m * p, target_rank)
            W_r = rng.randn(p * n, target_rank)
            
            # Check constraints
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(target_rank):
                            s += U_r[i*m+j, c] * V_r[j*p+k, c] * W_r[k*n+i, c]
                        if abs(s - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                # Also check that wrong combinations give 0
                # Actually, the tensor only has 1s at valid positions,
                # and 0 elsewhere. But for matmul, ALL (i,j,k) triples are valid!
                # Wait, no. In matmul (2x3)(3x6), the result is 2x6.
                # T[i,j,k] represents the contribution to output[i,k] from input1[i,j]*input2[j,k].
                # So T[i,j,k] = 1 for ALL i in [0,2), j in [0,3), k in [0,6).
                # There are no "wrong" triples - all 36 are valid.
                # So we only need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all 36 triples.
                best_rank = target_rank
                return U_r, V_r, W_r, target_rank
        
        # Try structured: set some entries to 0 to reduce effective rank
        # Use Kronecker structure
        # U[i*m+j, c] = A[i,j, c1] * B[i,j, c2] where c = c1*p + c2 or similar
        
    return None

def construct():
    n, m, p = DIMENSIONS
    base_rank = n * m * p
    
    # Try to find a lower-rank decomposition
    result = _try_rank_reduction(n, m, p, base_rank, _DEADLINE)
    
    if result is not None:
        U_r, V_r, W_r, r = result
        # Convert to lists with integer/half-integer entries
        # The random values are floats, so we need to round or find exact values
        # For now, return the schoolbook if we can't get clean values
        # Check if values are close to integers or half-integers
        all_clean = True
        for arr in [U_r, V_r, W_r]:
            for val in arr.flat:
                rounded = round(val * 2) / 2
                if abs(val - rounded) > 1e-4:
                    all_clean = False
                    break
            if not all_clean:
                break
        
        if all_clean:
            U_list = [[round(val * 2) / 2 for val in row] for row in U_r]
            V_list = [[round(val * 2) / 2 for val in row] for row in V_r]
            W_list = [[round(val * 2) / 2 for val in row] for row in W_r]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
    rank = base_rank
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
