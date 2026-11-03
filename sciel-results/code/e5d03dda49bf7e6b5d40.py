"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import product
import time
import random
import sys

DIMENSIONS = [4, 5, 5]
TIME_LIMIT = 140.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    # Schoolbook rank
    r_school = n * m * p
    # Build schoolbook matrices
    U_sb = np.zeros((n * m, r_school))
    V_sb = np.zeros((m * p, r_school))
    W_sb = np.zeros((p * n, r_school))
    
    # Use vectorized construction for speed, though loop is fast enough
    # This ensures we are well within time limits
    indices = list(product(range(n), range(m), range(p)))
    for c, (i, j, k) in enumerate(indices):
        U_sb[i * m + j, c] = 1.0
        V_sb[j * p + k, c] = 1.0
        W_sb[k * n + i, c] = 1.0

    def _pack(U, V, W, r):
        return U[:, :r].tolist(), V[:, :r].tolist(), W[:, :r].tolist()

    def _is_valid(U, V, W, r):
        """Check if the decomposition is exact for matrix multiplication."""
        # Check time
        if time.time() - start_time > TIME_LIMIT - 10:
            return False
        # For each term c in the decomposition, verify that:
        # sum_c U[:,c] * V[:,c] * W[:,c] gives the correct matrix multiplication tensor
        # The tensor T[i,j,k] = 1 if i,j,k are valid indices, 0 otherwise
        # Actually, for matrix multiplication: (AB)[i,k] = sum_j A[i,j] B[j,k]
        # The tensor has components T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
        # U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
        # The decomposition should satisfy:
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # We need to check this for all i, j, k
        # For efficiency, we can check a subset first, then verify fully if it looks good
        # But for correctness, we must check all
        
        # Build the full tensor check
        # T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # The true tensor has T[i,j,k] = 1 for all valid (i,j,k) since it's the identity of matrix mult
        # Wait, no. The matrix multiplication tensor is:
        # T[i,j,k] = 1 if we're computing (AB)[i,k] using A[i,j] and B[j,k]
        # Actually, the standard tensor for matmul is:
        # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # And the decomposition is: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        
        # Let's verify by checking a few random entries first, then all if needed
        rng = np.random.default_rng(42)
        # Check 100 random entries
        for _ in range(100):
            i = rng.integers(0, n)
            j = rng.integers(0, m)
            k = rng.integers(0, p)
            lhs = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(r))
            rhs = 1.0  # The true tensor value
            if abs(lhs - rhs) > 1e-10:
                return False
        
        # If random check passes, do full check
        for i in range(n):
            if time.time() - start_time > TIME_LIMIT - 5:
                return False
            for j in range(m):
                for k in range(p):
                    lhs = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(r))
                    if abs(lhs - 1.0) > 1e-10:
                        return False
        return True

    def _try_random_search(max_rank, num_trials=100):
        """Try to find a lower-rank decomposition using random search."""
        rng = np.random.default_rng(42)
        # Define the set of allowed values
        values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        for r in range(max_rank, r_school):
            if time.time() - start_time > TIME_LIMIT - 20:
                return None
            for trial in range(num_trials):
                if time.time() - start_time > TIME_LIMIT - 10:
                    return None
                # Generate random half-integer matrices
                # We need integer/half-integer entries. 
                # Sampling from the discrete set is correct.
                U = rng.choice(values, size=(n*m, r))
                V = rng.choice(values, size=(m*p, r))
                W = rng.choice(values, size=(p*n, r))
                
                if _is_valid(U, V, W, r):
                    return U, V, W, r
        return None

    # Check time before attempting any complex search
    if time.time() - start_time > TIME_LIMIT - 30:
        U_list, V_list, W_list = _pack(U_sb, V_sb, W_sb, r_school)
        return {"U": U_list, "V": V_list, "W": W_list}

    # Try to find a lower-rank decomposition
    # The theoretical lower bound is at least max(n*m, m*p, p*n) = max(20, 25, 20) = 25
    # But in practice, for 4x5x5, the minimum rank is likely higher
    # Let's try from a reasonable lower bound up to schoolbook
    min_possible = max(n*m, m*p, p*n)
    
    # Try structured search: use the schoolbook but try to merge terms
    # For now, try random search from a higher rank down
    result = _try_random_search(min_possible + 50, num_trials=50)
    
    if result is not None:
        U, V, W, r = result
        U_list, V_list, W_list = _pack(U, V, W, r)
        return {"U": U_list, "V": V_list, "W": W_list}

    # Return the schoolbook decomposition as the safe, verified baseline.
    U_list, V_list, W_list = _pack(U_sb, V_sb, W_sb, r_school)
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
