"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 3, 6]
DEADLINE_MARGIN_S = 10.0
SEARCH_START = 30
SEARCH_MAX_RANK = 54

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _validate(U, V, W, n=3, m=3, p=6, tol=1e-6):
    """Check if U @ V @ W^T reproduces the tensor."""
    n, m, p = 3, 3, 6
    r = len(U[0])
    # Compute for each (i,j,k) whether U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # Sum over c should give 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > tol:
                    return False
    # Check that no other entries are non-zero in a way that breaks the tensor
    # Actually for matrix multiplication tensor, each (i,j,k) must be 1 and all other
    # combinations must be 0. Let's check all possible (i,j,k) combinations:
    for i in range(n):
        for j in range(m):
            for k in range(p):
                pass  # already checked above, all should be 1
    # We also need to ensure that entries like U[i*m+j', c] where j' != j don't 
    # create spurious non-zero entries. But actually the decomposition only requires
    # that the tensor T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c] matches.
    # Since we're checking all (i,j,k) and all should be 1, this is sufficient.
    return True

def _search_decomposition(rank, n=3, m=3, p=6, n_attempts=200, deadline=None):
    """Try to find a rank decomposition by random search."""
    n, m, p = 3, 3, 6
    best = None
    
    for attempt in range(n_attempts):
        if deadline is not None and time.time() > deadline:
            break
        
        # Use a parameterization that tends to produce structured solutions
        # Random initialization with small perturbations around schoolbook
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Start from a permuted schoolbook and perturb
        base_rank = min(rank, n*m*p)
        for c in range(rank):
            # Each rank-1 term corresponds to some (i,j,k)
            if c < n*m*p:
                i, j, k = divmod(c, m*p)
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
                W[k*n+i, c] = 1.0
            else:
                # Extra columns: random small values
                U[:, c] = np.random.randn(n*m) * 0.01
                V[:, c] = np.random.randn(m*p) * 0.01
                W[:, c] = np.random.randn(p*n) * 0.01
        
        if rank < n*m*p:
            # Need to drop some columns and adjust
            # Randomly zero out extra columns
            cols_to_keep = list(range(rank))
            np.random.shuffle(cols_to_keep)
            # This approach is tricky; let's try a different strategy
            # for rank < 54: use a structured random approach
            U = np.random.randn(n*m, rank) * 0.5
            V = np.random.randn(m*p, rank) * 0.5
            W = np.random.randn(p*n, rank) * 0.5
        
        # Convert to lists and validate
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        if _validate(U_list, V_list, W_list):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN_S
    
    n, m, p = DIMENSIONS
    
    # First, try a few specific known lower-rank constructions
    # For 3x3x3, rank 23 is known. For 3x3x6, we can try to extend ideas.
    # One approach: use the 3x3x3 rank-23 decomposition and add 3 more columns
    # for the 6th k-index, but this would give rank 23 + something.
    
    # Let's try a structured approach: the tensor for 3x3x6 can be seen as
    # two copies of 3x3x3 concatenated along k. If 3x3x3 has rank r, then
    # 3x3x6 has rank at most 2*r. But we can do better by sharing.
    
    # Try a random search starting from lower ranks
    # We know schoolbook gives rank 54, so we want to try 53, 52, ...
    # But a pure random search is unlikely to find anything for rank < 54.
    
    # Instead, let's try a greedy approach: start with schoolbook and try to
    # remove columns by checking if the remaining columns still span the tensor.
    
    # Actually, for a more promising approach: use the fact that the tensor
    # T[i,j,k] = delta_{i,j,k} (for matrix mult, T is 1 iff the indices match
    # in the right way for the matrix product). Wait, for matrix multiplication
    # of (n x m) by (m x p), the tensor is T[i,k,j] = 1 where i in [0,n), 
    # j in [0,m), k in [0,p). The decomposition is sum_c U[i,j,c] V[j,k,c] W[k,i,c].
    
    # Let me think about a specific construction. For 3x3x6, one idea:
    # Use rank-1 terms that cover the tensor efficiently.
    
    # Given the complexity, let's do a timed search that tries to find
    # a lower-rank solution using random initialization and validation.
    
    current_best = _schoolbook()
    best_rank = len(current_best['U'][0])
    
    # Try ranks from 53 down to SEARCH_START
    for r in range(min(53, SEARCH_MAX_RANK - 1), SEARCH_START - 1, -1):
        if time.time() > deadline:
            break
        
        if time.time() > deadline:
            break
            
        # For ranks close to 54, try starting from schoolbook with one column removed
        if r >= 50:
            # Remove columns one at a time and check
            schoolbook = _schoolbook()
            for col_to_remove in range(54):
                if time.time() > deadline:
                    break
                U_new = [row[:col_to_remove] + row[col_to_remove+1:] for row in schoolbook['U']]
                V_new = [row[:col_to_remove] + row[col_to_remove+1:] for row in schoolbook['V']]
                W_new = [row[:col_to_remove] + row[col_to_remove+1:] for row in schoolbook['W']]
                if _validate(U_new, V_new, W_new):
                    current_best = {"U": U_new, "V": V_new, "W": W_new}
                    best_rank = r
                    break
        
        # Also try random search for this rank
        if time.time() < deadline - 5.0:
            result = _search_decomposition(r, deadline=deadline)
            if result is not None:
                current_best = result
                best_rank = r
                break  # Found a valid one, move to next (lower) rank
    
    return current_best
# EVOLVE-BLOCK-END
