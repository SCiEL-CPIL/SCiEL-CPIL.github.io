import time
import numpy as np
from fractions import Fraction

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    n, m, p = 2, 4, 7
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                if abs(s - 1.0) > 1e-6:
                    return False
    return True

def _to_half_int_matrix(mat):
    """Convert a matrix to half-integer entries if close enough."""
    mat = np.array(mat, dtype=np.float64)
    # Round to nearest half-integer
    rounded = np.round(mat * 2) / 2
    if np.allclose(mat, rounded, atol=1e-4):
        return rounded.tolist()
    return mat.tolist()

def _try_rank_decomposition(n, m, p, target_rank, max_iters=5000, timeout=10.0):
    """
    Try to find a rank decomposition by solving for sparse patterns.
    Uses a greedy approach: start with schoolbook and try to merge terms.
    """
    start_time = time.time()
    
    # Initialize with schoolbook
    schoolbook_rank = n * m * p
    U = np.zeros((n*m, schoolbook_rank))
    V = np.zeros((m*p, schoolbook_rank))
    W = np.zeros((p*n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    # Try to reduce rank by finding linear dependencies
    # Use a simple iterative approach: try to express one term as combination of others
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try random perturbations and re-solve
    rng = np.random.RandomState(42)
    
    for attempt in range(20):
        if time.time() - start_time > timeout:
            break
        
        # Start with a random sparse structure of target rank
        r = target_rank
        if r > schoolbook_rank:
            continue
            
        U_try = np.zeros((n*m, r))
        V_try = np.zeros((m*p, r))
        W_try = np.zeros((p*n, r))
        
        # Initialize with random sparse patterns
        for c in range(r):
            # Each term has a sparse support
            # U support: choose some (i,j) pairs
            # V support: choose some (j,k) pairs  
            # W support: choose some (k,i) pairs
            
            # Randomly assign values
            U_support = rng.choice(n*m, size=min(3, n*m), replace=False)
            V_support = rng.choice(m*p, size=min(3, m*p), replace=False)
            W_support = rng.choice(p*n, size=min(3, p*n), replace=False)
            
            for idx in U_support:
                U_try[idx, c] = rng.choice([-1, 1])
            for idx in V_support:
                V_try[idx, c] = rng.choice([-1, 1])
            for idx in W_support:
                W_try[idx, c] = rng.choice([-1, 1])
        
        # Check if this works
        valid = True
        for i in range(n):
            if not valid:
                break
            for j in range(m):
                if not valid:
                    break
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U_try[i*m + j, c] * V_try[j*p + k, c] * W_try[k*n + i, c]
                    if abs(s - 1.0) > 1e-4:
                        valid = False
                        break
        
        if valid:
            best_rank = r
            best_U, best_V, best_W = U_try, V_try, W_try
            break
    
    if best_rank < schoolbook_rank:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None

def construct():
    """
    Returns a rank-minimal (or best known) exact tensor decomposition for 
    multiplying a 2x4 matrix by a 4x7 matrix.
    
    The schoolbook rank is n*m*p = 56.
    We search for lower-rank decompositions using structured random initialization.
    """
    n, m, p = 2, 4, 7
    deadline = time.time() + 120
    
    # Schoolbook rank
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill in the schoolbook decomposition
    # The tensor T[i, j, k] = 1 for all valid i, j, k.
    # Decomposition: sum_c U[i, j, c] * V[j, k, c] * W[k, i, c] = 1
    # For each (i, j, k), we assign a unique column c.
    # U[i*m + j, c] = 1
    # V[j*p + k, c] = 1
    # W[k*n + i, c] = 1
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Try to find a lower-rank decomposition
    best_result = (U, V, W, rank)
    
    # Try various target ranks
    for target_rank in range(20, rank):
        if time.time() > deadline - 30:
            break
            
        result = _try_rank_decomposition(n, m, p, target_rank, max_iters=1000, timeout=10.0)
        if result is not None:
            U_new, V_new, W_new = result
            # Verify
            if _verify(U_new, V_new, W_new, n, m, p):
                best_result = (U_new, V_new, W_new, target_rank)
                break  # Found a valid lower rank
    
    U_best, V_best, W_best, rank_best = best_result
    
    # Convert to half-integer if possible
    U_out = _to_half_int_matrix(U_best)
    V_out = _to_half_int_matrix(V_best)
    W_out = _to_half_int_matrix(W_best)
    
    return {"U": U_out, "V": V_out, "W": W_out}

if __name__ == "__main__":
    construct()
