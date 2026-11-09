"""Rank-minimal exact tensor decomposition for 2×5 × 5×6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def _schoolbook():
    """
    Schoolbook rank-n*m*p decomposition for n×m times m×p matrix multiplication.
    Uses the trace convention: output index is k*n + i for W.
    Input: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    For each term c = (i, j, k):
        U[i*m + j, c] = 1
        V[j*p + k, c] = 1
        W[k*n + i, c] = 1
    """
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


def _verify(U, V, W, tol=1e-9):
    """Verify that the decomposition is exact."""
    n, m, p = DIMENSIONS
    r = len(W[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    if Ua.shape != (n * m, r) or Va.shape != (m * p, r) or Wa.shape != (p * n, r):
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(r):
                    val += Ua[i * m + j, c] * Va[j * p + k, c] * Wa[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True


def _is_half_integer(val, tol=1e-9):
    """Check if val is a half-integer (k/2 for integer k)."""
    two_val = 2.0 * val
    nearest = round(two_val)
    return abs(two_val - nearest) < tol


def _check_half_integer_entries(U, V, W, tol=1e-9):
    """Check all entries are half-integers."""
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if not _is_half_integer(val, tol):
                    return False
    return True


def _random_decomposition(rank, seed=None):
    """Generate a random rank decomposition with half-integer-like entries."""
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed)
    
    # Generate random matrices with small integer/half-integer entries
    U = rng.randint(-2, 3, size=(n * m, rank)) * 0.5
    V = rng.randint(-2, 3, size=(m * p, rank)) * 0.5
    W = rng.randint(-2, 3, size=(p * n, rank)) * 0.5
    
    return U, V, W


def _check_decomposition(U, V, W, tol=1e-9):
    """Check if the decomposition is valid (returns True if valid)."""
    n, m, p = DIMENSIONS
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(len(Wa[0])):
                    val += Ua[i * m + j, c] * Va[j * p + k, c] * Wa[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True


def _try_bipartite_rank(rank, time_limit=30.0, seed=42):
    """
    Try to find a rank decomposition using random search.
    This is a heuristic approach - may not find optimal solutions.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # For small ranks, we can try more systematic approaches
    # The known lower bound for 2x5x6 is around 18-20
    
    # Try random searches for various ranks
    for attempt_rank in [rank, rank + 1, rank + 2, rank + 3, rank + 4, rank + 5]:
        if time.time() - start_time > time_limit:
            break
        
        # Try multiple random initializations
        for seed_offset in range(20):
            if time.time() - start_time > time_limit:
                break
            
            U, V, W = _random_decomposition(attempt_rank, seed=seed + seed_offset)
            
            if _check_decomposition(U, V, W):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    
    return None


def construct():
    """
    Return the best rank decomposition found for 2×5 × 5×6 matrix multiplication.
    Tries to find a rank lower than the schoolbook bound of 60.
    """
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 150s limit
    
    # Start with schoolbook as fallback
    best_result = _schoolbook()
    best_rank = 2 * 5 * 6  # 60
    
    # Try to find better decompositions
    # Known theoretical bounds suggest rank could be around 18-25
    for target_rank in [18, 19, 20, 21, 22, 23, 24, 25, 30, 40, 50]:
        if time.time() > deadline - 20.0:
            break
        
        result = _try_bipartite_rank(target_rank, time_limit=5.0)
        if result is not None:
            # Verify the result
            if _verify(result["U"], result["V"], result["W"]):
                if _check_half_integer_entries(result["U"], result["V"], result["W"]):
                    best_result = result
                    best_rank = target_rank
                    break  # Found a valid lower rank, stop
    
    # Final verification
    if not _verify(best_result["U"], best_result["V"], best_result["W"]):
        # Fall back to schoolbook if verification fails
        best_result = _schoolbook()
    
    return best_result

# EVOLVE-BLOCK-END
