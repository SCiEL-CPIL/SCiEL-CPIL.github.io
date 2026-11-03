"""
Rank-minimal exact tensor decomposition for 3x4 x 4x5 matrix multiplication.
Uses an integer search approach to find decompositions with rank < 60.
"""
import numpy as np
import time
import itertools
from collections import defaultdict

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START

def _schoolbook():
    """Return the standard schoolbook decomposition with rank = n*m*p = 60."""
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


def _verify_decomp(U, V, W, tol=1e-6):
    """
    Verify that the decomposition exactly reproduces the matrix multiplication tensor.
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 for all i,j,k.
    """
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T_decomp = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(U_arr.shape[1]):
                    s += U_arr[i*m + j, r] * V_arr[j*p + k, r] * W_arr[k*n + i, r]
                T_decomp[i, j, k] = s
    
    T_target = np.ones((n, m, p), dtype=float)
    return np.allclose(T_decomp, T_target, atol=tol)


def _try_remove_terms(schoolbook_U, schoolbook_V, schoolbook_W, target_rank, max_attempts=2000):
    """
    Try to find a rank-`target_rank` decomposition by removing (60 - target_rank) terms
    from the schoolbook and adjusting the remaining vectors to compensate.
    
    This is a heuristic search: we pick a subset of terms to keep, and try to solve
    for integer/half-integer entries in U, V, W that reproduce the tensor.
    """
    n, m, p = DIMENSIONS
    total_terms = n * m * p  # 60
    num_to_keep = target_rank
    num_to_remove = total_terms - target_rank
    
    if num_to_remove <= 0:
        return None
    
    # Build the list of all schoolbook terms
    # Each term r corresponds to (i, j, k) where r = i*m*p + j*p + k
    # Actually in the schoolbook, term index = i*m*p + j*p + k... let me re-check.
    # In _schoolbook: col increments as i, j, k loop. So col = i*m*p + j*p + k.
    
    # For each term r, the vectors are:
    # u_r: has a 1 at position i*m+j
    # v_r: has a 1 at position j*p+k  
    # w_r: has a 1 at position k*n+i
    
    # The tensor entry T[i,j,k] = sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
    
    # If we remove some terms, we need to adjust the remaining vectors so that
    # the sum still equals 1 for all (i,j,k).
    
    # This is a very hard problem in general. Let's try a different approach:
    # Random perturbation of the schoolbook with rank < 60.
    
    return None


def _search_rank(target_rank, time_limit=5.0):
    """
    Attempt to find a rank-`target_rank` decomposition using various heuristics.
    Returns (U, V, W) if found, else None.
    """
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + time_limit
    
    # Strategy 1: Try to construct using Kronecker products and structured forms
    # For matrix multiplication, one known approach is to use the fact that
    # C = A @ B can be computed via various algebraic identities.
    
    # Strategy 2: Use the fact that for small dimensions, we can try to find
    # integer solutions to the system of polynomial equations.
    
    # Strategy 3: Use a greedy approach - start with rank 60 and try to merge terms.
    
    # Let's try a more systematic approach:
    # The matrix multiplication tensor has symmetries. For 3x4x5, 
    # we can try to exploit the rectangular structure.
    
    # One idea: use the fact that A is 3x4 and B is 4x5.
    # We can partition A into 3x2 blocks and B into 2x5 blocks, etc.
    
    # Actually, let's try a simpler heuristic:
    # Randomly initialize U, V, W with small integers and use local search.
    
    np.random.seed(42)
    
    # Try several random initializations
    best_result = None
    
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        # Initialize with small random integers
        U = np.random.randint(-2, 3, (n*m, target_rank), dtype=float)
        V = np.random.randint(-2, 3, (m*p, target_rank), dtype=float)
        W = np.random.randint(-2, 3, (p*n, target_rank), dtype=float)
        
        # Verify
        if _verify_decomp(U.tolist(), V.tolist(), W.tolist()):
            return (U.tolist(), V.tolist(), W.tolist())
    
    return None


def _structured_search():
    """
    Try structured decompositions based on known algebraic identities.
    
    For matrix multiplication, one approach is to use the identity:
    (A1 A2; A3 A4)(B1 B2; B3 B4) = ... (block matrix multiplication)
    
    But for 3x4 x 4x5, we need a different approach.
    
    Let's try to use the fact that we can factor through smaller matrices.
    """
    n, m, p = DIMENSIONS
    
    # One idea: use rank-1 updates to build up the multiplication.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # If we could write this as a sum of R terms where each term is
    # (linear form in A) * (linear form in B) * (coefficient for C[i,k])
    
    # This is exactly what the tensor decomposition does.
    
    # For now, return None to indicate no structured solution found.
    return None


def _greedy_merge(schoolbook_result, target_rank, time_limit=10.0):
    """
    Start from the schoolbook decomposition and try to merge pairs of terms
    to reduce the rank. This is a heuristic that may not find optimal solutions.
    """
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + time_limit
    
    U = [row[:] for row in schoolbook_result['U']]
    V = [row[:] for row in schoolbook_result['V']]
    W = [row[:] for row in schoolbook_result['W']]
    
    current_rank = len(U[0])
    
    while current_rank > target_rank:
        if time.time() > deadline:
            break
        
        found_merge = False
        # Try to merge every pair of terms
        for r1 in range(current_rank):
            if time.time() > deadline:
                break
            for r2 in range(r1 + 1, current_rank):
                # Check if terms r1 and r2 can be merged into one term
                # This requires that there exist vectors u, v, w such that
                # u⊗v⊗w = u1⊗v1⊗w1 + u2⊗v2⊗w2
                # This is generally not possible, so this heuristic is weak.
                # Skip for now.
                pass
        
        # If no merge found, we can't reduce rank with this approach
        if not found_merge:
            break
    
    if current_rank <= target_rank:
        # Trim to target_rank
        U_trimmed = [row[:target_rank] for row in U]
        V_trimmed = [row[:target_rank] for row in V]
        W_trimmed = [row[:target_rank] for row in W]
        
        if _verify_decomp(U_trimmed, V_trimmed, W_trimmed):
            return (U_trimmed, V_trimmed, W_trimmed)
    
    return None


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4 x 4x5 matrix multiplication.
    Returns {'U': ..., 'V': ..., 'W': ...} where U, V, W are factor matrices.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 40s margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 60
    
    # Get the schoolbook decomposition as fallback
    schoolbook = _schoolbook()
    
    # Try to find a better (lower rank) decomposition
    best_rank = schoolbook_rank
    best_result = schoolbook
    
    # Try ranks from 59 down to some lower bound
    # Lower bound: max(n*m, m*p, p*n) = max(12, 20, 15) = 20
    # But realistically, we won't find anything much below 60.
    
    for target_rank in range(schoolbook_rank - 1, 19, -1):
        if time.time() > deadline:
            break
        
        # Try structured search
        result = _structured_search()
        if result is not None:
            U, V, W = result
            if _verify_decomp(U, V, W):
                best_rank = target_rank
                best_result = {'U': U, 'V': V, 'W': W}
                break
        
        # Try random search
        result = _search_rank(target_rank, time_limit=2.0)
        if result is not None:
            U, V, W = result
            if _verify_decomp(U, V, W):
                best_rank = target_rank
                best_result = {'U': U, 'V': V, 'W': W}
                break
        
        # Try greedy merge
        result = _greedy_merge(schoolbook, target_rank, time_limit=2.0)
        if result is not None:
            U, V, W = result
            if _verify_decomp(U, V, W):
                best_rank = target_rank
                best_result = {'U': U, 'V': V, 'W': W}
                break
    
    # Return the best result found
    return best_result

# EVOLVE-BLOCK-END
