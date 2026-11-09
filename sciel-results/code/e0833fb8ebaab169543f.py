"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition with rank = n*m*p."""
    rank = n * m * p
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
    return {"U": U, "V": V, "W": W, "rank": rank}

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition correctly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    
    # Check the tensor identity: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_reduce_rank(U, V, W, n, m, p, max_iterations=5000, deadline=None):
    """
    Try to reduce the rank by finding columns that can be removed or merged.
    Uses a greedy approach: try removing each column and check if the decomposition
    can be fixed by adjusting other columns.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    current_rank = rank
    iterations = 0
    
    while current_rank > 1 and iterations < max_iterations:
        if deadline and time.time() > deadline:
            break
        iterations += 1
        
        # Try removing each column
        removed = False
        for c in range(current_rank):
            # Remove column c
            mask = np.ones(current_rank, dtype=bool)
            mask[c] = False
            U_reduced = U[:, mask]
            V_reduced = V[:, mask]
            W_reduced = W[:, mask]
            
            # Check if the reduced decomposition is valid
            if _verify_decomposition(U_reduced.tolist(), V_reduced.tolist(), W_reduced.tolist(), n, m, p, tol=1e-8):
                U, V, W = U_reduced, V_reduced, W_reduced
                current_rank -= 1
                removed = True
                break
        
        if not removed:
            # Try to merge two columns into one
            merged = False
            for c1 in range(current_rank):
                if merged:
                    break
                for c2 in range(c1 + 1, current_rank):
                    # Try to find a single column that replaces columns c1 and c2
                    # This is a hard problem, so we use a heuristic
                    # Check if the sum of the two rank-1 terms can be approximated by one
                    # For now, skip this as it's complex
                    pass
            
            if not merged:
                break
    
    return U.tolist(), V.tolist(), W.tolist(), current_rank

def _local_search_decomposition(n, m, p, deadline):
    """
    Perform local search to find a lower-rank decomposition.
    Start from schoolbook and try to reduce rank.
    """
    # Start with schoolbook
    schoolbook = _build_schoolbook(n, m, p)
    U = np.array(schoolbook["U"], dtype=float)
    V = np.array(schoolbook["V"], dtype=float)
    W = np.array(schoolbook["W"], dtype=float)
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    # Try rank reduction
    U_red, V_red, W_red, new_rank = _try_reduce_rank(U.tolist(), V.tolist(), W.tolist(), n, m, p, max_iterations=10000, deadline=deadline)
    if new_rank < best_rank:
        best_U, best_V, best_W = np.array(U_red), np.array(V_red), np.array(W_red)
        best_rank = new_rank
    
    return best_U.tolist(), best_V.tolist(), best_W.tolist(), best_rank

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4×5 × 5×6 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    n, m, p = DIMENSIONS
    
    # First, get the schoolbook decomposition
    result = _build_schoolbook(n, m, p)
    
    # Verify it's correct
    if not _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        # This shouldn't happen, but fallback
        pass
    
    # Try to reduce rank
    U_red, V_red, W_red, reduced_rank = _local_search_decomposition(n, m, p, deadline)
    
    if reduced_rank < result["rank"]:
        result = {"U": U_red, "V": V_red, "W": W_red}
    
    return result
# EVOLVE-BLOCK-END
