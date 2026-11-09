"""Combined program: structural schoolbook base with index-convention verification."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    U = np.array(U, dtype=np.int64)
    V = np.array(V, dtype=np.int64)
    W = np.array(W, dtype=np.int64)
    
    # Check shapes
    if U.shape != (n * m, rank):
        return False
    if V.shape != (m * p, rank):
        return False
    if W.shape != (p * n, rank):
        return False
    
    # For each (a, b, c) where a = i*m+j, b = j*p+k, c = k'*n+i'
    # The tensor value T[i,j,k] = 1 if and only if the indices are consistent
    # i.e., i from a matches i' from c, j from a matches j from b, k from b matches k' from c
    for a in range(n * m):
        i_a = a // m
        j_a = a % m
        for b in range(m * p):
            j_b = b // p
            k_b = b % p
            for c in range(p * n):
                k_c = c // n
                i_c = c % n
                
                # Expected value: 1 if indices are consistent, 0 otherwise
                expected = 1 if (i_a == i_c and j_a == j_b and k_b == k_c) else 0
                
                # Compute sum_r U[a,r] * V[b,r] * W[c,r]
                total = np.sum(U[a, :] * V[b, :] * W[c, :])
                
                if total != expected:
                    return False
    
    return True

def _schoolbook_decomposition(n, m, p):
    """Generate the schoolbook decomposition with rank = n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * (m * p) + j * p + k
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, r] = 1
                V[v_idx, r] = 1
                W[w_idx, r] = 1
    
    return U.tolist(), V.tolist(), W.tolist(), rank

def _try_rank_reduction(U, V, W, n, m, p, target_rank, deadline, max_iterations=50):
    """
    Try to reduce the rank from current to target_rank by finding linear dependencies.
    Uses a greedy approach: try to eliminate rank-1 components by checking if they
    can be expressed as linear combinations of others.
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    current_rank = U.shape[1]
    
    if current_rank <= target_rank:
        return U.tolist(), V.tolist(), W.tolist(), current_rank
    
    # Try to find a subset of rank columns that spans the same space
    # Use a simple approach: try removing one column at a time if it's linearly dependent
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        if U.shape[1] <= target_rank:
            break
        
        # Try to find a column that can be removed
        removed = False
        for r in range(U.shape[1]):
            if time.time() > deadline:
                break
            
            # Check if column r is in the span of other columns
            # For each factor, check if the column is linearly dependent
            cols_U = np.delete(U, r, axis=1)
            cols_V = np.delete(V, r, axis=1)
            cols_W = np.delete(W, r, axis=1)
            
            # Check if U[:, r] is in span of U without column r
            try:
                rank_U_full = np.linalg.matrix_rank(np.column_stack([U[:, r], cols_U]))
                rank_U_reduced = np.linalg.matrix_rank(cols_U)
                
                rank_V_full = np.linalg.matrix_rank(np.column_stack([V[:, r], cols_V]))
                rank_V_reduced = np.linalg.matrix_rank(cols_V)
                
                rank_W_full = np.linalg.matrix_rank(np.column_stack([W[:, r], cols_W]))
                rank_W_reduced = np.linalg.matrix_rank(cols_W)
                
                if rank_U_full == rank_U_reduced and rank_V_full == rank_V_reduced and rank_W_full == rank_W_reduced:
                    # Column r is linearly dependent in all three factors
                    U = cols_U
                    V = cols_V
                    W = cols_W
                    removed = True
                    break
            except Exception:
                continue
        
        if not removed:
            break
    
    return U.tolist(), V.tolist(), W.tolist(), U.shape[1]

def _randomized_rank_search(n, m, p, target_rank, deadline, num_trials=100):
    """
    Try randomized approach: generate random factor matrices of rank target_rank
    and check if they can be adjusted to satisfy the tensor equation.
    """
    best_result = None
    
    for trial in range(num_trials):
        if time.time() > deadline:
            break
        
        # Generate random factors with small integer/half-integer entries
        np.random.seed(trial)
        
        U = np.random.randint(-2, 3, size=(n * m, target_rank))
        V = np.random.randint(-2, 3, size=(m * p, target_rank))
        W = np.random.randint(-2, 3, size=(p * n, target_rank))
        
        # Check if this decomposition is valid
        if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p, target_rank):
            return U.tolist(), V.tolist(), W.tolist(), target_rank
    
    return None

def _structured_search(n, m, p, deadline):
    """
    Structured search for lower-rank decompositions.
    Tries various strategies to find rank < n*m*p.
    """
    # Start with schoolbook
    U, V, W, rank = _schoolbook_decomposition(n, m, p)
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try to reduce rank incrementally
    for target_rank in range(best_rank - 1, max(15, best_rank // 2), -1):
        if time.time() > deadline:
            break
        
        # Try rank reduction
        result = _try_rank_reduction(best_U, best_V, best_W, n, m, p, target_rank, deadline)
        if result:
            U, V, W, new_rank = result
            if new_rank < best_rank:
                best_rank = new_rank
                best_U, best_V, best_W = U, V, W
    
    # Try randomized search for lower ranks
    for target_rank in range(max(15, best_rank // 2), 10, -1):
        if time.time() > deadline:
            break
        
        result = _randomized_rank_search(n, m, p, target_rank, deadline, num_trials=20)
        if result:
            U, V, W, new_rank = result
            if new_rank < best_rank:
                best_rank = new_rank
                best_U, best_V, best_W = U, V, W
                break  # Found a lower rank, stop searching
    
    return best_U, best_V, best_W, best_rank

def construct():
    deadline = time.time() + 110.0  # safety margin before 150s limit
    
    n, m, p = 3, 4, 5
    
    # First, get the schoolbook decomposition as a safe fallback
    U_school, V_school, W_school, rank_school = _schoolbook_decomposition(n, m, p)
    
    # Verify schoolbook is correct
    if not _verify_decomposition(U_school, V_school, W_school, n, m, p, rank_school):
        # This should never happen, but if it does, we have a bug
        return {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to find a lower-rank decomposition
    try:
        U_best, V_best, W_best, rank_best = _structured_search(n, m, p, deadline)
        
        # Final verification
        if _verify_decomposition(U_best, V_best, W_best, n, m, p, rank_best):
            return {"U": U_best, "V": V_best, "W": W_best}
        else:
            # Fallback to schoolbook
            return {"U": U_school, "V": V_school, "W": W_school}
    except Exception as e:
        # Any error, fallback to schoolbook
        return {"U": U_school, "V": V_school, "W": W_school}

# EVOLVE-BLOCK-END
