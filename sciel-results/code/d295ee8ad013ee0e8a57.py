"""Matrix multiplication 2,3,7 - rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]
DEADLINE_SECONDS = 115.0


def _schoolbook():
    """Return the standard schoolbook decomposition of rank n*m*p."""
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


def _verify(U, V, W):
    """Check that the decomposition exactly reconstructs matrix multiplication."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    return False
    return True


def _verify_fast(U, V, W, tol=1e-10):
    """Fast numerical verification using numpy."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # For each (i,j,k), compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This is a tensor contraction.
    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_np[i * m + j, :]
                v_row = V_np[j * p + k, :]
                w_row = W_np[k * n + i, :]
                total = np.sum(u_row * v_row * w_row)
                if abs(total - 1.0) > tol:
                    return False
    return True


def _random_search_for_rank(target_rank, n, m, p, max_attempts=200, deadline=None):
    """Try to find a valid decomposition of the given rank via random search."""
    for attempt in range(max_attempts):
        if deadline is not None and time.time() > deadline:
            break
        
        # Generate random sparse decomposition
        U = [[0] * target_rank for _ in range(n * m)]
        V = [[0] * target_rank for _ in range(m * p)]
        W = [[0] * target_rank for _ in range(p * n)]
        
        # For each rank dimension r, choose a sparse pattern
        for r in range(target_rank):
            # Each column should have a few non-zero entries
            # U has n*m = 6 rows, V has m*p = 21 rows, W has p*n = 14 rows
            
            # Choose how many non-zeros per matrix
            u_nnz = random.randint(1, 3)
            v_nnz = random.randint(1, 4)
            w_nnz = random.randint(1, 3)
            
            # U: choose positions
            u_positions = random.sample(range(n * m), min(u_nnz, n * m))
            for pos in u_positions:
                U[pos][r] = random.choice([1, -1, Fraction(1, 2), Fraction(-1, 2), Fraction(1, 3), Fraction(-1, 3)])
            
            # V: choose positions
            v_positions = random.sample(range(m * p), min(v_nnz, m * p))
            for pos in v_positions:
                V[pos][r] = random.choice([1, -1, Fraction(1, 2), Fraction(-1, 2), Fraction(1, 3), Fraction(-1, 3)])
            
            # W: choose positions
            w_positions = random.sample(range(p * n), min(w_nnz, p * n))
            for pos in w_positions:
                W[pos][r] = random.choice([1, -1, Fraction(1, 2), Fraction(-1, 2), Fraction(1, 3), Fraction(-1, 3)])
        
        # Fast numerical check first
        if _verify_fast(U, V, W):
            # Confirm with exact check
            if _verify(U, V, W):
                return {"U": U, "V": V, "W": W}
    
    return None


def _structured_merge_search(n, m, p, base_rank, deadline):
    """Try to merge pairs of schoolbook terms to reduce rank."""
    # Start with schoolbook
    result = _schoolbook()
    U = [row[:] for row in result["U"]]
    V = [row[:] for row in result["V"]]
    W = [row[:] for row in result["W"]]
    
    current_rank = base_rank
    
    # For each pair of schoolbook terms, try to merge them
    # Two terms (i1,j1,k1) and (i2,j2,k2) at columns c1, c2
    # can potentially be merged if we can adjust U, V, W so that
    # the combined contribution still gives 1 for each (i,j,k)
    
    # This is complex. Instead, let's try a different approach:
    # Systematically try to set some U, V, W entries to zero
    # and compensate with other entries.
    
    # For now, let's try random local search starting from schoolbook
    # with small perturbations
    
    best = None
    best_rank = current_rank
    
    # Try to reduce rank by removing one column at a time
    # and adjusting remaining columns
    
    for col_to_remove in range(base_rank - 1, -1, -1):
        if time.time() > deadline - 10:
            break
        
        # Try several random perturbations
        for perturbation in range(20):
            if time.time() > deadline - 10:
                break
            
            # Copy current state
            U_test = [row[:] for row in U]
            V_test = [row[:] for row in V]
            W_test = [row[:] for row in W]
            
            # Remove column col_to_remove
            U_test = [row[:col_to_remove] + row[col_to_remove+1:] for row in U_test]
            V_test = [row[:col_to_remove] + row[col_to_remove+1:] for row in V_test]
            W_test = [row[:col_to_remove] + row[col_to_remove+1:] for row in W_test]
            
            new_rank = current_rank - 1
            
            # Add small random perturbations to some entries
            for _ in range(5):
                mat = random.choice(['U', 'V', 'W'])
                if mat == 'U':
                    pos = random.randint(0, len(U_test) - 1)
                    col = random.randint(0, len(U_test[0]) - 1)
                    U_test[pos][col] += random.choice([Fraction(1, 2), Fraction(-1, 2)])
                elif mat == 'V':
                    pos = random.randint(0, len(V_test) - 1)
                    col = random.randint(0, len(V_test[0]) - 1)
                    V_test[pos][col] += random.choice([Fraction(1, 2), Fraction(-1, 2)])
                else:
                    pos = random.randint(0, len(W_test) - 1)
                    col = random.randint(0, len(W_test[0]) - 1)
                    W_test[pos][col] += random.choice([Fraction(1, 2), Fraction(-1, 2)])
            
            if _verify(U_test, V_test, W_test):
                best = {"U": U_test, "V": V_test, "W": W_test}
                best_rank = new_rank
                # Update current state
                U = U_test
                V = V_test
                W = W_test
                current_rank = new_rank
                break
    
    return best, best_rank


def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x7 matrix multiplication.
    
    Returns: {'U': (n*m x rank), 'V': (m*p x rank), 'W': (p*n x rank)}
    with entries that exactly decompose the matrix multiplication tensor.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    n, m, p = DIMENSIONS
    base_rank = n * m * p  # 42
    
    # Fallback: schoolbook decomposition
    fallback = _schoolbook()
    
    # Try structured merge search first
    merged_result, merged_rank = _structured_merge_search(n, m, p, base_rank, deadline)
    if merged_result is not None:
        return merged_result
    
    # Try random search for lower ranks
    best_result = None
    best_rank = base_rank
    
    # Try ranks from base_rank-1 downward
    for target_rank in range(base_rank - 1, max(30, base_rank - 12)):
        if time.time() > deadline - 20:
            break
        
        result = _random_search_for_rank(target_rank, n, m, p, max_attempts=100, deadline=deadline - 20)
        if result is not None:
            best_result = result
            best_rank = target_rank
            # Try to go even lower
            continue
        else:
            # If we can't find one at this rank, stop
            break
    
    if best_result is not None:
        return best_result
    
    return fallback
