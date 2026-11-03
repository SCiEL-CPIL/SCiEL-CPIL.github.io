"""Matrix multiplication 3x3 by 3x4 - rank minimization via structured search with smooth surrogate."""
import time
import math
import random
import numpy as np

DIMENSIONS = [3, 3, 4]

_START_TIME = time.time()
DEADLINE = _START_TIME + 105  # 15s safety margin before 120s limit

def _time_remaining():
    return DEADLINE - time.time()

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix
    by a 3x4 matrix.
    
    The decomposition requires: for all i,j,k
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    with integer/half-integer entries.
    
    Strategy:
    1. Start from schoolbook (rank 36) and try to merge columns
    2. Use structured initialization exploiting the diagonal tensor structure
    3. Local search with smooth surrogate objective
    4. Value set: half-integers {-2, -3/2, -1, -1/2, 0, 1/2, 1, 3/2, 2}
    """
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Schoolbook fallback (rank 36)
    rank_school = n * m * p
    U_sb = [[0] * rank_school for _ in range(n * m)]
    V_sb = [[0] * rank_school for _ in range(m * p)]
    W_sb = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_school
    
    # Half-integer value set
    half_ints = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    
    # Try to find lower rank decompositions
    # Lower bound: max(n*m, m*p, p*n) = max(9, 12, 12) = 12
    # For 3x3x3, rank is 23. For 3x3x4, likely 24-28.
    ranks_to_try = list(range(24, 36)) + list(range(18, 24)) + list(range(14, 18))
    
    for target_rank in ranks_to_try:
        if _time_remaining() < 5:
            break
        
        if best_rank <= target_rank:
            continue
        
        # More attempts for lower ranks (harder)
        max_attempts = 6 if target_rank >= 28 else (10 if target_rank >= 24 else 15)
        if _time_remaining() < 15:
            max_attempts = min(max_attempts, 3)
        
        for attempt in range(max_attempts):
            if _time_remaining() < 2:
                break
            
            try:
                result = _try_structured_search(n, m, p, target_rank, attempt * 7919 + 104729, half_ints)
                if result is not None:
                    best_result = result
                    best_rank = target_rank
                    break
            except Exception:
                continue
    
    return best_result

def _compute_decomposition_error(U, V, W, n, m, p):
    """Compute sum of squared errors: sum_{i,j,k} (sum_c U*V*W - 1)^2"""
    r = U.shape[1]
    total_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                total_err += (s - 1.0) ** 2
    return total_err

def _is_half_integer(x, tol=1e-9):
    """Check if x is a half-integer (k/2 for integer k)."""
    return abs(x * 2 - round(x * 2)) < tol

def _smooth_badness(U, V, W, n, m, p, half_ints):
    """
    Smooth surrogate objective:
    - Squared decomposition error
    - Penalty for non-half-integer values (smooth: distance to nearest half-integer squared)
    """
    r = U.shape[1]
    decomp_err = _compute_decomposition_error(U, V, W, n, m, p)
    
    # Penalty for non-half-integer values
    def _half_int_penalty(matrix):
        pen = 0.0
        for val in matrix.flatten():
            # Distance to nearest half-integer
            nearest = round(val * 2) / 2.0
            pen += (val - nearest) ** 2
        return pen
    
    penalty = _half_int_penalty(U) + _half_int_penalty(V) + _half_int_penalty(W)
    return decomp_err + 0.1 * penalty

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify exact decomposition: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1"""
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(r):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(total - 1) > tol:
                    return False
    return True

def _try_structured_search(n, m, p, target_rank, seed, half_ints):
    """
    Structured search: initialize with schoolbook-like structure, then local search.
    """
    np.random.seed(seed)
    random.seed(seed)
    r = target_rank
    
    # Initialize with structured pattern:
    # For each (i,j,k) pair, we need a "slot". With rank < n*m*p, we must merge.
    # Strategy: assign each (i,j,k) to a column based on a hash, then adjust values.
    
    # Alternative: start with random half-integer initialization
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # Random initialization from half_ints
    for c in range(r):
        # Each column gets a random pattern
        U[:, c] = np.random.choice(half_ints, size=n * m)
        V[:, c] = np.random.choice(half_ints, size=m * p)
        W[:, c] = np.random.choice(half_ints, size=p * n)
    
    # Ensure no column is trivially zero
    for c in range(r):
        if np.all(U[:, c] == 0):
            U[np.random.randint(0, n * m), c] = 1
        if np.all(V[:, c] == 0):
            V[np.random.randint(0, m * p), c] = 1
        if np.all(W[:, c] == 0):
            W[np.random.randint(0, p * n), c] = 1
    
    current_score = _smooth_badness(U, V, W, n, m, p, half_ints)
    
    # Local search: modify individual entries
    max_iters = 2000
    patience = 0
    max_patience = 200
    
    for it in range(max_iters):
        if _time_remaining() < 1:
            break
        if patience > max_patience:
            break
        
        # Pick a random matrix and entry to modify
        mat_choice = np.random.randint(0, 3)
        if mat_choice == 0:
            mat = U
            rows = n * m
        elif mat_choice == 1:
            mat = V
            rows = m * p
        else:
            mat = W
            rows = p * n
        
        idx = np.random.randint(0, rows)
        col = np.random.randint(0, r)
        old_val = mat[idx, col]
        
        # Try a new value from half_ints (or nearby)
        new_val = np.random.choice(half_ints)
        if new_val == old_val:
            continue
        
        mat[idx, col] = new_val
        new_score = _smooth_badness(U, V, W, n, m, p, half_ints)
        
        if new_score < current_score:
            current_score = new_score
            patience = 0
        else:
            mat[idx, col] = old_val
            patience += 1
    
    # Check if we have a valid solution
    # Round to nearest half-integer and verify
    U_rounded = np.round(U * 2) / 2.0
    V_rounded = np.round(V * 2) / 2.0
    W_rounded = np.round(W * 2) / 2.0
    
    if _is_half_integer(U_rounded[0, 0]) and _verify(
        U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p
    ):
        return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
    
    # Try a more aggressive approach: column merging from schoolbook
    result = _try_column_merging(n, m, p, target_rank, seed, half_ints)
    if result is not None:
        return result
    
    return None

def _try_column_merging(n, m, p, target_rank, seed, half_ints):
    """
    Start from schoolbook (rank 36) and try to merge columns to reduce rank.
    Two schoolbook columns (i1,j1,k1) and (i2,j2,k2) can be merged if
    there exist coefficients a, b such that the combined column works.
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # Schoolbook columns: each (i,j,k) is one column
    # Column (i,j,k): U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1, rest 0
    
    # Try to merge pairs of columns
    # For merging columns c1=(i1,j1,k1) and c2=(i2,j2,k2) with weights a, b:
    # The new column has:
    #   U[i1*m+j1] += a, U[i2*m+j2] += b
    #   V[j1*p+k1] += a, V[j2*p+k2] += b
    #   W[k1*n+i1] += a, W[k2*n+i2] += b
    # This works if for all (i,j,k):
    #   a * [i==i1 and j==j1 and k==k1] + b * [i==i2 and j==j2 and k==k2] = [i==i and j==j and k==k]
    # This only works if (i1,j1,k1) == (i,j,k) or (i2,j2,k2) == (i,j,k), so no merging is possible
    # with just 2 columns. We need more complex combinations.
    
    # Alternative: use a different ansatz
    # For rank r, try to find U, V, W such that the decomposition works.
    # Use a structured form based on the indices.
    
    # Ansatz: U[i*m+j, c] depends on i, j, c
    #         V[j*p+k, c] depends on j, k, c
    #         W[k*n+i, c] depends on k, i, c
    
    # For 3x3x4, try a form where:
    # U[i*m+j, c] = u_i * v_j for some factors
    # But this is too restrictive.
    
    # Let's try a direct search with the smooth objective but better initialization.
    r = target_rank
    
    # Initialize with a pattern that respects the structure:
    # For column c, assign it to "handle" certain (i,j,k) triples
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # Assign each (i,j,k) to a column
    assignments = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = (i * m + j) * p + k
                col = col % r  # Wrap around
                if col not in assignments:
                    assignments[col] = []
                assignments[col].append((i, j, k))
    
    # For each column, set values to satisfy the assigned (i,j,k) triples
    for col, triples in assignments.items():
        if len(triples) == 1:
            i, j, k = triples[0]
            U[i * m + j, col] = 1
            V[j * p + k, col] = 1
            W[k * n + i, col] = 1
        else:
            # Multiple triples in one column - need to solve
            # For each (i,j,k) in triples: U[i*m+j]*V[j*p+k]*W[k*n+i] = 1
            # This is a system of equations. Try to find a solution.
            # Simple approach: set all involved entries to 1 and hope it works
            # (it won't in general, but it's a starting point)
            for i, j, k in triples:
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
    
    # Now do local search
    current_score = _smooth_badness(U, V, W, n, m, p, half_ints)
    
    max_iters = 1500
    patience = 0
    max_patience = 150
    
    for it in range(max_iters):
        if _time_remaining() < 1:
            break
        if patience > max_patience:
            break
        
        mat_choice = np.random.randint(0, 3)
        if mat_choice == 0:
            mat = U
            rows = n * m
        elif mat_choice == 1:
            mat = V
            rows = m * p
        else:
            mat = W
            rows = p * n
        
        idx = np.random.randint(0, rows)
        col = np.random.randint(0, r)
        old_val = mat[idx, col]
        
        new_val = np.random.choice(half_ints)
        if new_val == old_val:
            continue
        
        mat[idx, col] = new_val
        new_score = _smooth_badness(U, V, W, n, m, p, half_ints)
        
        if new_score < current_score:
            current_score = new_score
            patience = 0
        else:
            mat[idx, col] = old_val
            patience += 1
    
    # Check if we have a valid solution
    U_rounded = np.round(U * 2) / 2.0
    V_rounded = np.round(V * 2) / 2.0
    W_rounded = np.round(W * 2) / 2.0
    
    if _verify(
        U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p
    ):
        return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
    
    return None

if __name__ == "__main__":
    result = construct()
    rank = len(result["U"][0])
    print(f"Found decomposition of rank {rank}")
