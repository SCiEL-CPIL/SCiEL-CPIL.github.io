"""Matrix multiplication 2x4x4 with Strassen-based rank-28 decomposition and finite-field rank search."""
import time
import random

DIMENSIONS = [2, 4, 4]

def _strassen_2x2():
    """
    Return the 7 Strassen multiplication patterns for 2x2 matrix multiplication.
    Each pattern is a dict with keys 'a', 'b', 'c' mapping local indices to coefficients.
    Local indices are 0..3 for 2x2 matrices (row-major).
    """
    patterns = [
        {'a': {0: 1, 3: 1}, 'b': {0: 1, 3: 1}, 'c': {0: 1, 3: 1}},
        {'a': {2: 1, 3: 1}, 'b': {0: 1}, 'c': {2: 1, 3: -1}},
        {'a': {0: 1}, 'b': {1: 1, 3: -1}, 'c': {1: 1, 3: 1}},
        {'a': {3: 1}, 'b': {2: 1, 0: -1}, 'c': {0: 1, 2: 1}},
        {'a': {0: 1, 1: 1}, 'b': {3: 1}, 'c': {0: -1, 1: 1}},
        {'a': {2: 1, 0: -1}, 'b': {0: 1, 1: 1}, 'c': {3: 1}},
        {'a': {1: 1, 3: -1}, 'b': {2: 1, 3: 1}, 'c': {0: 1}},
    ]
    return patterns

def _construct_strassen_28():
    """Construct the base rank-28 Strassen decomposition."""
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_patterns = _strassen_2x2()
    blocks = [
        (0, 0, 0, 0, 0, 0),
        (0, 2, 2, 0, 0, 0),
        (0, 0, 0, 2, 0, 2),
        (0, 2, 2, 2, 0, 2),
    ]
    
    col = 0
    for (a_r, a_c, b_r, b_c, c_r, c_c) in blocks:
        for pattern in strassen_patterns:
            for local_a_idx, coeff in pattern['a'].items():
                local_r = local_a_idx // 2
                local_c = local_a_idx % 2
                global_i = a_r + local_r
                global_j = a_c + local_c
                U[global_i * m + global_j][col] += coeff
            for local_b_idx, coeff in pattern['b'].items():
                local_r = local_b_idx // 2
                local_c = local_b_idx % 2
                global_j = b_r + local_r
                global_k = b_c + local_c
                V[global_j * p + global_k][col] += coeff
            for local_c_idx, coeff in pattern['c'].items():
                local_r = local_c_idx // 2
                local_c = local_c_idx % 2
                global_i = c_r + local_r
                global_k = c_c + local_c
                W[global_k * n + global_i][col] += coeff
            col += 1
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Check dimensions
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    for r in range(rank):
        if len(U[r]) != rank:
            return False
        if len(V[r]) != rank:
            return False
        if len(W[r]) != rank:
            return False
    
    # Check tensor: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # T[i,j,k] should be 1 if i==k, 0 otherwise
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                expected = 1 if i == k else 0
                if val != expected:
                    return False
    return True

def _search_lower_rank(fallback_result, max_rank=27, min_rank=20, time_limit=100.0):
    """
    Search for a rank decomposition lower than the fallback rank using finite-field methods.
    Returns a result dict if found, else None.
    """
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + time_limit
    
    # The tensor T[i,j,k] = 1 if i==k else 0
    # We want U (n*m x r), V (m*p x r), W (p*n x r) such that
    # sum_{r'} U[i*m+j, r'] * V[j*p+k, r'] * W[k*n+i, r'] = T[i,j,k]
    
    # Strategy: try to find solutions over a finite field, then check if they lift to integers
    # For each candidate rank r, we have 32r unknowns and 8 independent equations (one for each (i,j) with i in 0..1, j in 0..3, requiring sum_r U[i*m+j,r]*V[j*p+i,r]*W[i*n+i,r] = 1, and similar for i!=k)
    
    # Actually, the equations are:
    # For each i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if i==k else 0
    # But many of these are zero. The nonzero ones are when i==k, i in {0,1}, k in {0,1}, so i=k=0 or i=k=1, and j in {0,1,2,3}.
    # So we have 8 equations: for i=k=0: sum_r U[0*m+j, r]*V[j*p+0, r]*W[0*n+0, r] = 1 for j=0..3
    #                  for i=k=1: sum_r U[1*m+j, r]*V[j*p+1, r]*W[1*n+1, r] = 1 for j=0..3
    # And 24 zero equations.
    
    # This is a non-linear system (products of three variables). Hard to solve directly.
    
    # Alternative: fix U and V, then W is determined linearly.
    # For fixed U and V, the equation becomes:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    # For each (i,k), this is: sum_j (for the specific j) ... no, it's for each j independently.
    # Actually, for fixed i,k, and for each j:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    # This is linear in W[k*n+i, r] for r=0..r-1.
    # So for fixed U,V, we have a linear system for W.
    
    # Let's try: for random U,V over integers in {-1,0,1}, solve for W and check if W has integer entries.
    
    best_result = None
    
    for r in range(min_rank, max_rank + 1):
        if time.time() > deadline - 10:
            break
        
        attempts = 0
        max_attempts = 500
        while attempts < max_attempts and time.time() < deadline - 10:
            attempts += 1
            
            # Generate random U and V with entries in {-1, 0, 1}
            random.seed(attempts * 12345 + r)
            U = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(n * m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(m * p)]
            
            # Solve for W: for each i in {0,1}, k in {0,1}:
            #   For each j in {0,1,2,3}:
            #     sum_{r'} U[i*m+j, r'] * V[j*p+k, r'] * W[k*n+i, r'] = T[i,j,k]
            # This gives 4 equations for 4 unknowns... no, r unknowns for 4 equations.
            # Actually, for fixed (i,k), we have 4 equations (one per j) and r unknowns (W[k*n+i, 0..r-1]).
            # If r > 4, the system is underdetermined. We need to find integer solutions.
            
            # Let's set up the linear system for W[k*n+i, :] given (i,k).
            # A_{j, r'} = U[i*m+j, r'] * V[j*p+k, r']
            # b_j = T[i,j,k] = 1 if i==k else 0
            # We need A * W_col = b, where W_col is W[k*n+i, :]
            
            # For i != k, b = 0, so we need A * W_col = 0.
            # For i == k, b_j = 1 for all j, so we need A * W_col = [1,1,1,1].
            
            W = [[0] * r for _ in range(p * n)]
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Build system A * w = b
                    A = []
                    b = []
                    for j in range(m):
                        row = [U[i * m + j][r_prime] * V[j * p + k][r_prime] for r_prime in range(r)]
                        A.append(row)
                        b.append(1 if i == k else 0)
                    
                    # Solve A * w = b using Gaussian elimination over rationals
                    # We'll use exact arithmetic
                    w = _solve_linear_system(A, b)
                    if w is None:
                        valid = False
                        break
                    # Check if w has integer or half-integer entries
                    W[k * n + i] = w
                if not valid:
                    break
            
            if valid:
                # Verify the full decomposition
                if _verify_decomposition(U, V, W, n, m, p):
                    best_result = {"U": U, "V": V, "W": W}
                    return best_result
        
        if best_result is not None:
            break
    
    return best_result

def _solve_linear_system(A, b):
    """
    Solve A * x = b using Gaussian elimination with exact rational arithmetic.
    A is a list of lists (m x n), b is a list of length m.
    Returns x (list of length n) or None if no solution.
    Uses fractions for exactness.
    """
    from fractions import Fraction
    
    m = len(A)
    n = len(A[0]) if m > 0 else 0
    
    # Augmented matrix
    aug = [[Fraction(A[i][j]) for j in range(n)] + [Fraction(b[i])] for i in range(m)]
    
    # Gaussian elimination
    rank = 0
    for col in range(n):
        # Find pivot
        pivot = -1
        for row in range(rank, m):
            if aug[row][col] != 0:
                pivot = row
                break
        if pivot == -1:
            continue
        
        # Swap rows
        if pivot != rank:
            aug[rank], aug[pivot] = aug[pivot], aug[rank]
        
        # Eliminate below
        for row in range(rank + 1, m):
            if aug[row][col] != 0:
                factor = aug[row][col] / aug[rank][col]
                for c in range(col, n + 1):
                    aug[row][c] -= factor * aug[rank][c]
        
        rank += 1
    
    # Check for inconsistency
    for row in range(rank, m):
        if aug[row][n] != 0:
            return None  # Inconsistent
    
    # Back substitution
    x = [Fraction(0) for _ in range(n)]
    for i in range(rank - 1, -1, -1):
        # Find the column for this row
        col = -1
        for c in range(n):
            if aug[i][c] != 0:
                col = c
                break
        if col == -1:
            continue
        s = aug[i][n]
        for c in range(col + 1, n):
            s -= aug[i][c] * x[c]
        x[col] = s / aug[i][col]
    
    # Check if all entries are integer or half-integer
    for val in x:
        num = val.numerator
        den = val.denominator
        if den not in (1, 2):
            return None
        if den == 2 and num % 2 != 0:
            # Half-integer, OK
            pass
    
    # Convert to int or Fraction
    result = []
    for val in x:
        if val.denominator == 1:
            result.append(int(val.numerator))
        else:
            result.append(val)
    
    return result

def construct():
    """
    Construct a minimal-rank tensor decomposition for 2x4 * 4x4 matrix multiplication.
    The output factor W follows the official trace convention W[k*n+i,r].
    """
    t0 = time.time()
    deadline = 120.0
    
    # Start with the known rank-28 Strassen solution
    result = _construct_strassen_28()
    
    # Try to find a lower rank solution
    if time.time() < deadline - 30:
        better = _search_lower_rank(result, max_rank=27, min_rank=20, time_limit=min(90.0, deadline - time.time() - 10))
        if better is not None:
            result = better
    
    return result
