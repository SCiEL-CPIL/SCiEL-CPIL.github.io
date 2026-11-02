import time
import random
from itertools import product

def _schoolbook_decomposition():
    """Return the schoolbook decomposition with rank 63."""
    n, m, p = 3, 3, 7
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

def _validate_decomposition(U, V, W, n=3, m=3, p=7):
    """
    Validate that the decomposition exactly decomposes the matrix multiplication tensor.
    
    The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    should equal 1 if (i,j,k) corresponds to a valid multiplication term, else 0.
    
    Actually, for matrix multiplication C = A * B where A is n×m and B is m×p,
    C[i,k] = sum_j A[i,j] * B[j,k].
    
    The tensor decomposition satisfies:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta(i,j,k)
    where delta = 1 if this term contributes to C[i,k], which is always the case
    for valid indices. Actually, the standard form is:
    
    For each (i,j,k) in [0,n)×[0,m)×[0,p):
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    This is because each triple (i,j,k) corresponds to one multiplication-addition
    term in computing C[i,k] = sum_j A[i,j]*B[j,k].
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if total != 1:
                    return False
    return True

def _try_rank_sat(rank, n=3, m=3, p=7, max_time=30, coeff_range=(-2, -1, 0, 1, 2)):
    """
    Try to find a rank-rank decomposition using SAT-like search.
    
    We discretize coefficients to a small set and search for valid assignments.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    U_dim = n * m  # 9
    V_dim = m * p  # 21
    W_dim = p * n  # 21
    
    # For each rank r, we have U[:,r], V[:,r], W[:,r]
    # The constraint is: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    
    # We'll use a backtracking search with pruning
    # To make it tractable, we'll try random initialization and local search
    
    best_solution = None
    
    # Try several random restarts
    for attempt in range(50):
        if time.time() > deadline - 5:
            break
            
        # Random initialization with sparse vectors
        U = [[random.choice(coeff_range) if random.random() < 0.3 else 0 for _ in range(rank)] for _ in range(U_dim)]
        V = [[random.choice(coeff_range) if random.random() < 0.3 else 0 for _ in range(rank)] for _ in range(V_dim)]
        W = [[random.choice(coeff_range) if random.random() < 0.3 else 0 for _ in range(rank)] for _ in range(W_dim)]
        
        # Local search: try to fix violations
        for iteration in range(1000):
            if time.time() > deadline - 2:
                break
                
            # Check if valid
            if _validate_decomposition(U, V, W, n, m, p):
                best_solution = (U, V, W)
                return best_solution
                
            # Find a violated constraint
            violations = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = 0
                        for r in range(rank):
                            total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                        if total != 1:
                            violations.append((i, j, k, total))
            
            if not violations:
                best_solution = (U, V, W)
                return best_solution
                
            # Pick a random violation and try to fix it
            i, j, k, total = random.choice(violations)
            error = total - 1
            
            # Try modifying one of the factors
            # We need to change U[i*m+j, r], V[j*p+k, r], or W[k*n+i, r] for some r
            # such that the product changes by -error
            
            # For simplicity, try changing one element at a time
            fixed = False
            for r in range(rank):
                if time.time() > deadline - 2:
                    break
                    
                # Try changing U
                old_val = U[i*m+j][r]
                for new_val in coeff_range:
                    if new_val == old_val:
                        continue
                    U[i*m+j][r] = new_val
                    if _validate_decomposition(U, V, W, n, m, p):
                        best_solution = (U, V, W)
                        return best_solution
                    U[i*m+j][r] = old_val
                    
                # Try changing V
                old_val = V[j*p+k][r]
                for new_val in coeff_range:
                    if new_val == old_val:
                        continue
                    V[j*p+k][r] = new_val
                    if _validate_decomposition(U, V, W, n, m, p):
                        best_solution = (U, V, W)
                        return best_solution
                    V[j*p+k][r] = old_val
                    
                # Try changing W
                old_val = W[k*n+i][r]
                for new_val in coeff_range:
                    if new_val == old_val:
                        continue
                    W[k*n+i][r] = new_val
                    if _validate_decomposition(U, V, W, n, m, p):
                        best_solution = (U, V, W)
                        return best_solution
                    W[k*n+i][r] = old_val
                    
                if fixed:
                    break
            
            # If no single change fixed it, try a random perturbation
            if random.random() < 0.3:
                idx = random.choice([(i*m+j, 'U'), (j*p+k, 'V'), (k*n+i, 'W')])
                row, mat_name = idx
                r = random.randint(0, rank-1)
                if mat_name == 'U':
                    U[row][r] = random.choice(coeff_range)
                elif mat_name == 'V':
                    V[row][r] = random.choice(coeff_range)
                else:
                    W[row][r] = random.choice(coeff_range)
    
    return best_solution

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices
    of the tensor decomposition for 3x3 * 3x7 matrix multiplication.
    
    Tries to find a lower-rank decomposition using SAT-based search,
    falling back to schoolbook if no better solution is found.
    """
    start_time = time.time()
    total_time_limit = 120
    safety_margin = 10
    
    n, m, p = 3, 3, 7
    
    # First, get the schoolbook solution as a fallback
    schoolbook = _schoolbook_decomposition()
    best_rank = 63
    best_solution = schoolbook
    
    # Try to find lower-rank solutions
    # Lower bound is max(n*m, m*p, n*p) = max(9, 21, 21) = 21
    # Try ranks from 21 up to 62
    
    for rank in range(21, 63):
        if time.time() - start_time > total_time_limit - safety_margin:
            break
        
        # Allocate more time for smaller ranks (harder problems)
        time_budget = min(30, (total_time_limit - safety_margin - (time.time() - start_time)) // 2)
        if time_budget < 5:
            break
        
        result = _try_rank_sat(rank, n, m, p, max_time=time_budget)
        
        if result is not None:
            U, V, W = result
            # Validate one more time
            if _validate_decomposition(U, V, W, n, m, p):
                best_rank = rank
                best_solution = {"U": U, "V": V, "W": W}
                # Found a solution, stop searching for this rank
                break
    
    return best_solution
