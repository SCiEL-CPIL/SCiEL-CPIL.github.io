"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication.
    Returns {'U': list of lists, 'V': list of lists, 'W': list of lists}
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The decomposition satisfies: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin
    
    # The tensor has N = n*m*p = 24 entries
    # Each entry (i,j,k) has a constraint: sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = 1
    # 
    # Key insight: We can use a structured approach.
    # For 2x3*3x4, the known minimal rank is 16 (or possibly lower).
    # 
    # Strategy: Use a smarter initialization and local search.
    # We'll work with integers and divide by a common factor at the end.
    # 
    # Actually, let's try a different approach: we can fix some entries to simplify.
    # For each rank R, we can try to find a solution where the matrices have sparse structure.
    
    # Let's try ranks from 14 to 24
    # For each rank, use multiple random restarts with local search
    
    def check_int(U_int, V_int, W_int, n, m, p, target):
        R = len(U_int[0]) if U_int else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    ui = U_int[i*m+j]
                    vj = V_int[j*p+k]
                    wk = W_int[k*n+i]
                    for r in range(R):
                        s += ui[r] * vj[r] * wk[r]
                    if s != target:
                        return False
        return True
    
    def check_float(U, V, W, n, m, p):
        R = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def compute_errors(U_int, V_int, W_int, n, m, p, R):
        """Compute the error for each tensor entry."""
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    ui = U_int[i*m+j]
                    vj = V_int[j*p+k]
                    wk = W_int[k*n+i]
                    for r in range(R):
                        s += ui[r] * vj[r] * wk[r]
                    errors.append(s)
        return errors
    
    def max_abs_error(U_int, V_int, W_int, n, m, p, R):
        """Compute the maximum absolute error."""
        errors = compute_errors(U_int, V_int, W_int, n, m, p, R)
        return max(abs(e - 1.0) for e in errors)
    
    best_result = None
    
    for R in range(14, 25):
        if time.time() > deadline:
            break
        
        found = False
        num_attempts = 50 if R <= 16 else 30
        
        for attempt in range(num_attempts):
            if time.time() > deadline:
                break
            
            # Initialize with random small integers
            # We want the sum to be approximately 1, so entries should be small
            U_int = [[random.randint(-3, 3) for _ in range(R)] for _ in range(n * m)]
            V_int = [[random.randint(-3, 3) for _ in range(R)] for _ in range(m * p)]
            W_int = [[random.randint(-3, 3) for _ in range(R)] for _ in range(p * n)]
            
            # Check initial
            if check_int(U_int, V_int, W_int, n, m, p, 1):
                best_result = (R, U_int, V_int, W_int)
                found = True
                break
            
            # Local search: minimize max absolute error
            current_max_err = max_abs_error(U_int, V_int, W_int, n, m, p, R)
            
            # Simulated annealing style local search
            steps = 3000
            for step in range(steps):
                if time.time() > deadline:
                    break
                
                if current_max_err < 1e-10:
                    # Found a solution with integer entries that sum to 1
                    best_result = (R, U_int, V_int, W_int)
                    found = True
                    break
                
                # Pick a random matrix and entry to modify
                mat_idx = random.randint(0, 2)
                if mat_idx == 0:
                    idx = random.randint(0, n*m-1)
                    r = random.randint(0, R-1)
                    delta = random.choice([-2, -1, 1, 2, 3])
                    old_val = U_int[idx][r]
                    U_int[idx][r] = old_val + delta
                elif mat_idx == 1:
                    idx = random.randint(0, m*p-1)
                    r = random.randint(0, R-1)
                    delta = random.choice([-2, -1, 1, 2, 3])
                    old_val = V_int[idx][r]
                    V_int[idx][r] = old_val + delta
                else:
                    idx = random.randint(0, p*n-1)
                    r = random.randint(0, R-1)
                    delta = random.choice([-2, -1, 1, 2, 3])
                    old_val = W_int[idx][r]
                    W_int[idx][r] = old_val + delta
                
                new_max_err = max_abs_error(U_int, V_int, W_int, n, m, p, R)
                
                if new_max_err < current_max_err:
                    current_max_err = new_max_err
                else:
                    # Revert with some probability (accept worse moves occasionally)
                    if random.random() > 0.1:
                        if mat_idx == 0:
                            U_int[idx][r] = old_val
                        elif mat_idx == 1:
                            V_int[idx][r] = old_val
                        else:
                            W_int[idx][r] = old_val
                        current_max_err = max_abs_error(U_int, V_int, W_int, n, m, p, R)
                
                if current_max_err < 1e-10:
                    best_result = (R, U_int, V_int, W_int)
                    found = True
                    break
            
            if found:
                break
        
        if found:
            break
    
    # If no low-rank integer solution found, try to find a rational solution
    # Or fall back to schoolbook
    if best_result is None:
        # Try to find a solution using a different approach
        # Use the fact that we can have half-integer entries
        # So we can search for integer solutions with target = 2
        for R in range(14, 25):
            if time.time() > deadline:
                break
            
            found = False
            for attempt in range(30):
                if time.time() > deadline:
                    break
                
                U_int = [[random.randint(-3, 3) for _ in range(R)] for _ in range(n * m)]
                V_int = [[random.randint(-3, 3) for _ in range(R)] for _ in range(m * p)]
                W_int = [[random.randint(-3, 3) for _ in range(R)] for _ in range(p * n)]
                
                if check_int(U_int, V_int, W_int, n, m, p, 2):
                    best_result = (R, U_int, V_int, W_int, 2)
                    found = True
                    break
                
                current_max_err = max(abs(e - 2.0) for e in compute_errors(U_int, V_int, W_int, n, m, p, R))
                
                steps = 2000
                for step in range(steps):
                    if time.time() > deadline:
                        break
                    
                    if current_max_err < 1e-10:
                        best_result = (R, U_int, V_int, W_int, 2)
                        found = True
                        break
                    
                    mat_idx = random.randint(0, 2)
                    if mat_idx == 0:
                        idx = random.randint(0, n*m-1)
                        r = random.randint(0, R-1)
                        delta = random.choice([-2, -1, 1, 2, 3])
                        old_val = U_int[idx][r]
                        U_int[idx][r] = old_val + delta
                    elif mat_idx == 1:
                        idx = random.randint(0, m*p-1)
                        r = random.randint(0, R-1)
                        delta = random.choice([-2, -1, 1, 2, 3])
                        old_val = V_int[idx][r]
                        V_int[idx][r] = old_val + delta
                    else:
                        idx = random.randint(0, p*n-1)
                        r = random.randint(0, R-1)
                        delta = random.choice([-2, -1, 1, 2, 3])
                        old_val = W_int[idx][r]
                        W_int[idx][r] = old_val + delta
                    
                    new_errors = compute_errors(U_int, V_int, W_int, n, m, p, R)
                    new_max_err = max(abs(e - 2.0) for e in new_errors)
                    
                    if new_max_err < current_max_err:
                        current_max_err = new_max_err
                    else:
                        if random.random() > 0.1:
                            if mat_idx == 0:
                                U_int[idx][r] = old_val
                            elif mat_idx == 1:
                                V_int[idx][r] = old_val
                            else:
                                W_int[idx][r] = old_val
                            current_max_err = max(abs(e - 2.0) for e in compute_errors(U_int, V_int, W_int, n, m, p, R))
                    
                    if current_max_err < 1e-10:
                        best_result = (R, U_int, V_int, W_int, 2)
                        found = True
                        break
                
                if found:
                    break
            
            if found:
                break
    
    # Fallback to schoolbook
    if best_result is None:
        R = n * m * p
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Extract result
    if len(best_result) == 5:
        R, U_int, V_int, W_int, target = best_result
        scale = 1.0 / target
    else:
        R, U_int, V_int, W_int = best_result
        scale = 1.0
    
    U_out = [[x * scale for x in row] for row in U_int]
    V_out = [[x * scale for x in row] for row in V_int]
    W_out = [[x * scale for x in row] for row in W_int]
    
    # Verify
    if not check_float(U_out, V_out, W_out, n, m, p):
        # Fall back to schoolbook
        R = n * m * p
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U_out, "V": V_out, "W": W_out}

def _check_decomposition(U, V, W, n, m, p):
    """Check if U, V, W form a valid decomposition."""
    R = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(R):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _check_int_decomposition(U_int, V_int, W_int, n, m, p, target):
    """Check if U_int, V_int, W_int form a valid integer decomposition."""
    R = len(U_int[0]) if U_int else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(R):
                    s += U_int[i * m + j][r] * V_int[j * p + k][r] * W_int[k * n + i][r]
                if s != target:
                    return False
    return True
# EVOLVE-BLOCK-END
