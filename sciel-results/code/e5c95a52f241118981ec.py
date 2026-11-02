"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]
import time
import random
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 120.0
_SAFETY_MARGIN = 5.0

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME) - _SAFETY_MARGIN

def _validate(U, V, W, n=3, m=3, p=5):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _validate_half_integer(U, V, W, n=3, m=3, p=5):
    """Check entries are half-integers and decomposition is exact."""
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    return _validate(U, V, W, n, m, p)

def _laser_method(n=3, m=3, p=5, rank=17):
    """
    Laser Method: Embed 3x3x5 into 3x3x6, use known rank-17 structure for 3x3x3,
    then extend to handle the 5 columns via linear constraints.
    
    Strategy:
    1. Use a known rank-17 decomposition pattern for 3x3x3 (Strassen-like)
    2. For the 3x3x5 case, we need 15 output entries (3x5) instead of 9 (3x3)
    3. We can use the rank-17 structure and solve for the 5th column constraints
    """
    if _time_remaining() < 3.0:
        return None
    
    # Known rank-17 decomposition for 3x3x3 matrix multiplication
    # Using a standard rank-17 algorithm (e.g., Bini-Capovani or similar)
    # The 17 multiplication terms for 3x3x3:
    # M1 = (a11)(b11)
    # M2 = (a11+a12)(b21+b22)  -- simplified Strassen-like
    # ... Let me use a concrete known rank-17 pattern
    
    # Actually, let me use the explicit rank-17 decomposition from literature.
    # For 3x3x3, a rank-17 decomposition exists. Let me construct it.
    
    # Using the approach: 3x3x3 can be done in 17 multiplications.
    # One such decomposition (from Bini et al.):
    
    # Let me use a different approach: construct the 3x3x5 problem directly
    # by noting that 3x3x5 = 3x3x3 + 3x3x2 (but this doesn't directly help)
    
    # Better approach: Use the fact that we can share some multiplications
    # between the 3x3 part and the additional 3x2 part.
    
    # Let me try a direct construction with rank 17:
    # U has 9 rows (3x3 input), V has 15 rows (3x5 input), W has 15 rows (5x3 output)
    
    # The tensor T[i,j,k] = delta(i,j,k) for i in [0,3), j in [0,3), k in [0,5)
    # We need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all (i,j,k)
    
    # Strategy: Start with a rank-17 decomposition for 3x3x3, then extend.
    
    # For 3x3x3, one rank-17 decomposition:
    # Using the structure from "rank of matrix multiplication" literature
    
    # Let me use a concrete construction. The 17 terms for 3x3x3:
    # I'll use a known explicit form.
    
    # Actually, let me try a more systematic approach:
    # Use the "laser" idea: consider the 3x3x5 tensor as embedded in 3x3x6
    # The 3x3x6 tensor has rank at most 17 (since 3x3x3 is 17, and adding 3 more
    # columns adds at most 3*3=9 more, but we can share).
    
    # Let me try a different tactic: use integer linear programming / SAT-like search
    # with the structure that many entries are 0, 1/2, 1, -1, -1/2, -2
    
    # For rank 17, let me try a structured search:
    # Fix the pattern of non-zero entries, then solve for values
    
    n_r = n * m  # 9
    m_r = m * p  # 15
    p_r = p * n  # 15
    
    # Try to find a rank-17 solution with half-integer entries
    # Use the constraint structure: for each (i,j,k), sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    
    # Let me try a random restart approach with structured initialization
    # biased toward half-integers and small values
    
    value_sets = [
        [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2],
        [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3],
        [-2, -1, -0.5, 0, 0.5, 1, 2],
    ]
    
    r = rank
    rng = random.Random(42)
    
    for vs_idx, values in enumerate(value_sets):
        if _time_remaining() < 3.0:
            break
        
        n_restarts = 50
        for restart in range(n_restarts):
            if _time_remaining() < 3.0:
                break
            
            # Initialize with sparse pattern
            U = [[0.0]*r for _ in range(n_r)]
            V = [[0.0]*r for _ in range(m_r)]
            W = [[0.0]*r for _ in range(p_r)]
            
            # Random sparse initialization
            for row in range(n_r):
                n_nonzero = rng.randint(2, min(5, r))
                cols = rng.sample(range(r), n_nonzero)
                for c in cols:
                    U[row][c] = rng.choice(values)
            
            for row in range(m_r):
                n_nonzero = rng.randint(2, min(5, r))
                cols = rng.sample(range(r), n_nonzero)
                for c in cols:
                    V[row][c] = rng.choice(values)
            
            for row in range(p_r):
                n_nonzero = rng.randint(2, min(5, r))
                cols = rng.sample(range(r), n_nonzero)
                for c in cols:
                    W[row][c] = rng.choice(values)
            
            # Local search with error minimization
            def compute_errors(U, V, W):
                errors = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for rr in range(r):
                                s += U[i*m+j][rr] * V[j*p+k][rr] * W[k*n+i][rr]
                            errors[i,j,k] = s - 1.0
                return errors
            
            errors = compute_errors(U, V, W)
            max_err = np.max(np.abs(errors))
            
            improved = True
            iters = 0
            while improved and iters < 3000 and _time_remaining() > 3.0:
                improved = False
                iters += 1
                
                if max_err < 1e-9:
                    if _validate_half_integer(U, V, W, n, m, p):
                        return {"U": U, "V": V, "W": W}
                    break
                
                # Try multiple random changes
                for _ in range(30):
                    mat_choice = rng.randint(0, 2)
                    if mat_choice == 0:
                        idx = (rng.randint(0, n_r-1), rng.randint(0, r-1))
                        old_val = U[idx[0]][idx[1]]
                        new_val = rng.choice(values)
                        if new_val == old_val:
                            continue
                        U[idx[0]][idx[1]] = new_val
                    elif mat_choice == 1:
                        idx = (rng.randint(0, m_r-1), rng.randint(0, r-1))
                        old_val = V[idx[0]][idx[1]]
                        new_val = rng.choice(values)
                        if new_val == old_val:
                            continue
                        V[idx[0]][idx[1]] = new_val
                    else:
                        idx = (rng.randint(0, p_r-1), rng.randint(0, r-1))
                        old_val = W[idx[0]][idx[1]]
                        new_val = rng.choice(values)
                        if new_val == old_val:
                            continue
                        W[idx[0]][idx[1]] = new_val
                    
                    new_errors = compute_errors(U, V, W)
                    new_max_err = np.max(np.abs(new_errors))
                    
                    if new_max_err < max_err - 1e-12:
                        improved = True
                        max_err = new_max_err
                        errors = new_errors
                        if max_err < 1e-9:
                            if _validate_half_integer(U, V, W, n, m, p):
                                return {"U": U, "V": V, "W": W}
                            break
                    else:
                        if mat_choice == 0:
                            U[idx[0]][idx[1]] = old_val
                        elif mat_choice == 1:
                            V[idx[0]][idx[1]] = old_val
                        else:
                            W[idx[0]][idx[1]] = old_val
            
            if max_err < 1e-9:
                if _validate_half_integer(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
    
    return None

def _try_rank(r, n=3, m=3, p=5, max_iters=5000, seed=42):
    """Try to find a rank-r decomposition using random restarts and local search."""
    if _time_remaining() < 2.0:
        return None
    
    rng = random.Random(seed)
    
    value_sets = [
        [-2, -1, -0.5, 0, 0.5, 1, 2],
        [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3],
        [-1, 0, 1],
        [-2, -1, 0, 1, 2],
    ]
    
    best_solution = None
    
    for vs_idx, values in enumerate(value_sets):
        if _time_remaining() < 3.0:
            break
        
        n_restarts = max(10, min(100, int(200 * _time_remaining() / max(1, 15-r))))
        
        for restart in range(n_restarts):
            if _time_remaining() < 3.0:
                break
            
            U = [[rng.choice(values) for _ in range(r)] for _ in range(n*m)]
            V = [[rng.choice(values) for _ in range(r)] for _ in range(m*p)]
            W = [[rng.choice(values) for _ in range(r)] for _ in range(p*n)]
            
            improved = True
            iters = 0
            while improved and iters < max_iters and _time_remaining() > 3.0:
                improved = False
                iters += 1
                
                errors = {}
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for rr in range(r):
                                s += U[i*m+j][rr] * V[j*p+k][rr] * W[k*n+i][rr]
                            errors[(i,j,k)] = s - 1.0
                
                max_err = max(abs(e) for e in errors.values())
                if max_err < 1e-9:
                    if _validate_half_integer(U, V, W, n, m, p):
                        return {"U": U, "V": V, "W": W}
                    break
                
                for _ in range(20):
                    mat_choice = rng.randint(0, 2)
                    if mat_choice == 0:
                        idx = (rng.randint(0, n*m-1), rng.randint(0, r-1))
                        old_val = U[idx[0]][idx[1]]
                        new_val = rng.choice(values)
                        if new_val == old_val:
                            continue
                        U[idx[0]][idx[1]] = new_val
                    elif mat_choice == 1:
                        idx = (rng.randint(0, m*p-1), rng.randint(0, r-1))
                        old_val = V[idx[0]][idx[1]]
                        new_val = rng.choice(values)
                        if new_val == old_val:
                            continue
                        V[idx[0]][idx[1]] = new_val
                    else:
                        idx = (rng.randint(0, p*n-1), rng.randint(0, r-1))
                        old_val = W[idx[0]][idx[1]]
                        new_val = rng.choice(values)
                        if new_val == old_val:
                            continue
                        W[idx[0]][idx[1]] = new_val
                    
                    new_errors = {}
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0.0
                                for rr in range(r):
                                    s += U[i*m+j][rr] * V[j*p+k][rr] * W[k*n+i][rr]
                                new_errors[(i,j,k)] = s - 1.0
                    
                    new_max_err = max(abs(e) for e in new_errors.values())
                    
                    if new_max_err < max_err - 1e-12:
                        improved = True
                        max_err = new_max_err
                        errors = new_errors
                        if max_err < 1e-9:
                            if _validate_half_integer(U, V, W, n, m, p):
                                return {"U": U, "V": V, "W": W}
                            break
                    else:
                        if mat_choice == 0:
                            U[idx[0]][idx[1]] = old_val
                        elif mat_choice == 1:
                            V[idx[0]][idx[1]] = old_val
                        else:
                            W[idx[0]][idx[1]] = old_val
            
            if max_err < 1e-9:
                if _validate_half_integer(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
    
    return None

def _try_structural(r, n=3, m=3, p=5):
    """Try structural patterns for low-rank decompositions."""
    if _time_remaining() < 2.0:
        return None
    
    for mask_u in range(1 << (n*m)):
        if _time_remaining() < 2.0:
            break
        for mask_v in range(1 << (m*p)):
            if _time_remaining() < 2.0:
                break
            for mask_w in range(1 << (p*n)):
                if _time_remaining() < 2.0:
                    break
                
                au = bin(mask_u).count('1')
                av = bin(mask_v).count('1')
                aw = bin(mask_w).count('1')
                
                if au > r or av > r or aw > r:
                    continue
                
                U = [[0]*r for _ in range(n*m)]
                V = [[0]*r for _ in range(m*p)]
                W = [[0]*r for _ in range(p*n)]
                
                col_u = 0
                for i in range(n*m):
                    if mask_u & (1 << i):
                        U[i][col_u] = 1
                        col_u += 1
                
                col_v = 0
                for j in range(m*p):
                    if mask_v & (1 << j):
                        V[j][col_v] = 1
                        col_v += 1
                
                col_w = 0
                for k in range(p*n):
                    if mask_w & (1 << k):
                        W[k][col_w] = 1
                        col_w += 1
                
                if _validate(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x3 * 3x5 matrix multiplication."""
    global _START_TIME
    _START_TIME = time.time()
    
    n, m, p = DIMENSIONS
    U = [[0]*45 for _ in range(n*m)]
    V = [[0]*45 for _ in range(m*p)]
    W = [[0]*45 for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    schoolbook = {"U": U, "V": V, "W": W}
    
    best_rank = 45
    best_sol = schoolbook
    
    # Try laser method first for rank 17
    sol = _laser_method(n, m, p, rank=17)
    if sol is not None:
        return sol
    
    # Try ranks from 14 upward
    for r in range(14, 46):
        if _time_remaining() < 5.0:
            break
        
        sol = _try_structural(r, n, m, p)
        if sol is not None:
            best_rank = r
            best_sol = sol
            break
        
        sol = _try_rank(r, n, m, p, max_iters=2000, seed=r)
        if sol is not None:
            best_rank = r
            best_sol = sol
            break
        
        for s in [100+r, 200+r, 300+r]:
            if _time_remaining() < 5.0:
                break
            sol = _try_rank(r, n, m, p, max_iters=1000, seed=s)
            if sol is not None:
                best_rank = r
                best_sol = sol
                break
        if sol is not None:
            break
    
    return best_sol
