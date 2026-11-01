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

def _try_rank(r, n=3, m=3, p=5, max_iters=5000, seed=42):
    """Try to find a rank-r decomposition using random restarts and local search."""
    if _time_remaining() < 2.0:
        return None
    
    rng = random.Random(seed)
    
    # Try different value sets
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
            
            # Random initialization with values from the set
            U = [[rng.choice(values) for _ in range(r)] for _ in range(n*m)]
            V = [[rng.choice(values) for _ in range(r)] for _ in range(m*p)]
            W = [[rng.choice(values) for _ in range(r)] for _ in range(p*n)]
            
            # Local search: try flipping individual entries
            improved = True
            iters = 0
            while improved and iters < max_iters and _time_remaining() > 3.0:
                improved = False
                iters += 1
                
                # Compute current error
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
                
                # Try random entry changes
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
                    
                    # Check if error decreased
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
                        # Revert
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
    
    # Pattern: some entries of U, V, W are fixed to 0 or 1
    # Try sparse patterns
    
    for mask_u in range(1 << (n*m)):
        if _time_remaining() < 2.0:
            break
        for mask_v in range(1 << (m*p)):
            if _time_remaining() < 2.0:
                break
            for mask_w in range(1 << (p*n)):
                if _time_remaining() < 2.0:
                    break
                
                # Count active entries
                au = bin(mask_u).count('1')
                av = bin(mask_v).count('1')
                aw = bin(mask_w).count('1')
                
                if au > r or av > r or aw > r:
                    continue
                
                # Build sparse U, V, W
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
    schoolbook = {"U": [[0]*45 for _ in range(9)], "V": [[0]*45 for _ in range(15)], "W": [[0]*45 for _ in range(15)]}
    # Build schoolbook as fallback
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
    
    # Try ranks from 14 upward
    for r in range(14, 46):
        if _time_remaining() < 5.0:
            break
        
        # First try structural patterns
        sol = _try_structural(r, n, m, p)
        if sol is not None:
            best_rank = r
            best_sol = sol
            break
        
        # Then try random/local search
        sol = _try_rank(r, n, m, p, max_iters=2000, seed=r)
        if sol is not None:
            best_rank = r
            best_sol = sol
            break
        
        # Try with different seeds
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
