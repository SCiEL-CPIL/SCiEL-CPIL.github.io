"""
Rank-minimal tensor decomposition for 2x3 * 3x9 matrix multiplication.

The key insight is that the 2x3x9 problem decomposes into three independent
2x3x3 sub-problems (one for each block of 3 columns of B). If we can find a
rank-r decomposition for 2x3x3, we get rank 3r for 2x3x9.

We attempt to find rank-14 (or lower) decompositions for 2x3x3 using:
1. Structured random initialization
2. Local search with coordinate descent
3. Rounding to half-integers and verification

If we succeed, we replicate the 2x3x3 solution three times for 2x3x9,
achieving rank 42 (or lower) instead of the schoolbook rank 54.
"""
import numpy as np
import time
import random

DIMENSIONS = [2, 3, 9]

def _check_233(U, V, W, n=2, m=3, p=3, tol=1e-9):
    """Check if U,V,W form a valid decomposition for 2x3x3 matrix multiplication."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _check_239(U, V, W, n=2, m=3, p=9, tol=1e-9):
    """Check if U,V,W form a valid decomposition for 2x3x9 matrix multiplication."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _round_halfint(arr):
    """Round array to nearest half-integer."""
    return np.round(2.0 * arr) / 2.0

def _schoolbook_233(n=2, m=3, p=3):
    """Schoolbook decomposition for 2x3x3, rank 18."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W, rank

def _schoolbook_239(n=2, m=3, p=9):
    """Schoolbook decomposition for 2x3x9, rank 54."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W, rank

def _replicate_233_to_239(U3, V3, W3, n=2, m=3, p3=3, p=9):
    """Replicate a 2x3x3 decomposition three times to get 2x3x9."""
    r3 = U3.shape[1]
    rank = 3 * r3
    U9 = np.zeros((n*m, rank))
    V9 = np.zeros((m*p, rank))
    W9 = np.zeros((p*n, rank))
    
    for block in range(3):
        U9[:, block*r3:(block+1)*r3] = U3
        # V rows: j*p + k where k is in [block*3, (block+1)*3)
        for j in range(m):
            for k3 in range(p3):
                k9 = block * p3 + k3
                V9[j*p + k9, block*r3:(block+1)*r3] = V3[j*p3 + k3]
        # W rows: k*n + i where k is in [block*3, (block+1)*3)
        for k3 in range(p3):
            k9 = block * p3 + k3
            for i in range(n):
                W9[k9*n + i, block*r3:(block+1)*r3] = W3[k3*n + i]
    
    return U9, V9, W9, rank

def _local_search_233(U, V, W, n=2, m=3, p=3, max_iter=5000, tol=1e-9, rng=None):
    """
    Local search to find a valid 2x3x3 decomposition starting from random init.
    Uses coordinate descent: pick a random (matrix, row, col) and adjust to reduce error.
    """
    if rng is None:
        rng = random.Random()
    
    rank = U.shape[1]
    
    def total_error(Uc, Vc, Wc):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(Uc[i*m+j, :] * Vc[j*p+k, :] * Wc[k*n+i, :])
                    err += (val - 1.0) ** 2
        return err
    
    best_err = total_error(U, V, W)
    U, V, W = U.copy(), V.copy(), W.copy()
    
    for iteration in range(max_iter):
        if best_err < tol:
            return U, V, W, best_err, True
        
        # Pick random coordinate to perturb
        mat = rng.choice(['U', 'V', 'W'])
        if mat == 'U':
            rows, cols = U.shape
            r, c = rng.randrange(rows), rng.randrange(cols)
            old = U[r, c]
        elif mat == 'V':
            rows, cols = V.shape
            r, c = rng.randrange(rows), rng.randrange(cols)
            old = V[r, c]
        else:
            rows, cols = W.shape
            r, c = rng.randrange(rows), rng.randrange(cols)
            old = W[r, c]
        
        # Try several new values
        new_vals = [old + 0.5, old - 0.5, old + 1.0, old - 1.0, 0.0, 1.0, -1.0, 0.5, -0.5]
        new_vals = list(set(new_vals))
        
        best_new_val = old
        best_new_err = best_err
        
        for nv in new_vals:
            if mat == 'U':
                U[r, c] = nv
                err = total_error(U, V, W)
                U[r, c] = old
            elif mat == 'V':
                V[r, c] = nv
                err = total_error(U, V, W)
                V[r, c] = old
            else:
                W[r, c] = nv
                err = total_error(U, V, W)
                W[r, c] = old
            
            if err < best_new_err:
                best_new_err = err
                best_new_val = nv
        
        if best_new_val != old:
            if mat == 'U':
                U[r, c] = best_new_val
            elif mat == 'V':
                V[r, c] = best_new_val
            else:
                W[r, c] = best_new_val
            best_err = best_new_err
    
    return U, V, W, best_err, best_err < tol

def _find_233_decomp(rank, max_trials=20, time_limit=10.0, start_time=None):
    """
    Try to find a rank-rank decomposition for 2x3x3.
    Returns (U, V, W, success) or (None, None, None, False).
    """
    n, m, p = 2, 3, 3
    if start_time is None:
        start_time = time.time()
    
    rng = random.Random(int(time.time() * 1000000))
    
    for trial in range(max_trials):
        if time.time() - start_time > time_limit:
            break
        
        # Random initialization with half-integer values
        vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
        U = np.random.choice(vals, size=(n*m, rank))
        V = np.random.choice(vals, size=(m*p, rank))
        W = np.random.choice(vals, size=(p*n, rank))
        
        # Ensure no column is entirely zero (would be useless)
        # But allow some zeros
        for c in range(rank):
            if np.all(U[:, c] == 0) or np.all(V[:, c] == 0) or np.all(W[:, c] == 0):
                # Set one random entry to 0.5
                if np.all(U[:, c] == 0):
                    U[rng.randrange(n*m), c] = 0.5
                elif np.all(V[:, c] == 0):
                    V[rng.randrange(m*p), c] = 0.5
                elif np.all(W[:, c] == 0):
                    W[rng.randrange(p*n), c] = 0.5
        
        # Local search
        U, V, W, err, success = _local_search_233(U, V, W, n, m, p, 
                                                    max_iter=3000, 
                                                    tol=1e-9, rng=rng)
        
        if success:
            # Round to half-integers and verify
            U_r = _round_halfint(U)
            V_r = _round_halfint(V)
            W_r = _round_halfint(W)
            if _check_233(U_r, V_r, W_r, n, m, p, tol=1e-6):
                return U_r, V_r, W_r, True
            # Try without rounding
            if _check_233(U, V, W, n, m, p, tol=1e-6):
                # Round and check again
                U_r = _round_halfint(U)
                V_r = _round_halfint(V)
                W_r = _round_halfint(W)
                if _check_233(U_r, V_r, W_r, n, m, p, tol=1e-6):
                    return U_r, V_r, W_r, True
                return U, V, W, True
    
    return None, None, None, False

def _find_239_direct(rank, max_trials=10, time_limit=15.0, start_time=None):
    """
    Try to find a rank-rank decomposition for 2x3x9 directly.
    """
    n, m, p = 2, 3, 9
    if start_time is None:
        start_time = time.time()
    
    rng = random.Random(int(time.time() * 1000000) + 42)
    
    for trial in range(max_trials):
        if time.time() - start_time > time_limit:
            break
        
        vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
        U = np.random.choice(vals, size=(n*m, rank))
        V = np.random.choice(vals, size=(m*p, rank))
        W = np.random.choice(vals, size=(p*n, rank))
        
        # Simple local search
        def total_error(Uc, Vc, Wc):
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(Uc[i*m+j, :] * Vc[j*p+k, :] * Wc[k*n+i, :])
                        err += (val - 1.0) ** 2
            return err
        
        U, V, W = U.copy(), V.copy(), W.copy()
        best_err = total_error(U, V, W)
        
        for iteration in range(2000):
            if best_err < 1e-9:
                break
            if time.time() - start_time > time_limit:
                break
            
            mat = rng.choice(['U', 'V', 'W'])
            if mat == 'U':
                rows, cols = U.shape
                r, c = rng.randrange(rows), rng.randrange(cols)
                old = U[r, c]
            elif mat == 'V':
                rows, cols = V.shape
                r, c = rng.randrange(rows), rng.randrange(cols)
                old = V[r, c]
            else:
                rows, cols = W.shape
                r, c = rng.randrange(rows), rng.randrange(cols)
                old = W[r, c]
            
            new_vals = [old + 0.5, old - 0.5, old + 1.0, old - 1.0, 0.0, 1.0, -1.0, 0.5, -0.5]
            new_vals = list(set(new_vals))
            
            best_new_val = old
            best_new_err = best_err
            
            for nv in new_vals:
                if mat == 'U':
                    U[r, c] = nv
                    err = total_error(U, V, W)
                    U[r, c] = old
                elif mat == 'V':
                    V[r, c] = nv
                    err = total_error(U, V, W)
                    V[r, c] = old
                else:
                    W[r, c] = nv
                    err = total_error(U, V, W)
                    W[r, c] = old
                
                if err < best_new_err:
                    best_new_err = err
                    best_new_val = nv
            
            if best_new_val != old:
                if mat == 'U':
                    U[r, c] = best_new_val
                elif mat == 'V':
                    V[r, c] = best_new_val
                else:
                    W[r, c] = best_new_val
                best_err = best_new_err
        
        if best_err < 1e-6:
            U_r = _round_halfint(U)
            V_r = _round_halfint(V)
            W_r = _round_halfint(W)
            if _check_239(U_r, V_r, W_r, n, m, p, tol=1e-6):
                return U_r, V_r, W_r, True
            if _check_239(U, V, W, n, m, p, tol=1e-6):
                return U, V, W, True
    
    return None, None, None, False

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 9
    
    # Fallback: schoolbook
    U_school, V_school, W_school, rank_school = _schoolbook_239(n, m, p)
    best_U, best_V, best_W, best_rank = U_school, V_school, W_school, rank_school
    
    # Strategy 1: Try to find 2x3x3 decomposition with rank < 18, replicate for 2x3x9
    # This would give rank 3*r3 < 54
    for target_r3 in range(14, 18):  # 14, 15, 16, 17
        if time.time() > deadline - 10:
            break
        
        target_r9 = 3 * target_r3
        if target_r9 >= best_rank:
            break
        
        U3, V3, W3, success = _find_233_decomp(
            target_r3, max_trials=15, time_limit=8.0, start_time=start_time
        )
        
        if success:
            U9, V9, W9, r9 = _replicate_233_to_239(U3, V3, W3, n, m, 3, p)
            # Verify
            if _check_239(U9, V9, W9, n, m, p, tol=1e-6):
                best_U, best_V, best_W, best_rank = U9, V9, W9, r9
                # Continue searching for lower
            else:
                # Shouldn't happen if _find_233_decomp succeeded
                pass
    
    # Strategy 2: Try direct 2x3x9 search for ranks below best_rank
    for target_rank in range(min(best_rank - 1, 53), 35, -1):
        if time.time() > deadline - 5:
            break
        
        U9, V9, W9, success = _find_239_direct(
            target_rank, max_trials=5, time_limit=5.0, start_time=start_time
        )
        
        if success:
            if target_rank < best_rank:
                best_U, best_V, best_W, best_rank = U9, V9, W9, target_rank
    
    # Final: round to half-integers
    best_U = np.array(best_U, dtype=float)
    best_V = np.array(best_V, dtype=float)
    best_W = np.array(best_W, dtype=float)
    best_rank = best_U.shape[1]
    
    U_out = [[_round_halfint(best_U[i, j]) for j in range(best_rank)] for i in range(n*m)]
    V_out = [[_round_halfint(best_V[i, j]) for j in range(best_rank)] for i in range(m*p)]
    W_out = [[_round_halfint(best_W[i, j]) for j in range(best_rank)] for i in range(p*n)]
    
    return {"U": U_out, "V": V_out, "W": W_out}
