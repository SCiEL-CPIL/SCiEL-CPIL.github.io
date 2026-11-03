"""Matrix multiplication 3x3x5 rank search with strict validation and deadline guard."""
import time
import random
import math
import sys

DIMENSIONS = [3, 3, 5]
_START = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _validate_strict(U, V, W):
    """
    Strictly check that U, V, W form an exact decomposition.
    U: (n*m, r), V: (m*p, r), W: (p*n, r).
    Identity: 
    For all a in [0, nm), b in [0, mp), c in [0, pn):
        sum_r U[a, r] * V[b, r] * W[c, r] == 1 if (a,b,c) is a valid multiplication index triplet
        sum_r U[a, r] * V[b, r] * W[c, r] == 0 otherwise
    
    Valid triplet: 
    a = i*m + j
    b = j*p + k
    c = k*n + i
    for some i in [0,n), j in [0,m), k in [0,p).
    """
    n, m, p = DIMENSIONS
    r = len(U[0]) if U else 0
    if r == 0:
        return False
        
    # Precompute the set of valid indices
    valid_indices = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_indices.add((a, b, c))
                
    # Check all possible (a, b, c)
    # Total entries: (n*m) * (m*p) * (p*n) = 9 * 15 * 15 = 2025
    # This is fast enough to do in pure Python.
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                s = 0
                ua = U[a]
                vb = V[b]
                wc = W[c]
                for idx in range(r):
                    s += ua[idx] * vb[idx] * wc[idx]
                
                if (a, b, c) in valid_indices:
                    if s != 1:
                        return False
                else:
                    if s != 0:
                        return False
    return True

def _anneal_strict(r, n_restarts, max_iter_per_restart, seed):
    """
    Simulated annealing with strict validation.
    """
    n, m, p = DIMENSIONS
    rng = random.Random(seed)
    
    # Objective function: Sum of squared errors over ALL entries
    # We want sum_r U[a,r]V[b,r]W[c,r] to be 1 for valid, 0 for invalid.
    # Let target[a,b,c] be 1 or 0.
    # Error = sum_{a,b,c} (sum_r ... - target)^2
    
    # Precompute targets
    targets = {}
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                # Check if valid
                i, j = divmod(a, m)
                j2, k = divmod(b, p)
                k2, i2 = divmod(c, n)
                if j == j2 and k == k2 and i == i2:
                    targets[(a,b,c)] = 1.0
                else:
                    targets[(a,b,c)] = 0.0
                    
    def compute_error(U, V, W):
        err = 0.0
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    s = 0.0
                    ua = U[a]
                    vb = V[b]
                    wc = W[c]
                    for idx in range(r):
                        s += ua[idx] * vb[idx] * wc[idx]
                    diff = s - targets[(a,b,c)]
                    err += diff * diff
        return err

    best_res = None
    best_val = float('inf')

    for restart in range(n_restarts):
        if _time_left() < 3.0:
            break
            
        # Initialize
        def rand_val():
            return rng.choice([-1, -0.5, 0, 0, 0.5, 1])
            
        U = [[rand_val() for _ in range(r)] for _ in range(n * m)]
        V = [[rand_val() for _ in range(r)] for _ in range(m * p)]
        W = [[rand_val() for _ in range(r)] for _ in range(p * n)]
        
        val = compute_error(U, V, W)
        if val < best_val:
            best_val = val
            best_res = (U, V, W)
            if best_val < 1e-9:
                if _validate_strict(U, V, W):
                    return U, V, W

        T = 10.0
        T_min = 1e-8
        cooling = 0.99
        
        for it in range(max_iter_per_restart):
            if _time_left() < 3.0:
                break
                
            mat = rng.choice([0, 1, 2])
            if mat == 0:
                row = rng.randrange(n * m)
                arr = U
            elif mat == 1:
                row = rng.randrange(m * p)
                arr = V
            else:
                row = rng.randrange(p * n)
                arr = W
            col = rng.randrange(r)
            old = arr[row][col]
            delta = rng.choice([-1, -0.5, 0.5, 1])
            new = old + delta
            arr[row][col] = new
            
            new_val = compute_error(U, V, W)
            d = new_val - val
            if d < 0 or rng.random() < math.exp(-d / T):
                val = new_val
                if val < best_val:
                    best_val = val
                    best_res = (U, V, W)
                    if best_val < 1e-9:
                        if _validate_strict(U, V, W):
                            return U, V, W
            else:
                arr[row][col] = old
                
            T *= cooling
            if T < T_min:
                T = T_min

    # If we found a low error, verify it strictly
    if best_res is not None and best_val < 1e-6:
        U, V, W = best_res
        if _validate_strict(U, V, W):
            return U, V, W
            
    return None

def construct():
    """
    Returns {'U', 'V', 'W'} with the best (lowest-rank) valid decomposition found.
    """
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 45
    
    # Schoolbook Fallback
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
                
    best = (schoolbook_rank, U_sb, V_sb, W_sb)
    
    # Search for lower ranks
    # Start from lower bound max(n*m, m*p, p*n) = 15
    # Try up to schoolbook
    rng = random.Random(42)
    
    # We will try a few ranks. If we find a valid one, we return it.
    # To save time, we focus on the most likely candidates if any.
    # But since we don't know the exact minimum, we scan upwards.
    
    # Optimization: Only try a few ranks with limited iterations to stay within time.
    for r in range(15, 45):
        if _time_left() < 5.0:
            break
            
        # Reduce iterations as we go higher if time is tight
        time_per_rank = _time_left() / (45 - r)
        max_iter = int(time_per_rank * 100) # rough estimate
        max_iter = max(100, min(2000, max_iter))
        
        result = _anneal_strict(r, n_restarts=3, max_iter_per_restart=max_iter, seed=rng.randint(0, 10**9))
        if result is not None:
            U, V, W = result
            # Double check
            if _validate_strict(U, V, W):
                best = (r, U, V, W)
                break # Found a valid one at this rank, it's the lowest we tried. 
                       # Since we scan from low to high, this is the best found so far.
                       # We could continue to see if we find lower? No, we scanned from 15 up.
                       # So the first valid one is the lowest rank found.
                       # But wait, we might have skipped a valid one at 15?
                       # The search is stochastic. If it fails at 15, it might find at 16.
                       # So we break at the first SUCCESS.
    
    r, U, V, W = best
    return {"U": U, "V": V, "W": W}
