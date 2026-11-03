import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast vectorized verification using einsum."""
    try:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        if U_arr.shape[1] != V_arr.shape[1] or U_arr.shape[1] != W_arr.shape[1]:
            return False, float('inf')
            
        rank = U_arr.shape[1]
        
        U_3d = U_arr.reshape(n, m, rank)
        V_3d = V_arr.reshape(m, p, rank)
        W_3d = W_arr.reshape(p, n, rank)
        
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        
        max_err = np.max(np.abs(T - 1.0))
        return max_err < tol, max_err
    except Exception:
        return False, float('inf')

def _verify_exact(U, V, W, n, m, p):
    """Verify using exact Fraction arithmetic."""
    try:
        U_frac = [[Fraction(x) for x in row] for row in U]
        V_frac = [[Fraction(x) for x in row] for row in V]
        W_frac = [[Fraction(x) for x in row] for row in W]
        
        rank = len(U[0])
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = Fraction(0)
                    for c in range(rank):
                        val += U_frac[i*m+j][c] * V_frac[j*p+k][c] * W_frac[k*n+i][c]
                    if val != 1:
                        return False
        return True
    except Exception:
        return False

def _try_polish(U, V, W, n, m, p, max_denom=20):
    """Try to snap to small rationals and verify."""
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    for denom in [1, 2, 3, 4, 5, 10, 20]:
        U_frac = [[Fraction(x).limit_denominator(denom) for x in row] for row in U_list]
        V_frac = [[Fraction(x).limit_denominator(denom) for x in row] for row in V_list]
        W_frac = [[Fraction(x).limit_denominator(denom) for x in row] for row in W_list]
        
        if _verify_exact(U_frac, V_frac, W_frac, n, m, p):
            return U_frac, V_frac, W_frac
    return None

def _random_int_search(n, m, p, r, max_time=5, seed=42):
    """Random search for integer/half-integer rank-r decomposition."""
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    # Values to sample from
    values = np.array([-1, -0.5, 0, 0.5, 1])
    
    best_err = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for iteration in range(1000):
        if time.time() - start_time > max_time:
            break
            
        # Generate sparse random matrices
        # Most entries should be 0
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        # Fill with random values from the set
        # We want to encourage non-zero entries to form valid terms
        # A simple strategy: for each rank index c, pick random non-zero entries
        for c in range(r):
            # Pick a few non-zero entries in each column
            # This is a heuristic to keep it sparse
            idx_U = rng.choice(n * m, size=max(1, n*m//4), replace=False)
            idx_V = rng.choice(m * p, size=max(1, m*p//4), replace=False)
            idx_W = rng.choice(p * n, size=max(1, p*n//4), replace=False)
            
            U[idx_U, c] = values[rng.randint(0, len(values), size=len(idx_U))]
            V[idx_V, c] = values[rng.randint(0, len(values), size=len(idx_V))]
            W[idx_W, c] = values[rng.randint(0, len(values), size=len(idx_W))]
            
        # Verify
        ok, err = _verify_fast(U, V, W, n, m, p, tol=1e-9)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if ok:
            # If we find a valid one, try to polish it
            polished = _try_polish(U, V, W, n, m, p)
            if polished:
                return polished[0], polished[1], polished[2], r
            else:
                # If polish fails but float is valid, it might be close to integer
                # But we need exact integers/half-integers
                # If the float solution is exact (e.g. 1.0, 0.5), polish should work
                # If not, we can't return it
                pass
                
    if best_U is not None and best_err < 1e-9:
        return best_U, best_V, best_W, r
        
    return None

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Start with the known valid schoolbook solution
    result = _schoolbook()
    best_rank = n * m * p
    
    # Attempt to find lower rank solutions
    # The theoretical minimum is max(nm, mp, pn) = 16.
    # We try ranks from 16 up to 47.
    # Given the difficulty, we prioritize lower ranks but limit time per rank.
    
    # We will try a few seeds for a few ranks.
    # If we find a valid one, we return it.
    
    # Ranks to try: 16, 17, ..., 47
    # To be efficient, we might just try a few specific ones or a range.
    # Let's try ranks from 16 to 40 in steps of 1, but with very short time limits.
    
    # Actually, finding rank < 48 is extremely hard. 
    # Let's just try a few seeds for rank 47, 46, etc.
    # If we don't find anything, we return schoolbook.
    
    # Strategy: Try rank 47 with a few seeds. If fail, try 46, etc.
    # But each attempt takes time.
    
    # Let's allocate 10 seconds per rank attempt.
    # We have ~140 seconds. So we can try about 10 ranks.
    # Let's try ranks 47 down to 38.
    
    for r in range(47, 37, -1):
        if time.time() > deadline - 5:
            break
            
        for seed in range(3): # 3 seeds per rank
            if time.time() > deadline - 2:
                break
                
            res = _random_int_search(n, m, p, r, max_time=4, seed=seed)
            if res is not None:
                U, V, W, rank = res
                if rank < best_rank:
                    best_rank = rank
                    polished = _try_polish(U, V, W, n, m, p)
                    if polished is not None:
                        U_f, V_f, W_f = polished
                        result = {"U": U_f, "V": V_f, "W": W_f}
                    else:
                        # If polish fails, we cannot return floats as they must be exact.
                        # However, if the float solution is extremely close to integers, 
                        # it might be valid. But _try_polish handles this.
                        # If _try_polish returns None, we discard this candidate.
                        pass
                    break # Found a valid one for this rank, move to next lower rank? 
                         # No, we want the MINIMAL rank. 
                         # So if we find rank 47, we should continue to try 46.
                         # But we break the seed loop.
    
    return result
