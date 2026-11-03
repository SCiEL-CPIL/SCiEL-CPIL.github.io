import numpy as np
import time
import random
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    # 1. Construct Schoolbook Decomposition (Rank n*m*p = 48)
    rank_sb = n * m * p
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_result = (U_sb, V_sb, W_sb)
    best_rank = rank_sb

    # 2. Search strategy:
    # Prioritize known likely minimal ranks (20-25) with high restarts.
    # Then try other ranks if time permits.
    
    # Define search order:
    # Try 20, 21, 22, 23, 24, 25 first.
    # Then 26-47 if time remains.
    
    search_ranks = list(range(20, 26)) + list(range(26, 48))
    
    for rank in search_ranks:
        if time.time() > deadline - 10:
            break
        
        # Allocate time based on rank priority
        # Lower ranks are harder, give them more time/restarts?
        # Or just fixed time per rank.
        
        # If we found a better rank, we stop?
        # The task asks for "rank-minimal".
        # So we must find the *minimum* possible.
        # If we find 20, we can stop?
        # Yes, if 20 is the lower bound.
        # But we don't know the lower bound for sure.
        # However, if we find 20, it's very likely minimal.
        # Let's assume if we find rank R, we try R-1, etc.
        
        # Actually, the current code breaks on first found.
        # This is dangerous if we find 25 but 20 exists.
        # We should iterate downwards and keep the best.
        
        # Revised: Iterate from low to high?
        # If we iterate 20, 21, ..., 48.
        # If we find 20, we can return immediately (assuming 19 is impossible or very hard).
        # If we don't find 20, we try 21.
        
        # Let's try ranks in increasing order from 16 to 48.
        # If we find a solution for rank R, we record it.
        # We continue searching for R-1? No, we are going up.
        # We should go down from 48 to 16.
        # If we find 47, we don't stop, we try 46.
        # If we find 20, we don't stop, we try 19...
        # But 19 is likely impossible.
        
        # To be safe:
        # Search 47 down to 16.
        # Keep the best (lowest) rank found.
        
        # Optimization:
        # If we find a rank R, we can stop if R is close to theoretical lower bound.
        # Lower bound for 3x4x4 is likely 20.
        
        # Let's stick to the current "find first valid in descending order" but 
        # ensure we don't waste time on high ranks if we can find low ones faster.
        
        # Actually, the current code:
        # for rank in range(47, 15, -1):
        #   found = _try_find_rank(...)
        #   if found:
        #      best = found
        #      break
        #
        # This stops at the FIRST (highest) rank found.
        # This is WRONG for "rank-minimal".
        # It should find the LOWEST rank.
        # So it should search 47, 46, ..., 16.
        # If it finds 47, it should NOT break. It should continue to 46.
        # Only break if we reach the lower bound or time out.
        
        # BUT, finding 47 is easy. Finding 20 is hard.
        # If we spend 9s on 47, we have less time for 20.
        
        # Better:
        # Try a few low ranks (20-25) first.
        # If found, great.
        # If not, try high ranks (40-47) to get a valid solution.
        # Then try middle ranks.
        
        pass 

    # The current code logic is flawed for "minimal" if it breaks on first found.
    # It finds the *highest* rank that is < 48.
    # We need to fix the search loop to find the *lowest*.
    
    # Let's rewrite the loop.
    
    best_rank_found = 48
    best_res = (U_sb, V_sb, W_sb)
    
    # Try ranks 20 to 25 first (high probability of being minimal)
    for rank in range(20, 26):
        if time.time() > deadline - 10:
            break
        found = _try_find_rank(n, m, p, rank, max_restarts=10, time_limit=5, deadline=deadline)
        if found is not None:
            best_res = found
            best_rank_found = rank
            # If we find 20, we can probably stop?
            # Let's assume 20 is the min.
            if rank == 20:
                break
            # If we find 21, keep searching for 20?
            # We are iterating 20, 21, 22...
            # So if we find 21, we have already tried 20 and failed.
            # So we can break?
            # No, we tried 20 with limited time.
            # Maybe 20 is possible but we didn't find it.
            # Let's just record the best.
    
    # If we didn't find anything low, try high ranks to ensure valid output
    if best_rank_found == 48:
        for rank in range(47, 25, -1):
            if time.time() > deadline - 10:
                break
            found = _try_find_rank(n, m, p, rank, max_restarts=5, time_limit=3, deadline=deadline)
            if found is not None:
                best_res = found
                best_rank_found = rank
                break # High ranks are easy, take the first one found
    
    U, V, W = best_res
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _try_find_rank(n, m, p, rank, max_restarts=10, time_limit=10, deadline=None):
    start_time = time.time()

    for restart in range(max_restarts):
        if time.time() > start_time + time_limit:
            break
        if deadline is not None and time.time() > deadline - 5:
            break

        # Strategy A: prune from schoolbook
        # Strategy B: structured random init
        if restart % 2 == 0:
            U, V, W = _init_prune(n, m, p, rank)
        else:
            U, V, W = _init_random(n, m, p, rank)

        # ALS iterations
        U, V, W, max_err = _als(U, V, W, n, m, p,
                                 iters=40,
                                 start_time=start_time,
                                 time_limit=time_limit)

        # Check
        if max_err < 1e-6:
            # Polish with SciPy LM
            U2, V2, W2 = _polish_lm(U, V, W, n, m, p,
                                  time_left=time_limit - (time.time() - start_time))
            if U2 is not None:
                return U2, V2, W2
            else:
                # Try direct snap
                snapped, ok = _snap_all(U, V, W, 1e-4)
                if ok and _check_exact(snapped[0], snapped[1], snapped[2], n, m, p):
                    return snapped

    return None

def _init_prune(n, m, p, rank):
    """Start from schoolbook, keep 'rank' random terms, re-solve."""
    nmp = n * m * p
    U, V, W = _schoolbook(n, m, p)

    # Choose 'rank' random columns to keep
    cols = random.sample(range(nmp), rank)
    U = U[:, cols]
    V = V[:, cols]
    W = W[:, cols]

    # Add small perturbation to allow re-solvability
    U += np.random.randn(n * m, rank) * 0.05
    V += np.random.randn(m * p, rank) * 0.05
    W += np.random.randn(p * n, rank) * 0.05
    return U, V, W

def _init_random(n, m, p, rank):
    """Structured random init with some schoolbook structure."""
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if random.random() < 0.25:
                        U[i * m + j, r] = random.choice([-1.0, 1.0])
                        V[j * p + k, r] = random.choice([-1.0, 1.0])
                        W[k * n + i, r] = random.choice([-1.0, 1.0])

    # Add noise to break symmetry
    U += np.random.randn(n * m, rank) * 0.2
    V += np.random.randn(m * p, rank) * 0.2
    W += np.random.randn(p * n, rank) * 0.2
    return U, V, W

def _als(U, V, W, n, m, p, iters=60, start_time=None, time_limit=10):
    """Alternating least squares for tensor decomposition."""
    max_err = float('inf')
    prev_err = float('inf')

    for it in range(iters):
        if start_time is not None and time.time() > start_time + time_limit:
            break

        # Update U
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                idx = i * m + j
                V_part = V[j * p:(j + 1) * p, :]
                W_part = W[np.arange(p) * n + i, :]
                A = V_part * W_part
                b = np.ones(p)
                try:
                    x = np.linalg.lstsq(A, b, rcond=None)[0]
                    U_new[idx, :] = x
                except Exception:
                    U_new[idx, :] = U[idx, :]
        U = U_new

        # Update V
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                idx = j * p + k
                U_part = U[np.arange(n) * m + j, :]
                W_part = W[k * n:(k + 1) * n, :]
                A = U_part * W_part
                b = np.ones(n)
                try:
                    x = np.linalg.lstsq(A, b, rcond=None)[0]
                    V_new[idx, :] = x
                except Exception:
                    V_new[idx, :] = V[idx, :]
        V = V_new

        # Update W
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                idx = k * n + i
                U_part = U[i * m:(i + 1) * m, :]
                V_part = V[np.arange(m) * p + k, :]
                A = U_part * V_part
                b = np.ones(m)
                try:
                    x = np.linalg.lstsq(A, b, rcond=None)[0]
                    W_new[idx, :] = x
                except Exception:
                    W_new[idx, :] = W[idx, :]
        W = W_new

        # Compute max error
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    e = abs(val - 1.0)
                    if e > max_err:
                        max_err = e

        if max_err < 1e-8:
            break
        if abs(prev_err - max_err) < 1e-11:
            break
        prev_err = max_err

    return U, V, W, max_err

def _polish_lm(U, V, W, n, m, p, time_left=5):
    """Polish using Levenberg-Marquardt."""
    if time_left < 0.5:
        return None, None, None

    start = time.time()
    rank = U.shape[1]
    
    # Flatten parameters
    params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def residuals(params):
        U_p = params[:n*m*rank].reshape(n*m, rank)
        V_p = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W_p = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        res = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_p[i*m+j, :], V_p[j*p+k, :] * W_p[k*n+i, :])
                    res.append(1.0 - val)
        return np.array(res)

    try:
        result = least_squares(residuals, params, method='lm', max_nfev=100)
        if result.success or result.cost < 1e-8:
            U_new = result.x[:n*m*rank].reshape(n*m, rank)
            V_new = result.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W_new = result.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
            
            # Check and snap
            snapped, ok = _snap_all(U_new, V_new, W_new, 1e-3)
            if ok and _check_exact(snapped[0], snapped[1], snapped[2], n, m, p):
                return snapped
            else:
                return None, None, None
    except Exception:
        return None, None, None

def _snap_all(U, V, W, tol=1e-6):
    """Snap all three to half-integers if close."""
    U_s = np.round(U * 2) / 2
    V_s = np.round(V * 2) / 2
    W_s = np.round(W * 2) / 2
    ok = (np.max(np.abs(U - U_s)) < tol and
          np.max(np.abs(V - V_s)) < tol and
          np.max(np.abs(W - W_s)) < tol)
    return (U_s, V_s, W_s), ok

def _check_exact(U, V, W, n, m, p, tol=1e-9):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True
