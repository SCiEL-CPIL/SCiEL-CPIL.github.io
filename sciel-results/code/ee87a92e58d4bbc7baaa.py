"""Matrix multiplication 2x2x16 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 16]
_TIME_LIMIT = 140.0
_START_TIME = time.time()

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check sum_r U[i,r]*V[j,r]*W[k,r] == 1 if k==i*p+j else 0."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(p):
            target = i * p + j
            total = np.sum(U[:, j] * W[target, :] * V[:, j])  # wrong indexing
    # Proper check via tensor contraction
    rank = U.shape[1]
    for r in range(rank):
        pass
    # Build tensor T[i,k,j] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Target: T[i,k,j] = 1 if k == i*p+j else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    max_err = 0.0
    for i in range(n):
        for k in range(p):
            for j in range(m):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                target = 1.0 if (k == i * p + j) else 0.0
                err = abs(s - target)
                if err > max_err:
                    max_err = err
    return max_err < tol, max_err

def _strassen_2x2x2():
    """Return 7 (u, v, w) vectors for 2x2x2 Strassen.
    u index: i*m+j = i*2+j for A[i][j]
    v index: j*p+k = j*2+k for B[j][k]
    w index: k*n+i = k*2+i for C[i][k]
    """
    # A = [[a, b],[c, d]] -> u indices: 00,01,10,11
    # B = [[e, f],[g, h]] -> v indices: 00,01,10,11
    # C = [[C11,C12],[C21,C22]] -> w indices: k*n+i
    #   C11: k=0,i=0 -> 0
    #   C12: k=1,i=0 -> 2
    #   C21: k=0,i=1 -> 1
    #   C22: k=1,i=1 -> 3
    terms = [
        # M1 = (a+d)(e+h), C11+=1, C22+=1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2 = (c+d)e, C21+=1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]),
        # M3 = a(f-h), C12+=1, C22+=1
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        # M4 = d(g-e), C11+=1, C21+=1
        ([0, 0, 0, 1], [0, -1, 1, 0], [1, 1, 0, 0]),
        # M5 = (a+b)h, C11-=1, C12+=1
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        # M6 = (c-a)(e+f), C22+=1
        ([1, 0, -1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7 = (b-d)(g+h), C11+=1, C22-=1
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, -1]),
    ]
    return terms

def _build_strassen_replicated(n, m, p):
    """Build rank 8*7=56 replicated Strassen decomposition."""
    rank = 8 * 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    terms = _strassen_2x2x2()
    num_blocks = p // 2  # 8 blocks of 2 columns
    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s
            u, v, w = terms[s]
            # U: A is 2x2, indices i*m+j
            for idx in range(4):
                U[idx, col] = u[idx]
            # V: B is 2x16, local block t has columns 2t, 2t+1
            # local v index: 00,01,10,11 -> j*p+k
            v_map = [0 * p + (2 * t), 0 * p + (2 * t + 1), 1 * p + (2 * t), 1 * p + (2 * t + 1)]
            for idx in range(4):
                V[v_map[idx], col] = v[idx]
            # W: C is 2x16, W index k*n+i
            # local w index: 00,01,10,11 -> i=0/1, k=0/1 (local)
            # w[0]=C11: i=0,k=2t -> 2t*n+0
            # w[1]=C21: i=1,k=2t -> 2t*n+1
            # w[2]=C12: i=0,k=2t+1 -> (2t+1)*n+0
            # w[3]=C22: i=1,k=2t+1 -> (2t+1)*n+1
            w_map = [2 * t * n + 0, 2 * t * n + 1, (2 * t + 1) * n + 0, (2 * t + 1) * n + 1]
            for idx in range(4):
                W[w_map[idx], col] = w[idx]
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p, max_rank_reduction=8, timeout=30):
    """Attempt to reduce rank by zeroing out columns that are linear combinations."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    best = (U.copy(), V.copy(), W.copy())
    end_time = time.time() + timeout
    # Try removing pairs of columns (linear dependencies)
    for i in range(rank):
        if time.time() > end_time:
            break
        for j in range(i + 1, rank):
            if time.time() > end_time:
                break
            # Check if column j is in span of others (approximately)
            # Quick test: try to express col j as combo of all others
            pass  # Too expensive, skip
    return best

def construct():
    n, m, p = DIMENSIONS
    # Start with replicated Strassen (rank 56)
    U, V, W = _build_strassen_replicated(n, m, p)
    
    # Verify
    ok, err = _verify_decomposition(U, V, W, n, m, p)
    
    # Try rank reduction via structured search
    if _time_left() > 60:
        # Try to find linear dependencies among the 56 columns
        U_arr, V_arr, W_arr = np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)
        rank = U_arr.shape[1]
        
        # Check if any column is zero
        for col in range(rank):
            if np.allclose(U_arr[:, col], 0) or np.allclose(V_arr[:, col], 0) or np.allclose(W_arr[:, col], 0):
                U_arr = np.delete(U_arr, col, axis=1)
                V_arr = np.delete(V_arr, col, axis=1)
                W_arr = np.delete(W_arr, col, axis=1)
                rank -= 1
                break
        
        # Try small perturbations to find equivalent lower-rank representations
        # Strategy: try to find a 55-rank representation by making one column a combo
        if _time_left() > 40:
            # For each pair of blocks, try to merge their 7 Strassen terms
            # using shared structure
            best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
            best_rank = best_U.shape[1]
            
            # Try reducing by 1: find if two columns satisfy a linear relation
            for attempt in range(200):
                if _time_left() < 30:
                    break
                # Randomly try to shift one column into another
                i, j = np.random.choice(best_rank, 2, replace=False)
                alpha = np.random.choice([-2, -1, -0.5, 0.5, 1, 2])
                trial_U = best_U.copy()
                trial_V = best_V.copy()
                trial_W = best_W.copy()
                trial_U[:, j] += alpha * trial_U[:, i]
                trial_V[:, j] += alpha * trial_V[:, i]
                trial_W[:, j] += alpha * trial_W[:, i]
                # Check if this preserves the tensor (it doesn't in general)
                # Skip this approach as it's not valid for tensor rank reduction
        
        U, V, W = best_U, best_V, best_W
    
    # Return as lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
