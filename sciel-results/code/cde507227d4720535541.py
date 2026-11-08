"""Combined program: gradient-based tensor decomposition search with structured search."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]
START_TIME = time.time()
TIME_LIMIT = 135  # seconds, safety margin before 150s kill

def _make_target(n, m, p):
    """Build the matrix multiplication tensor T[a,b,c] with 1s at valid positions."""
    T = np.zeros((p * n, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k * n + i, i * m + j, j * p + k] = 1.0
    return T

def _decomp_tensor(U, V, W):
    """Compute T_decomp[a,b,c] = sum_r W[a,r]*U[b,r]*V[c,r]."""
    return np.einsum('ar,br,cr->abc', W, U, V)

def _validate(U, V, W, n, m, p, tol=1e-6):
    """Check if the decomposition is exact (returns bool + error)."""
    T = _make_target(n, m, p)
    Td = _decomp_tensor(np.asarray(U, float), np.asarray(V, float), np.asarray(W, float))
    err = np.linalg.norm(Td - T)
    return err < tol, err

def _rank_of(U, V, W):
    if not U or not U[0]:
        return 0
    return len(U[0])

def _try_rank_gradient(rank, n, m, p, max_iters, seed, target):
    """Gradient descent on ||T_decomp - T||^2 to find exact decomposition."""
    rng = np.random.RandomState(seed)
    nm, mp, pn = n * m, m * p, p * n

    U = rng.randn(nm, rank) * 0.3
    V = rng.randn(mp, rank) * 0.3
    W = rng.randn(pn, rank) * 0.3

    best_err = float('inf')
    best = (U.copy(), V.copy(), W.copy())

    lr = 0.02
    for it in range(max_iters):
        Td = _decomp_tensor(U, V, W)
        dT = Td - target  # gradient of 0.5*||Td-T||^2 w.r.t. Td
        # dU[b,r] = sum_{a,c} dT[a,b,c]*W[a,r]*V[c,r]
        dU = np.einsum('abc,ar,cr->br', dT, W, V)
        dV = np.einsum('abc,ar,br->cr', dT, W, U)
        dW = np.einsum('abc,br,cr->ar', dT, U, V)
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        err = np.linalg.norm(dT)
        if err < best_err:
            best_err = err
            best = (U.copy(), V.copy(), W.copy())
        if err < 1e-6:
            break
        # adaptive lr
        if it > 0 and err > best_err * 1.5:
            lr *= 0.5
    return best, best_err

def _polish(U, V, W, n, m, p, target, max_iters=2000):
    """High-precision polish with decreasing lr to drive error below 1e-6."""
    U, V, W = [np.asarray(x, float) for x in (U, V, W)]
    best_err = float('inf')
    best = (U.copy(), V.copy(), W.copy())
    lr = 0.005
    for it in range(max_iters):
        Td = _decomp_tensor(U, V, W)
        dT = Td - target
        dU = np.einsum('abc,ar,cr->br', dT, W, V)
        dV = np.einsum('abc,ar,br->cr', dT, W, U)
        dW = np.einsum('abc,br,cr->ar', dT, U, V)
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        err = np.linalg.norm(dT)
        if err < best_err:
            best_err = err
            best = (U.copy(), V.copy(), W.copy())
        if err < 1e-9:
            break
        if it % 100 == 99:
            lr *= 0.7
    return best, best_err

def _greedy_improve(U, V, W, n, m, p, target, max_rounds=200):
    """Reduce rank by removing columns that don't break validity."""
    U = np.asarray(U, float); V = np.asarray(V, float); W = np.asarray(W, float)
    current_rank = U.shape[1]
    for _ in range(max_rounds):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
        removed = False
        for c in range(current_rank):
            U_new = np.delete(U, c, axis=1)
            V_new = np.delete(V, c, axis=1)
            W_new = np.delete(W, c, axis=1)
            ok, err = _validate(U_new, V_new, W_new, n, m, p)
            if ok:
                U, V, W = U_new, V_new, W_new
                current_rank -= 1
                removed = True
                break
        if not removed:
            break
    return U, V, W, current_rank

def _to_lists(U, V, W):
    # Round near-integer/half-integer values to clean representation
    def clean(M):
        out = []
        for row in M:
            r = []
            for x in row:
                # snap to nearest 0.5
                q = round(x * 2) / 2
                r.append(q)
            out.append(r)
        return out
    return {"U": clean(U), "V": clean(V), "W": clean(W)}

def _make_schoolbook(n, m, p):
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

def _try_structured_rank(n, m, p, rank, target):
    """
    Try structured decompositions based on patterns.
    For 3x3 x 3x6, try patterns where U depends on (i,j), V on (j,k), W on (k,i)
    with specific sparse assignments.
    """
    nm, mp, pn = n * m, m * p, p * n
    vals = [-1, -0.5, 0, 0.5, 1]
    
    # Try sparse structured patterns
    for seed in range(20):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            return None
        rng = np.random.RandomState(seed)
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        # Assign non-zero entries sparsely
        for r in range(rank):
            # Choose which (i,j) pairs get non-zero in U for this rank component
            i_idx = rng.randint(0, n)
            j_idx = rng.randint(0, m)
            U[i_idx * m + j_idx, r] = rng.choice(vals)
            
            k_idx = rng.randint(0, p)
            V[j_idx * p + k_idx, r] = rng.choice(vals)
            
            W[k_idx * n + i_idx, r] = rng.choice(vals)
        
        ok, err = _validate(U, V, W, n, m, p)
        if ok:
            return U, V, W
    
    return None

def _try_random_sparse(n, m, p, rank, target, max_attempts=500):
    """Try random sparse decompositions with integer/half-integer entries."""
    nm, mp, pn = n * m, m * p, p * n
    vals = [-1, -0.5, 0, 0.5, 1]
    
    for attempt in range(max_attempts):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            return None
        
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        # Each entry has ~30% chance of being non-zero
        for r in range(rank):
            for idx in range(nm):
                if random.random() < 0.3:
                    U[idx, r] = random.choice(vals)
            for idx in range(mp):
                if random.random() < 0.3:
                    V[idx, r] = random.choice(vals)
            for idx in range(pn):
                if random.random() < 0.3:
                    W[idx, r] = random.choice(vals)
        
        ok, err = _validate(U, V, W, n, m, p)
        if ok:
            return U, V, W
    
    return None

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = n * m * p
    lower_bound = max(nm, mp, pn)

    target = _make_target(n, m, p)

    # Fallback schoolbook
    U_fb, V_fb, W_fb = _make_schoolbook(n, m, p)
    best_result = _to_lists(U_fb, V_fb, W_fb)
    best_rank = schoolbook_rank

    # Search from a reasonable lower bound up to schoolbook
    # The true rank is likely well below 54; search upward from ~25.
    search_start = max(18, lower_bound)
    search_end = schoolbook_rank

    for rank in range(search_start, search_end + 1):
        if time.time() - START_TIME > TIME_LIMIT - 15:
            break

        found = False
        
        # First try structured and sparse random approaches (fast)
        if rank <= 36:
            res = _try_structured_rank(n, m, p, rank, target)
            if res is not None:
                U, V, W = res
                (U, V, W), err2 = _polish(U, V, W, n, m, p, target, max_iters=500)
                if err2 < 1e-6:
                    U, V, W, r = _greedy_improve(U, V, W, n, m, p, target)
                    if r < best_rank:
                        best_rank = r
                        best_result = _to_lists(U, V, W)
                        found = True
        
        if not found:
            res = _try_random_sparse(n, m, p, rank, target, max_attempts=200)
            if res is not None:
                U, V, W = res
                (U, V, W), err2 = _polish(U, V, W, n, m, p, target, max_iters=500)
                if err2 < 1e-6:
                    U, V, W, r = _greedy_improve(U, V, W, n, m, p, target)
                    if r < best_rank:
                        best_rank = r
                        best_result = _to_lists(U, V, W)
                        found = True
        
        # Then try gradient-based search
        if not found:
            for seed in [42, 123, 7, 2024, 999, 314, 137, 555]:
                if time.time() - START_TIME > TIME_LIMIT - 10:
                    break
                max_iters = 1000
                (U, V, W), err = _try_rank_gradient(rank, n, m, p, max_iters, seed, target)
                if err < 1e-6:
                    # polish for extra precision
                    (U, V, W), err2 = _polish(U, V, W, n, m, p, target, max_iters=1500)
                    if err2 < 1e-6:
                        # greedy rank reduction
                        U, V, W, r = _greedy_improve(U, V, W, n, m, p, target)
                        if r < best_rank:
                            best_rank = r
                            best_result = _to_lists(U, V, W)
                            found = True
                        break
        
        if found:
            # try to find even lower rank
            continue
        else:
            # if can't find this rank, lower ranks unlikely; stop
            break

    return best_result
# EVOLVE-BLOCK-END
