"""Matrix multiplication 3x3 by 3x6: rank-minimal exact tensor decomposition.
SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N_DIM, M_DIM, P_DIM = 3, 3, 6


def _build_support(n, m, p):
    """Return on-support indices as arrays."""
    on = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                on.append((a, b, c))
    on = np.array(on, dtype=np.int64)
    return on


def _verify(U, V, W, n, m, p, on):
    """Exact (float) verification of the decomposition using vectorization."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    nm, mp, pn = n * m, m * p, p * n
    r = U.shape[1]
    if V.shape[1] != r or W.shape[1] != r:
        return False
    if U.shape[0] != nm or V.shape[0] != mp or W.shape[0] != pn:
        return False

    # Check on-support: sum_r U[a,r] V[b,r] W[c,r] == 1
    on_a, on_b, on_c = on[:, 0], on[:, 1], on[:, 2]
    vals = np.einsum('tr,tr->t', U[on_a], V[on_b] * W[on_c])
    if np.max(np.abs(vals - 1.0)) > 1e-8:
        return False

    # Check off-support
    ai = np.array([a // m for a in range(nm)])
    aj = np.array([a % m for a in range(nm)])
    bj = np.array([b // p for b in range(mp)])
    bk = np.array([b % p for b in range(mp)])
    ck = np.array([c // n for c in range(pn)])
    ci = np.array([c % n for c in range(pn)])
    
    for a in range(nm):
        for b in range(mp):
            if aj[a] != bj[b]:
                continue
            for c in range(pn):
                if bk[b] != ck[c] or ci[c] != ai[a]:
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val) > 1e-8:
                        return False
    return True


def _schoolbook(n, m, p):
    """Guaranteed rank n*m*p decomposition."""
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


def _als_decompose(U, V, W, on, n_iter, deadline, rng):
    """Alternating Least Squares to minimize sum (sum_r U[a,r]V[b,r]W[c,r] - 1)^2 on support."""
    on_a, on_b, on_c = on[:, 0], on[:, 1], on[:, 2]
    r = U.shape[1]
    
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for it in range(n_iter):
        if time.time() > deadline:
            break
            
        # Update U
        for a in range(U.shape[0]):
            mask = (on_a == a)
            if not np.any(mask):
                continue
            M = V[on_b[mask]] * W[on_c[mask]] # (T_a, r)
            A = M.T @ M + 1e-8 * np.eye(r)
            b_vec = M.T @ np.ones(len(M))
            U[a] = np.linalg.solve(A, b_vec)
            
        # Update V
        for b in range(V.shape[0]):
            mask = (on_b == b)
            if not np.any(mask):
                continue
            M = U[on_a[mask]] * W[on_c[mask]]
            A = M.T @ M + 1e-8 * np.eye(r)
            b_vec = M.T @ np.ones(len(M))
            V[b] = np.linalg.solve(A, b_vec)
            
        # Update W
        for c in range(W.shape[0]):
            mask = (on_c == c)
            if not np.any(mask):
                continue
            M = U[on_a[mask]] * V[on_b[mask]]
            A = M.T @ M + 1e-8 * np.eye(r)
            b_vec = M.T @ np.ones(len(M))
            W[c] = np.linalg.solve(A, b_vec)
            
        # Check convergence
        vals = np.einsum('tr,tr->t', U[on_a], V[on_b] * W[on_c])
        loss = np.sum((vals - 1.0)**2)
        if loss < 1e-10:
            break
            
    return U, V, W


def _try_rank(n, m, p, r, on, n_restarts, deadline, rng):
    """Try to find an exact decomposition of rank r using ALS."""
    nm, mp, pn = n * m, m * p, p * n
    
    for restart in range(n_restarts):
        if time.time() > deadline - 1:
            break
            
        U = rng.standard_normal((nm, r)) * 0.1
        V = rng.standard_normal((mp, r)) * 0.1
        W = rng.standard_normal((pn, r)) * 0.1
        
        U, V, W = _als_decompose(U, V, W, on, n_iter=20, deadline=deadline - 1, rng=rng)
        
        # Verify
        if _verify(U, V, W, n, m, p, on):
            # Try to snap to half-integers
            U2 = np.round(U * 2) / 2
            V2 = np.round(V * 2) / 2
            W2 = np.round(W * 2) / 2
            if _verify(U2, V2, W2, n, m, p, on):
                return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
            
            # Try integers
            U3 = np.round(U)
            V3 = np.round(V)
            W3 = np.round(W)
            if _verify(U3, V3, W3, n, m, p, on):
                return {"U": U3.tolist(), "V": V3.tolist(), "W": W3.tolist()}
                
    return None


def construct():
    """Find a rank-minimal exact tensor decomposition for 3x3 * 3x6."""
    start = time.time()
    deadline = start + 140  # safety margin
    n, m, p = N_DIM, M_DIM, P_DIM

    base = _schoolbook(n, m, p)
    base_rank = n * m * p
    on = _build_support(n, m, p)
    
    rng = np.random.default_rng(12345)
    
    best = base
    best_rank = base_rank
    
    # Lower bound: max(n*m, m*p, p*n) = 18
    for r in range(18, base_rank):
        if time.time() > deadline - 5:
            break
            
        time_left = deadline - 5 - time.time()
        # ALS is fast, allow many restarts
        n_restarts = min(100, int(time_left / 0.01) + 1)
        
        res = _try_rank(n, m, p, r, on, n_restarts, deadline - 5, rng)
        if res is not None:
            best = res
            best_rank = r
            break
            
    return best
# EVOLVE-BLOCK-END
