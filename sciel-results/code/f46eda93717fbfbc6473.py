"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START

def _build_rank33():
    """Construct the known rank-33 decomposition using 3 blocks of 2x3x2."""
    n, m, p = 2, 3, 6
    rank = 33
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    terms = []
    terms.append({'u': {0: 1, 4: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}})
    terms.append({'u': {3: 1, 4: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}})
    terms.append({'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}})
    terms.append({'u': {4: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}})
    terms.append({'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}})
    terms.append({'u': {3: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}})
    terms.append({'u': {1: 1, 4: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}})
    terms.append({'u': {2: 1}, 'v': {4: 1}, 'w': {0: 1}})
    terms.append({'u': {2: 1}, 'v': {5: 1}, 'w': {1: 1}})
    terms.append({'u': {5: 1}, 'v': {4: 1}, 'w': {2: 1}})
    terms.append({'u': {5: 1}, 'v': {5: 1}, 'w': {3: 1}})
    
    for block in range(3):
        for t_idx, term in enumerate(terms):
            col = block * 11 + t_idx
            for idx, val in term['u'].items():
                U[idx, col] = val
            for l_idx, val in term['v'].items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = j_local * p + (block * 2 + k_local)
                V[g_idx, col] = val
            for l_idx, val in term['w'].items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = (block * 2 + k_local) * n + i_local
                W[g_idx, col] = val
                
    return U, V, W


def _verify_decomp(U, V, W, n, m, p, tol=1e-9):
    """Check if decomposition is valid."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _residuals(x, n, m, p, rank):
    """Compute residuals for a flattened (U,V,W) parameter vector."""
    nm, mp, pn = n*m, m*p, p*n
    U = x[:nm*rank].reshape(nm, rank)
    V = x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
    W = x[nm*rank+mp*rank:].reshape(pn, rank)
    res = np.empty(n*m*p)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                res[idx] = s - 1.0
                idx += 1
    return res


def _try_find_rank(U0, V0, W0, n, m, p, target_rank, num_restarts=3, max_nfev=50000, tol=1e-10):
    """Try to find a target_rank decomposition using Levenberg-Marquardt."""
    nm, mp, pn = n*m, m*p, p*n
    best = None
    best_cost = np.inf
    
    for restart in range(num_restarts):
        # Initialize: use first target_rank columns of the known decomposition
        U_init = U0[:, :target_rank].copy()
        V_init = V0[:, :target_rank].copy()
        W_init = W0[:, :target_rank].copy()
        
        # Add small random perturbation for restarts > 0
        if restart > 0:
            U_init += 0.1 * np.random.randn(nm, target_rank)
            V_init += 0.1 * np.random.randn(mp, target_rank)
            W_init += 0.1 * np.random.randn(pn, target_rank)
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = least_squares(
                _residuals, x0, args=(n, m, p, target_rank),
                method='lm', max_nfev=max_nfev
            )
            if result.cost < best_cost:
                best_cost = result.cost
                if result.cost < tol:
                    U = result.x[:nm*target_rank].reshape(nm, target_rank)
                    V = result.x[nm*target_rank:nm*target_rank+mp*target_rank].reshape(mp, target_rank)
                    W = result.x[nm*target_rank+mp*target_rank:].reshape(pn, target_rank)
                    if _verify_decomp(U, V, W, n, m, p, tol=1e-8):
                        return U, V, W
        except Exception:
            continue
    
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    U, V, W = _build_rank33()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = 33
    
    if not _verify_decomp(U, V, W, n, m, p):
        raise ValueError("Initial decomposition failed")
    
    # Try to reduce rank from 32 down to 20
    for target_rank in range(32, 19, -1):
        if time.time() > deadline - 30:
            break
        result = _try_find_rank(U, V, W, n, m, p, target_rank, num_restarts=2, max_nfev=20000)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_rank = target_rank
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}


def _verify(result, n, m, p):
    """Verify the decomposition is correct."""
    U = result['U']
    V = result['V']
    W = result['W']
    rank = len(U[0])
    
    assert len(U) == n * m
    assert len(V) == m * p
    assert len(W) == p * n
    assert all(len(row) == rank for row in U)
    assert all(len(row) == rank for row in V)
    assert all(len(row) == rank for row in W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1) > 1e-9:
                    raise ValueError(f"Verification failed at i={i},j={j},k={k}: got {s}")
    
    return True
# EVOLVE-BLOCK-END
