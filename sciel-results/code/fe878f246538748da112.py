"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return U, V, W


def _check_exact(U, V, W, n, m, p, tol=1e-9):
    """Verify T[i,j,k] == sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] for all (i,j,k)."""
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _check_exact_frac(U, V, W, n, m, p):
    """Exact rational verification."""
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0, 1)
                for c in range(r):
                    s += Fraction(U[i * m + j][c]) * Fraction(V[j * p + k][c]) * Fraction(W[k * n + i][c])
                if s != Fraction(1, 1):
                    return False
    return True


def _try_rank_optimization(n, m, p, target_rank, seed, max_time=10.0):
    """
    Try to find a valid decomposition at the given rank using gradient-based optimization.
    Returns (U, V, W) as lists or None.
    """
    from scipy.optimize import minimize
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    # Parameter layout: U (nm*rank), V (mp*rank), W (pn*rank)
    total_params = nm * target_rank + mp * target_rank + pn * target_rank
    
    def unpack(params):
        U = params[:nm * target_rank].reshape(nm, target_rank)
        V = params[nm * target_rank:nm * target_rank + mp * target_rank].reshape(mp, target_rank)
        W = params[nm * target_rank + mp * target_rank:].reshape(pn, target_rank)
        return U, V, W
    
    def loss_and_grad(params):
        U, V, W = unpack(params)
        loss = 0.0
        # Compute gradient
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(target_rank):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    err = val - 1.0
                    loss += err * err
                    for c in range(target_rank):
                        grad_U[i * m + j, c] += 2 * err * V[j * p + k, c] * W[k * n + i, c]
                        grad_V[j * p + k, c] += 2 * err * U[i * m + j, c] * W[k * n + i, c]
                        grad_W[k * n + i, c] += 2 * err * U[i * m + j, c] * V[j * p + k, c]
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return loss, grad
    
    best_loss = float('inf')
    best_params = None
    
    n_restarts = 3
    for restart in range(n_restarts):
        if time.time() - start_time > max_time:
            break
        
        # Initialize with small random values
        params = rng.randn(total_params) * 0.5
        
        try:
            result = minimize(
                loss_and_grad, params,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.fun < best_loss:
                best_loss = result.fun
                best_params = result.x
        except Exception:
            pass
        
        if best_loss < 1e-8:
            break
    
    if best_params is None or best_loss > 1e-6:
        return None
    
    # Extract and verify
    U, V, W = unpack(best_params)
    if _check_exact(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-6):
        return U.tolist(), V.tolist(), W.tolist()
    
    return None


def _try_structured_reduction(n, m, p, base_rank, seed, max_time=5.0):
    """
    Try to reduce rank by finding linear dependencies among columns of the schoolbook decomposition.
    This is a heuristic approach that may or may not work.
    """
    start_time = time.time()
    rng = random.Random(seed)
    
    # Get schoolbook decomposition
    U0, V0, W0 = _schoolbook(n, m, p)
    rank = len(U0[0])
    
    # Try to find if any column can be expressed as a linear combination of others
    # This is a complex problem, so we use a simple heuristic
    # For now, just return None to indicate no reduction found
    return None


def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin under 150s
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U0, V0, W0 = _schoolbook(n, m, p)
    best_rank = n * m * p
    best = (U0, V0, W0)
    
    # Try to find lower rank decompositions
    # Lower bound: max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # But realistically, we expect something higher
    min_possible = max(n * m, m * p, p * n)
    
    # Try ranks from lower to higher, with time budget
    # We'll try a few candidate ranks below schoolbook
    candidate_ranks = []
    for r in range(min_possible, best_rank + 1):
        if r % 5 == 0:  # Try every 5th rank
            candidate_ranks.append(r)
        if time.time() > deadline - 30:  # Leave 30s for finalization
            break
    
    # Sort by rank (ascending)
    candidate_ranks.sort()
    
    for target_rank in candidate_ranks:
        if time.time() > deadline - 20:  # Leave 20s for finalization
            break
        
        # Try optimization-based approach
        result = _try_rank_optimization(n, m, p, target_rank, seed=42 + target_rank, max_time=8.0)
        if result is not None:
            U, V, W = result
            # Verify once more
            if _check_exact(U, V, W, n, m, p, tol=1e-6):
                best = (U, V, W)
                best_rank = target_rank
                break  # Found the lowest rank, stop
    
    # If no improvement, return schoolbook
    U, V, W = best
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
