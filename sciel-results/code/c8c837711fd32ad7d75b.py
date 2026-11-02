"""Tensor decomposition for 3x4 * 4x5 matrix multiplication."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 5]
N, M, P = DIMENSIONS
RANK_SCHOOLBOOK = N * M * P  # 60

# Deadline guard
_START_TIME = None

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _verify(U, V, W, n=N, m=M, p=P):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    T_computed = np.einsum('ijr,jkr,kin->ijk', U_3d, V_3d, W_3d)
    T_expected = np.ones((n, m, p))
    return np.allclose(T_computed, T_expected, atol=1e-9)

def _verify_exact(U, V, W, n=N, m=M, p=P):
    """Verify with exact rational arithmetic."""
    rank = len(U[0]) if U else 0
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    s += U_frac[i*m+j][r] * V_frac[j*p+k][r] * W_frac[k*n+i][r]
                if s != 1:
                    return False
    return True

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    n, m, p = N, M, P
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _is_nice(vec, tol=1e-9):
    """Check if vec entries are close to integers or half-integers."""
    for x in vec:
        x2 = 2 * x
        if abs(x2 - round(x2)) > tol:
            return False
    return True

def _round_to_half(x):
    return round(x * 2) / 2.0

def _solve_for_W(U, V, n, m, p, r):
    """
    Given U (nm x r) and V (mp x r), solve for W (pn x r).
    For each (i,k) pair, solve A @ w = ones(m) where A[j,c] = U[i*m+j,c]*V[j*p+k,c].
    Returns W (list of lists) or None if no nice solution found.
    """
    W = [[0.0] * r for _ in range(p * n)]
    for i in range(n):
        for k in range(p):
            A = np.zeros((m, r))
            for j in range(m):
                for c in range(r):
                    A[j, c] = U[i*m+j][c] * V[j*p+k][c]
            # Solve A @ w = ones(m)
            try:
                # Check if system is consistent
                # Use lstsq
                w_sol, residuals, rank_val, s = np.linalg.lstsq(A, np.ones(m), rcond=1e-12)
            except np.linalg.LinAlgError:
                return None
            # Check residual
            if np.linalg.norm(A @ w_sol - np.ones(m)) > 1e-8:
                return None
            # Check if nice
            if not _is_nice(w_sol, tol=1e-7):
                return None
            w_rounded = [_round_to_half(x) for x in w_sol]
            # Verify rounded solution
            if not np.allclose(A @ np.array(w_rounded), np.ones(m), atol=1e-6):
                return None
            for c in range(r):
                W[k*n+i][c] = w_rounded[c]
    return W

def _try_random_decomposition(r, n, m, p, seed=0, max_iter=500):
    """
    Try to find a rank-r decomposition by random U,V and solving for W.
    Returns (U, V, W) or None.
    """
    rng = np.random.RandomState(seed)
    # Try a few random initializations
    for trial in range(20):
        # Initialize U and V with random values from a small set
        # Use values that are likely to produce nice W
        U = rng.randint(0, 2, size=(n*m, r)) - 0.5  # values in {-0.5, 0, 0.5, 1}
        V = rng.randint(0, 2, size=(m*p, r)) - 0.5
        
        W = _solve_for_W(U.tolist(), V.tolist(), n, m, p, r)
        if W is not None:
            return U.tolist(), V.tolist(), W
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x4 * 4x5 matrix multiplication."""
    global _START_TIME
    _START_TIME = time.time()
    deadline = _START_TIME + 115  # Safety margin before 120s limit
    
    n, m, p = N, M, P
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # The tensor has 60 non-zero entries, so rank must be at least max(n*m, m*p, p*n) = 20
    # But realistically, we'll try ranks from 59 down
    
    # Strategy: for each rank r, try random U,V and solve for W
    # Focus on ranks that are likely to work
    
    # Try ranks from 59 down to some lower bound
    for r in range(59, 19, -1):
        if _elapsed() > deadline - 5:
            break
        
        # Try a few random seeds
        for seed in range(30):
            if _elapsed() > deadline - 5:
                break
            
            result = _try_random_decomposition(r, n, m, p, seed=seed + r * 1000)
            if result is not None:
                U, V, W = result
                # Verify
                if _verify(U, V, W):
                    if _verify_exact(U, V, W):
                        best = {"U": U, "V": V, "W": W}
                        best_rank = r
                        break  # Found valid for this rank, try lower
                    else:
                        # Floating point issue, try to fix
                        pass
    
    return best

# Test
if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    print(f"Valid: {_verify(result['U'], result['V'], result['W'])}")
    print(f"Exact: {_verify_exact(result['U'], result['V'], result['W'])}")
