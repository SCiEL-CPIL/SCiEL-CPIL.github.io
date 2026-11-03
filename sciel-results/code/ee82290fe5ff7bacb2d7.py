"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    # Try to find a lower-rank decomposition
    best = None
    best_rank = n * m * p

    # Lower bound: max(n*m, m*p, p*n)
    min_rank = max(n * m, m * p, p * n)

    for target_rank in range(min_rank, best_rank):
        if time.time() > deadline:
            break

        # Try a few random restarts at this rank
        for restart in range(3):
            if time.time() > deadline:
                break
            sol = _try_rank(n, m, p, target_rank, deadline)
            if sol is not None:
                best = sol
                best_rank = target_rank
                break  # Found valid at this rank, try lower
        # If we found one at this rank, continue to try lower ranks
        if best is not None and best_rank == target_rank:
            continue
        # If no solution found at this rank, higher ranks are unlikely to help
        # but we keep the schoolbook as fallback
        break

    if best is not None:
        return best

    # Fallback: schoolbook decomposition
    return _schoolbook(n, m, p)


def _schoolbook(n, m, p):
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


def _verify(U, V, W, n, m, p):
    """Verify exact decomposition using Fraction arithmetic."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = Fraction(0)
                for c in range(rank):
                    total += Fraction(U[u_idx][c]) * Fraction(V[v_idx][c]) * Fraction(W[w_idx][c])
                if total != 1:
                    return False
    # Also check that non-tensor entries are 0 (rows/cols not in tensor)
    # For matrix multiplication tensor, every (u_idx, v_idx, w_idx) that
    # doesn't correspond to a valid (i,j,k) should sum to 0.
    # Actually the tensor only has 1s at valid (i,j,k) positions.
    # The decomposition must produce 0 everywhere else.
    # Check a few non-tensor positions
    for u_idx in range(n * m):
        for v_idx in range(m * p):
            for w_idx in range(p * n):
                # Check if this is a valid tensor position
                i = u_idx // m
                j = u_idx % m
                j2 = v_idx // p
                k = v_idx % p
                k2 = w_idx // n
                i2 = w_idx % n
                if not (j == j2 and k == k2 and i == i2):
                    total = Fraction(0)
                    for c in range(rank):
                        total += Fraction(U[u_idx][c]) * Fraction(V[v_idx][c]) * Fraction(W[w_idx][c])
                    if total != 0:
                        return False
    return True


def _try_rank(n, m, p, target_rank, deadline):
    """Try to find a rank-target_rank decomposition via iterative refinement."""
    np.random.seed(int(time.time() * 1000) % (2**31))

    U = np.random.randn(n * m, target_rank) * 0.1
    V = np.random.randn(m * p, target_rank) * 0.1
    W = np.random.randn(p * n, target_rank) * 0.1

    for iteration in range(80):
        if time.time() > deadline:
            return None

        # Update U: for each (i,j), solve linear system
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                A = np.zeros((p, target_rank))
                for k in range(p):
                    v_idx = j * p + k
                    w_idx = k * n + i
                    A[k, :] = V[v_idx, :] * W[w_idx, :]
                b = np.ones(p)
                try:
                    U[u_idx, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass

        # Update V: for each (j,k), solve linear system
        for j in range(m):
            for k in range(p):
                v_idx = j * p + k
                A = np.zeros((n, target_rank))
                for i in range(n):
                    u_idx = i * m + j
                    w_idx = k * n + i
                    A[i, :] = U[u_idx, :] * W[w_idx, :]
                b = np.ones(n)
                try:
                    V[v_idx, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass

        # Update W: for each (k,i), solve linear system
        for k in range(p):
            for i in range(n):
                w_idx = k * n + i
                A = np.zeros((m, target_rank))
                for j in range(m):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    A[j, :] = U[u_idx, :] * V[v_idx, :]
                b = np.ones(m)
                try:
                    W[w_idx, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass

        # Check convergence: compute residual
        if iteration % 10 == 0:
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                        err = abs(val - 1.0)
                        if err > max_err:
                            max_err = err
            if max_err < 1e-12:
                # Convert to exact fractions for verification
                U_frac = [[float(U[r][c]) for c in range(target_rank)] for r in range(n * m)]
                V_frac = [[float(V[r][c]) for c in range(target_rank)] for r in range(m * p)]
                W_frac = [[float(W[r][c]) for c in range(target_rank)] for r in range(p * n)]
                # Round to nearest rational with small denominator
                U_frac = [[_to_frac(x) for x in row] for row in U_frac]
                V_frac = [[_to_frac(x) for x in row] for row in V_frac]
                W_frac = [[_to_frac(x) for x in row] for row in W_frac]
                if _verify(U_frac, V_frac, W_frac, n, m, p):
                    return {"U": U_frac, "V": V_frac, "W": W_frac}

    return None


def _to_frac(x):
    """Convert a float to a Fraction, rounding to nearest simple rational."""
    if abs(x) < 1e-15:
        return 0
    # Try to find a simple fraction
    from fractions import Fraction
    # Round to 6 decimal places first
    x_rounded = round(x, 6)
    f = Fraction(x_rounded).limit_denominator(10000)
    return f
# EVOLVE-BLOCK-END
