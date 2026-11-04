"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import numpy as np

# Keep the seed program interface exactly:
# construct() -> {'U', 'V', 'W'}
# The seed is rank 60 schoolbook; we return it immediately if no better
# decomposition is found within a strict time budget.

def _schoolbook(n=2, m=3, p=10):
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
    return U, V, W


def _verify(U, V, W, n=2, m=3, p=10):
    # Verify tensor decomposition:
    # T_{i,j,k} = sum_r U_{(i,m)+j,r} V_{(j,p)+k,r} W_{(k,n)+i,r}
    # where W is indexed as W[k*n + i, r] per task convention.
    U = np.asarray(U, dtype=np.int64)
    V = np.asarray(V, dtype=np.int64)
    W = np.asarray(W, dtype=np.int64)
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False

    # Precompute W transposed for speed: Wt[k*n+i, r]
    Wt = W  # already indexed as (k*n+i, r)

    # Build T as (n,m,p)
    T = np.zeros((n, m, p), dtype=np.int64)
    for r in range(rank):
        # U[:,r] is length n*m
        Ur = U[:, r]
        Vr = V[:, r]
        Wr = Wt[:, r]
        # outer-ish: for each (i,j) and (k,i') we need i==i'
        # We'll loop over i for clarity; rank is small.
        for i in range(n):
            Uij = Ur[i * m:(i + 1) * m]
            Wki = Wr[i::n]  # length p, entries W[k*n + i, r]
            # For each k, contribution to T[i, j, k] = Uij[j] * Vr[j*p + k] * Wki[k]
            for j in range(m):
                u = Uij[j]
                if u == 0:
                    continue
                Vr_j = Vr[j * p:(j + 1) * p]
                T[i, j, :] += u * Vr_j * Wki
    # Expected T_{i,j,k} = 1
    return np.all(T == 1)


def construct():
    n, m, p = DIMENSIONS
    # Deadline guard: hard cap well below 120s to avoid timeout.
    t0 = time.perf_counter()
    deadline = t0 + 100.0  # leave margin

    # Seed baseline
    U0, V0, W0 = _schoolbook(n, m, p)
    best = (len(U0[0]), U0, V0, W0)

    # Quick verification of baseline (very fast for rank 60)
    if _verify(U0, V0, W0, n, m, p):
        # Try to find a lower-rank decomposition if there's enough time.
        # We do a lightweight search over random integer/half-integer entries.
        # This is not guaranteed to improve, but can occasionally hit a known
        # decomposition if we sample near it.
        # We only do this if the verification above was fast.
        if time.perf_counter() < deadline - 60.0:
            # Use numpy for speed; search small ranks first.
            # We'll use a simple local search: start from random, verify,
            # and occasionally mutate.
            # Since rank < 60 for M_{2,3,10} is nontrivial, we keep the
            # search very small to avoid wasting time.
            rng = np.random.default_rng(0)
            for target_rank in [16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 40, 44, 48, 50, 52, 54, 56, 58]:
                if time.perf_counter() > deadline - 30.0:
                    break
                # Try a few random inits for this rank
                for attempt in range(20):
                    if time.perf_counter() > deadline - 10.0:
                        break
                    # Random half-integer entries in [-2,2]
                    U = (rng.integers(-4, 5, size=(n * m, target_rank)) / 2).astype(np.float64)
                    V = (rng.integers(-4, 5, size=(m * p, target_rank)) / 2).astype(np.float64)
                    W = (rng.integers(-4, 5, size=(p * n, target_rank)) / 2).astype(np.float64)
                    # Convert to Fraction for exact verification
                    try:
                        Uf = [[Fraction(x).limit_denominator(1000) for x in row] for row in U]
                        Vf = [[Fraction(x).limit_denominator(1000) for x in row] for row in V]
                        Wf = [[Fraction(x).limit_denominator(1000) for x in row] for row in W]
                    except Exception:
                        continue
                    if _verify(Uf, Vf, Wf, n, m, p):
                        # Found a valid lower-rank decomposition
                        if target_rank < best[0]:
                            best = (target_rank, Uf, Vf, Wf)
                            # If we got a very low rank, stop early
                            if target_rank <= 20:
                                break
                if best[0] <= 20:
                    break

    # Return the best found (possibly the seed)
    _, Ub, Vb, Wb = best
    return {"U": Ub, "V": Vb, "W": Wb}
# EVOLVE-BLOCK-END
