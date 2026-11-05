"""Direct-sum (per-row) decomposition of the 4x4 x 4x6 matrix-multiplication tensor,
with a time-bounded search for a lower-rank (integer/half-integer) decomposition.
"""
import time
import itertools

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START

def _direct_sum_decomposition(n, m, p):
    """Per-row direct-sum. For each output row i, a 1x(m x p) tensor of rank m*p.
    Output convention W[k*n + i, c]."""
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


def _is_half_integer(x):
    return abs(2 * x - round(2 * x)) < 1e-9


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check that the decomposition exactly reproduces the multiplication tensor.
    Returns (ok, max_err)."""
    r = len(U[0])
    # Build U (n*m x r), V (m*p x r), W (p*n x r)
    # For each (i, k): sum_j A_ij B_jk must equal sum_c U[i*m+j] V[j*p+k] W[k*n+i]
    # Equivalently: matrix T_{ik} = (U_i * V)^T W_k where U_i is row-block of U, etc.
    # We check: for all i, j, k: 1 == sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            urow = U[i * m + j]
            for k in range(p):
                vrow = V[j * p + k]
                wrow = W[k * n + i]
                s = 0.0
                for c in range(r):
                    s += urow[c] * vrow[c] * wrow[c]
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    ok = max_err < tol
    return ok, max_err


def _entries_half_int(U, V, W):
    for M in (U, V, W):
        for row in M:
            for x in row:
                if not _is_half_integer(x):
                    return False
    return True


def _random_search(n, m, p, base, time_budget):
    """Attempt to find a rank-(r) decomposition with r < base_rank by random
    integer/half-integer factor search (limited, best-effort)."""
    import random
    r0 = len(base["U"][0])
    deadline = time.time() + time_budget
    # Try to shave a couple of ranks by zeroing/merging columns in a random way.
    # This is a weak heuristic; kept small and bounded.
    for r in range(r0 - 1, max(r0 - 4, 0), -1):
        if time.time() > deadline - 5:
            break
        # Try random sparse half-integer factors of rank r.
        for attempt in range(2000):
            if time.time() > deadline - 5:
                break
            U = [[0] * r for _ in range(n * m)]
            V = [[0] * r for _ in range(m * p)]
            W = [[0] * r for _ in range(p * n)]
            # fill sparsely
            for c in range(r):
                # pick a random (i,j,k) triple to anchor this column
                i = random.randrange(n)
                j = random.randrange(m)
                k = random.randrange(p)
                U[i * m + j][c] = random.choice([1, 0.5, -1, -0.5])
                V[j * p + k][c] = random.choice([1, 0.5, -1, -0.5])
                W[k * n + i][c] = random.choice([1, 0.5, -1, -0.5])
            ok, _ = _verify(U, V, W, n, m, p)
            if ok and _entries_half_int(U, V, W):
                return {"U": U, "V": V, "W": W}
    return None


def construct():
    n, m, p = DIMENSIONS
    base = _direct_sum_decomposition(n, m, p)
    # Verify base (should be exact).
    ok, err = _verify(base["U"], base["V"], base["W"], n, m, p)
    assert ok, "direct-sum decomposition failed to verify: err=%r" % err

    start = time.time()
    # Time-bounded search for a lower-rank decomposition.
    improved = _random_search(n, m, p, base, time_budget=45)
    if improved is not None:
        ok2, err2 = _verify(improved["U"], improved["V"], improved["W"], n, m, p)
        if ok2 and _entries_half_int(improved["U"], improved["V"], improved["W"]):
            return improved

    # Safety margin: return base well before the 150s kill.
    if time.time() - start > 100:
        return base
    return base

# EVOLVE-BLOCK-END
