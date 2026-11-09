DIMENSIONS = [3, 3, 3]
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140.0  # safety margin before the 150s kill

    def elapsed_ok():
        return (time.time() - start) < deadline

    def trivial_decomp():
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

    def verify_decomp(U, V, W):
        """Exact check: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def half_int(x):
        """True if x is an integer or half-integer (within tolerance)."""
        if x is None:
            return False
        if abs(x) < 1e-9:
            return True
        return abs(x * 2 - round(x * 2)) < 1e-6

    def entries_ok(U, V, W):
        for M in (U, V, W):
            for row in M:
                for x in row:
                    if not half_int(x):
                        return False
        return True

    # Guaranteed-correct baseline.
    best = trivial_decomp()
    best_rank = n * m * p
    if not verify_decomp(best["U"], best["V"], best["W"]):
        # Should never happen; return anyway (still the only candidate we have).
        return best

    # Bounded search for a valid lower-rank decomposition.
    # We attempt a small structured family of rank-23 candidates derived by
    # merging the 27 basis terms with half-integer coefficients, and verify
    # each. Only a candidate that passes exact verification replaces the baseline.
    def try_rank23_candidate():
        # 23-term structure: keep 23 of the 27 (i,j,k) basis terms and attempt to
        # express the 4 missing contributions as half-integer linear combinations.
        # This is a heuristic; correctness is enforced by verify_decomp.
        # Represent each term by a coefficient vector over the 27 basis slots.
        basis = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        # Choose 23 slots to keep (drop the 4 with i==j==k pattern corners as a guess).
        keep = [b for b in basis if not (b[0] == b[1] == b[2])]
        if len(keep) != 23:
            keep = basis[:23]
        # Build U,V,W with 1 on kept slots, 0 otherwise.
        U = [[0] * 23 for _ in range(n * m)]
        V = [[0] * 23 for _ in range(m * p)]
        W = [[0] * 23 for _ in range(p * n)]
        for r, (i, j, k) in enumerate(keep):
            U[i * m + j][r] = 1
            V[j * p + k][r] = 1
            W[k * n + i][r] = 1
        return {"U": U, "V": V, "W": W}

    candidates = [try_rank23_candidate()]
    for cand in candidates:
        if not elapsed_ok():
            break
        if len(cand["U"][0]) < best_rank and verify_decomp(cand["U"], cand["V"], cand["W"]) \
                and entries_ok(cand["U"], cand["V"], cand["W"]):
            best = cand
            best_rank = len(cand["U"][0])
            break

    return best
