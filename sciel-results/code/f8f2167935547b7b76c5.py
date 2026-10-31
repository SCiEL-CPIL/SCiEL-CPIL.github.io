"""Fresh design for the ternary covering-code task K_q(n, radius)."""
# EVOLVE-BLOCK-START
import time
import itertools

import numpy as np


def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin below the 120 s wall-clock limit

    # ---- exact cover test via a precomputed ball matrix (NumPy) ----
    A = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    N = A.shape[0]
    # M[i, j] == 1  <=>  Hamming(A[i], A[j]) <= radius
    M = (q ** n - 1 - (A[:, None, :] == A[None, :, :]).astype(np.int8).prod(axis=2)) <= radius

    def uncovered_count(S):
        return int(np.sum(np.any(M[S], axis=1)))

    def is_covering(S):
        return uncovered_count(S) == 0

    # ---- greedy max-coverage seed (fast) ----
    cov = np.zeros(N, dtype=bool)
    S = []
    for j in range(N):
        if cov[j]:
            continue
        S.append(j)
        cov |= M[:, j]
        if not cov.any():
            break

    best_S = list(S)
    best_len = len(S)

    # ---- local-search refinement (add / split / delete) ----
    while len(best_S) > 1 and time.time() < deadline - 3.0:
        S = list(best_S)
        improved = False
        # delete: drop a redundant codeword
        for i in range(len(S)):
            if is_covering(S[:i] + S[i + 1:]):
                S = S[:i] + S[i + 1:]
                improved = True
                break
        if not improved:
            # split: replace one codeword with two that cover at least as much
            for i in range(len(S)):
                old = S[i]
                alt = [S[:i] + S[i + 1:]]
                for d in range(n):
                    for v in range(q):
                        cand = A[old].copy()
                        cand[d] = v
                        j = int(np.where((A == cand).all(axis=1))[0][0])
                        if j != old:
                            alt.append(S[:i] + [j] + S[i + 1:])
                for candS in alt:
                    if is_covering(candS):
                        S = candS
                        improved = True
                        break
                if improved:
                    break
        if improved and len(S) < best_len:
            best_S = S
            best_len = len(S)
        elif not improved:
            break

    return [A[i].tolist() for i in best_S]
# EVOLVE-BLOCK-END
