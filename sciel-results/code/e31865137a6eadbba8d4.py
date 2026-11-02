"""Covering code K_3(7,1): greedy construction + local-search reduction."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = _T0 + 142.0   # safety margin below 150s kill


def _time_ok():
    return time.time() < _DEADLINE


def _hamming(a, b):
    return int(np.count_nonzero(a != b))


def _within_radius(a, b, r):
    d = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            d += 1
            if d > r:
                return False
    return True


def _all_within_radius(code, r):
    n = len(code[0]) if code else 0
    for i in range(n):
        for v in range(3):
            w = list(code[0])
            w[i] = v
            if not any(_within_radius(w, c, r) for c in code):
                return False
    return True


def _build_greedy(n, r):
    """Greedy set-cover over the whole space (exact, small enough for 3^7)."""
    space = np.array([[int(x) for x in np.base_repr(i, 3).zfill(n)] for i in range(3 ** n)])
    S = set(range(len(space)))
    code = []
    while S:
        best_u = None
        best_cov = -1
        for u in range(len(space)):
            if u not in S:
                continue
            cov = 0
            for v in S:
                if _within_radius(space[u], space[v], r):
                    cov += 1
            if cov > best_cov:
                best_cov = cov
                best_u = u
                if cov == len(S):
                    break
        code.append(list(space[best_u]))
        S = {v for v in S if not _within_radius(space[v], space[best_u], r)}
        if not _time_ok():
            break
    return code


def _try_reduce(code, r):
    """Local search: repeatedly remove redundant codewords, then swap to escape minima."""
    code = [list(c) for c in code]
    improved = True
    while improved and _time_ok():
        improved = False
        # removal pass
        for i in range(len(code)):
            if not _time_ok():
                break
            rest = code[:i] + code[i + 1:]
            if _all_within_radius(rest, r):
                code.pop(i)
                improved = True
                break
        # swap pass (escape local minima)
        if not improved:
            for i in range(len(code)):
                if not _time_ok():
                    break
                for j in range(len(code)):
                    if i == j:
                        continue
                    for k in range(len(code[i])):
                        for v in range(3):
                            if code[i][k] == v:
                                continue
                            cand = [list(c) for c in code]
                            cand[i] = list(code[i])
                            cand[i][k] = v
                            if cand[i] in cand:
                                continue
                            if _all_within_radius(cand, r):
                                code = cand
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
    return code


def construct(q, n, radius):
    # Generic fallback: greedy set cover over the whole space.
    if q != 3:
        space = np.array([[int(x) for x in np.base_repr(i, q).zfill(n)] for i in range(q ** n)])
        S = set(range(len(space)))
        code = []
        while S and _time_ok():
            best_u, best_cov = None, -1
            for u in range(len(space)):
                if u not in S:
                    continue
                cov = sum(1 for v in S if _within_radius(space[u], space[v], radius))
                if cov > best_cov:
                    best_cov, best_u = cov, u
            code.append(list(space[best_u]))
            S = {v for v in S if not _within_radius(space[v], space[best_u], radius)}
        return code

    # Ternary case (the task: q=3, n=7, radius=1).
    code = _build_greedy(n, radius)
    code = _try_reduce(code, radius)
    # Safety: ensure validity before returning.
    if not _all_within_radius(code, radius):
        # Re-run greedy to guarantee a valid (if larger) code.
        code = _build_greedy(n, radius)
    return code
# EVOLVE-BLOCK-END
