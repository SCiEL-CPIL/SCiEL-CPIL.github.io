"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

_START = time.time()
_DEADLINE = _START + 130  # 20s safety margin before 150s kill


def _elapsed():
    return time.time() - _START


def _remaining():
    return _DEADLINE - time.time()


def _score(U):
    """Compute the lower bound score for a set U."""
    if not U or 0 not in U:
        return 0.0
    n = len(U)
    maxU = max(U)
    if maxU == 0:
        return 0.0
    U_set = set(U)
    sum_set = set()
    diff_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / math.log(2 * maxU + 1)


def _build_binary(n):
    """Build a binary-like set (powers of 2)."""
    U = [0]
    for i in range(1, n + 1):
        U.append(2 ** i)
    return U


def _build_geometric(n, r, base=1):
    """Build a geometric progression set."""
    U = []
    x = base
    for i in range(n):
        U.append(x)
        x *= r
    U.sort()
    if 0 not in U:
        U.insert(0, 0)
    return U


def _build_mixed(n, r1=3, r2=5):
    """Build a mixed geometric set."""
    U = set([0])
    x = 1
    for i in range(n):
        U.add(x)
        if i % 2 == 0:
            x *= r1
        else:
            x *= r2
    return sorted(U)


def _local_search(U, max_iter=50):
    """Local search to improve the set."""
    best_U = list(U)
    best_score = _score(best_U)

    for _ in range(max_iter):
        if _remaining() < 5:
            break

        improved = False
        n = len(best_U)

        # Try removing elements
        for i in range(len(best_U)):
            if best_U[i] == 0:
                continue
            new_U = best_U[:i] + best_U[i+1:]
            if len(new_U) < 2:
                continue
            s = _score(new_U)
            if s > best_score + 1e-9:
                best_U = new_U
                best_score = s
                improved = True
                break

        if improved:
            continue

        # Try adding elements
        maxU = max(best_U)
        candidates = [1, 2, maxU + 1, maxU * 2, maxU * 3]
        random.shuffle(candidates)
        for c in candidates:
            if c in best_U or c < 0:
                continue
            new_U = sorted(best_U + [c])
            s = _score(new_U)
            if s > best_score + 1e-9:
                best_U = new_U
                best_score = s
                improved = True
                break

        if improved:
            continue

        # Try replacing elements
        for i in range(len(best_U)):
            if best_U[i] == 0:
                continue
            maxU = max(best_U)
            replacements = [best_U[i] + 1, best_U[i] - 1, best_U[i] * 2,
                            maxU + random.randint(1, 10)]
            random.shuffle(replacements)
            for r in replacements:
                if r < 0 or r in best_U or r == best_U[i]:
                    continue
                new_U = best_U.copy()
                new_U[i] = r
                new_U.sort()
                s = _score(new_U)
                if s > best_score + 1e-9:
                    best_U = new_U
                    best_score = s
                    improved = True
                    break
            if improved:
                break

        if not improved:
            break

    return best_U


def _evolve_initial(seed=42):
    """Create an initial set with evolution."""
    random.seed(seed)
    n = random.randint(5, 15)
    r = random.choice([2, 3, 4, 5, 7, 10])
    U = _build_geometric(n, r)
    U = _local_search(U, max_iter=30)
    return U


def construct():
    """Construct a finite set U of nonnegative integers containing 0 to maximize the score."""
    best_U = [0, 1, 3]
    best_score = _score(best_U)

    # Try several initial constructions
    initial_sets = [
        [0, 1, 3],
        _build_binary(5),
        _build_geometric(8, 3),
        _build_geometric(10, 2),
        _build_mixed(10, 3, 5),
    ]

    for U in initial_sets:
        if _remaining() < 10:
            break
        s = _score(U)
        if s > best_score:
            best_score = s
            best_U = list(U)

    # Local search on the best
    if _remaining() > 15:
        best_U = _local_search(best_U, max_iter=50)
        best_score = _score(best_U)

    # Try more evolved starts
    for seed in [1, 7, 42, 123, 456]:
        if _remaining() < 10:
            break
        U = _evolve_initial(seed)
        s = _score(U)
        if s > best_score:
            best_score = s
            best_U = list(U)

    # Final local search
    if _remaining() > 5:
        best_U = _local_search(best_U, max_iter=30)

    # Ensure 0 is in the set
    if 0 not in best_U:
        best_U.insert(0, 0)

    return sorted(best_U)
# EVOLVE-BLOCK-END
