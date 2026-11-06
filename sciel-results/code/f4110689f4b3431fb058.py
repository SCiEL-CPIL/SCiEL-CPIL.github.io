import time
import random

N = 256


def _compute_sums(S):
    s = set(S)
    sums = set()
    for a in s:
        for b in s:
            if a <= b:
                sums.add(a + b)
    return sums


def _is_sidon(S):
    s = set(S)
    if len(s) != len(S):
        return False
    sums = set()
    lst = sorted(s)
    for i in range(len(lst)):
        for j in range(i, len(lst)):
            sm = lst[i] + lst[j]
            if sm in sums:
                return False
            sums.add(sm)
    return True


def _blocked_elements(sums):
    """Set of y in [1,N] that CANNOT be added (i.e. are 'blocked').

    y is blocked iff 2y is a sum, or exists a sum s with (s - 2y) a sum
    (equivalently s = 2y + d for some difference d between two sums).
    """
    blocked = set()
    for s in sums:
        if s % 2 == 0:
            y = s // 2
            if 1 <= y <= N:
                blocked.add(y)
    for a in sums:
        for b in sums:
            d = a - b
            if d < 0:
                continue
            y = (a - d) // 2
            if 1 <= y <= N:
                blocked.add(y)
    return blocked


def _is_maximal(S, sums):
    s = set(S)
    blocked = _blocked_elements(sums)
    for y in range(1, N + 1):
        if y in s:
            continue
        if y not in blocked:
            return False
    return True


def _greedy_build(order, sums_check=True):
    """Build a maximal Sidon set by adding elements in the given order."""
    S = []
    sums = set()
    for x in order:
        # new sums: 2x and x+s for s in S
        ok = True
        if 2 * x in sums:
            ok = False
        else:
            for s in S:
                if x + s in sums:
                    ok = False
                    break
        if ok:
            S.append(x)
            sums.add(2 * x)
            for s in S[:-1]:
                sums.add(x + s)
    return S, sums


def _local_search_reduce(S, deadline, rng):
    """Remove elements while preserving inclusion-maximality to shrink |S|."""
    current = list(S)
    current_sums = _compute_sums(current)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        indices = list(range(len(current)))
        rng.shuffle(indices)
        for idx in indices:
            if time.time() > deadline:
                break
            cand = current[:idx] + current[idx + 1:]
            if len(cand) < 1:
                continue
            cand_sums = _compute_sums(cand)
            if _is_maximal(cand, cand_sums):
                current = cand
                current_sums = cand_sums
                improved = True
                break
    return current


def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin well under the 60s wall clock
    rng = random.Random(12345)

    best_set = None
    best_size = N + 1

    def consider(S):
        nonlocal best_set, best_size
        if not S:
            return
        if not _is_sidon(S):
            return
        if not _is_maximal(S, _compute_sums(S)):
            return
        if len(S) < best_size:
            best_size = len(S)
            best_set = list(S)

    # 1) Greedy ascending baseline (deterministic, quick)
    order = list(range(1, N + 1))
    S, sums = _greedy_build(order)
    consider(S)

    # 2) Randomized greedy restarts -> shrink each via local search
    trials = 0
    while time.time() < deadline:
        trials += 1
        order = list(range(1, N + 1))
        rng.shuffle(order)
        S, sums = _greedy_build(order)
        if not S:
            continue
        # quick maximal check on the raw build
        if _is_maximal(S, sums):
            reduced = _local_search_reduce(S, deadline, rng)
            consider(reduced)
        # Also try reducing even if not yet maximal (reduction may help)
        if time.time() < deadline:
            reduced = _local_search_reduce(S, deadline, rng)
            consider(reduced)

    # 3) Extra polish pass on the best set found (more removal attempts)
    if best_set is not None and time.time() < deadline:
        polished = _local_search_reduce(best_set, deadline, rng)
        consider(polished)

    if best_set is None:
        # Fallback: guaranteed valid maximal set from ascending greedy
        S, sums = _greedy_build(list(range(1, N + 1)))
        best_set = S

    # Final exact verification
    if not _is_sidon(best_set):
        S, sums = _greedy_build(list(range(1, N + 1)))
        best_set = S

    return best_set
