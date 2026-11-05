import time
import numpy as np
from heapq import heappush, heappop

TOTAL_STATES = 12
MASK_COUNT = 1 << TOTAL_STATES

def _build_transition_tables(perm_a, perm_b):
    """Vectorized build of transition tables for all 4096 subsets."""
    indices = np.arange(MASK_COUNT, dtype=np.int32)
    # For each mask and each state bit, compute contribution
    bits = np.zeros((MASK_COUNT, TOTAL_STATES), dtype=np.int32)
    for s in range(TOTAL_STATES):
        bits[:, s] = (indices >> s) & 1
    # trans[mask] = OR over s in mask of (1 << perm[s])
    table_a = np.zeros(MASK_COUNT, dtype=np.int32)
    table_b = np.zeros(MASK_COUNT, dtype=np.int32)
    for s in range(TOTAL_STATES):
        table_a += bits[:, s] << perm_a[s]
        table_b += bits[:, s] << perm_b[s]
    return table_a, table_b

def _threshold_of(perm_a, perm_b):
    """BFS on subset automaton; returns shortest reset length or -1."""
    full = MASK_COUNT - 1
    if full & (full - 1) == 0:
        return 0
    t_a, t_b = _build_transition_tables(perm_a, perm_b)
    dist = np.full(MASK_COUNT, -1, dtype=np.int32)
    dist[full] = 0
    frontier = [full]
    head = 0
    while head < len(frontier):
        cur = frontier[head]
        head += 1
        d = dist[cur]
        if cur & (cur - 1) == 0:
            return int(d)
        na = int(t_a[cur])
        nb = int(t_b[cur])
        if dist[na] < 0:
            dist[na] = d + 1
            frontier.append(na)
        if dist[nb] < 0:
            dist[nb] = d + 1
            frontier.append(nb)
    return -1

def _known_tight_candidates():
    """Yield structured families known to contain Černý-tight automata."""
    n = TOTAL_STATES
    # Family A: a = full cycle, b = identity with single defect
    cyc = [(i + 1) % n for i in range(n)]
    for k in range(n):
        for j in range(n):
            if j != k:
                b = list(range(n))
                b[k] = j
                yield list(cyc), b
    # Family B: a = cycle, b = identity with two defects
    for k1 in range(n):
        for k2 in range(k1 + 1, n):
            for j1 in range(n):
                for j2 in range(n):
                    if j1 == k1 and j2 == k2:
                        continue
                    b = list(range(n))
                    b[k1] = j1
                    b[k2] = j2
                    yield list(cyc), b
    # Family C: a = shift by d, b = identity with one defect
    for d in range(1, n):
        sh = [(i + d) % n for i in range(n)]
        for k in range(n):
            for j in range(n):
                if j != k:
                    b = list(range(n))
                    b[k] = j
                    yield list(sh), b
    # Family D: a = cycle, b = near-permutation (one state maps to another's image)
    for k in range(n):
        for j in range(n):
            if j != k:
                b = [(i + 1) % n for i in range(n)]
                b[k] = j
                yield list(cyc), b

def _local_mutation_search(best_a, best_b, time_budget, rng):
    """Hill-climbing with random perturbations around a known-good automaton."""
    n = TOTAL_STATES
    best_score = _threshold_of(best_a, best_b)
    cur_a, cur_b = list(best_a), list(best_b)
    while time_budget() > 0:
        # Pick a random state and letter, change its destination
        letter = rng.random() < 0.5
        state = rng.randint(0, n - 1)
        new_dest = rng.randint(0, n - 1)
        if letter:
            old = cur_a[state]
            cur_a[state] = new_dest
            sc = _threshold_of(cur_a, cur_b)
            cur_a[state] = old
        else:
            old = cur_b[state]
            cur_b[state] = new_dest
            sc = _threshold_of(cur_a, cur_b)
            cur_b[state] = old
        if sc > best_score:
            best_score = sc
            if letter:
                cur_a[state] = new_dest
            else:
                cur_b[state] = new_dest
            if best_score >= (n - 1) ** 2:
                break
    return best_score, list(cur_a), list(cur_b)

def construct():
    t0 = time.time()
    deadline = t0 + 82.0
    n = TOTAL_STATES
    target = (n - 1) ** 2  # 121

    def remaining():
        return deadline - time.time()

    best_score = -1
    best_a = None
    best_b = None

    def consider(a, b):
        nonlocal best_score, best_a, best_b
        sc = _threshold_of(a, b)
        if sc > best_score:
            best_score = sc
            best_a = list(a)
            best_b = list(b)
        return sc

    # Phase 1: structured families
    for a, b in _known_tight_candidates():
        if time.time() > deadline - 15.0:
            break
        sc = consider(a, b)
        if sc >= target:
            return [best_a, best_b]

    # Phase 2: local mutation refinement
    if best_a is not None:
        import random as _rnd
        rng = _rnd.Random(137)
        budget_start = time.time()
        while time.time() < deadline - 3.0:
            sc, a2, b2 = _local_mutation_search(best_a, best_b, remaining, rng)
            if sc > best_score:
                best_score = sc
                best_a, best_b = a2, b2
                if best_score >= target:
                    break
            else:
                break

    # Phase 3: broad random scan as safety net
    if time.time() < deadline - 2.0:
        import random as _rnd
        rng2 = _rnd.Random(2024)
        for _ in range(2000):
            if time.time() > deadline - 1.0:
                break
            ra = [rng2.randint(0, n - 1) for _ in range(n)]
            rb = [rng2.randint(0, n - 1) for _ in range(n)]
            rb[0] = rb[1]  # ensure a collision for sync likelihood
            consider(ra, rb)
            if best_score >= target:
                break

    if best_a is None:
        best_a = [(i + 1) % n for i in range(n)]
        best_b = list(range(n))
        best_b[0] = 1

    return [best_a, best_b]
