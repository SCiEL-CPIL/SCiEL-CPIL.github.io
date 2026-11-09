# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

N = 192
K = 3
MAXK = 48
DEADLINE = 72.0  # wall-clock seconds; evaluator kills at 90s

def _valid_addition(x, S):
    # x divides sum of any K distinct OTHER selected ints -> violation
    if x in S:
        return False
    for a in S:
        others = [v for v in S if v != a]
        for t in combinations(others, K):
            if sum(t) % a == 0:
                return False
    return True

def _count_violations(S):
    """Return (total_violations, {element: violation_count})."""
    counts = {a: 0 for a in S}
    for a in S:
        others = [v for v in S if v != a]
        for t in combinations(others, K):
            if sum(t) % a == 0:
                counts[a] += 1
    return sum(counts.values()), counts

def _greedy_init(order, k):
    S = []
    for x in order:
        if len(S) >= k:
            break
        if _valid_addition(x, S):
            S.append(x)
    return S

def _greedy_large_first():
    S = _greedy_init(list(range(N, 0, -1)), MAXK)
    return S

def _local_search(S, deadline, rng):
    S = list(S)
    cur_viol, cur_counts = _count_violations(S)
    best_S = list(S)
    best_viol = cur_viol
    best_score = len(best_S)
    universe = set(range(1, N + 1))

    while time.time() < deadline:
        # pick a violated element at random
        viol_elements = [a for a in S if cur_counts[a] > 0]
        if not viol_elements:
            break
        a = rng.choice(viol_elements)
        # candidate replacements: random elements not in S
        cands = [c for c in universe if c not in S]
        if not cands:
            break
        c = rng.choice(cands)
        new_S = [v for v in S if v != a] + [c]
        new_viol, new_counts = _count_violations(new_S)
        # accept if no worse (min-conflicts) with slight random tie-break
        if new_viol <= cur_viol:
            S = new_S
            cur_viol, cur_counts = new_viol, new_counts
            if cur_viol == 0:
                # valid set; try to grow it
                if len(S) < MAXK:
                    S = _greedy_init([c2 for c2 in range(1, N + 1) if c2 not in S], MAXK - len(S))
                    cur_viol, cur_counts = _count_violations(S)
                    if cur_viol == 0:
                        if len(S) > best_score:
                            best_score = len(S)
                            best_S = list(S)
                        return list(S)
            if len(S) > best_score:
                best_score = len(S)
                best_S = list(S)
        # occasional swap between two elements to escape
        if rng.random() < 0.3 and len(S) >= 2:
            i = rng.randrange(len(S))
            j = rng.randrange(len(S))
            if i != j:
                c = rng.choice([c2 for c2 in universe if c2 not in S] or [S[i]])
                new_S = list(S)
                new_S[i] = c
                new_viol, new_counts = _count_violations(new_S)
                if new_viol <= cur_viol:
                    S = new_S
                    cur_viol, cur_counts = new_viol, new_counts
                    if len(S) > best_score:
                        best_score = len(S)
                        best_S = list(S)

    return best_S if best_viol == 0 else best_S

def construct():
    start = time.time()
    deadline = start + DEADLINE
    rng = random.Random(12345)

    # baseline greedy
    S0 = _greedy_large_first()
    best = S0
    best_viol, _ = _count_violations(S0)
    best_score = len(best)

    # try to grow baseline
    if best_viol == 0 and best_score < MAXK:
        grown = _greedy_init([c for c in range(1, N + 1) if c not in set(best)], MAXK - best_score)
        if grown:
            cand = list(best) + grown
            v, _ = _count_violations(cand)
            if v == 0 and len(cand) > best_score:
                best = cand
                best_score = len(best)

    # local search restarts
    attempts = 0
    while time.time() < deadline - 2.0:
        attempts += 1
        # random subset init of size near target
        size = rng.randint(30, MAXK)
        init = rng.sample(range(1, N + 1), size)
        res = _local_search(init, deadline - 1.0, rng)
        v, _ = _count_violations(res)
        if v == 0 and len(res) > best_score:
            best = res
            best_score = len(best)
        # also try from greedy seed
        if attempts % 3 == 0:
            res2 = _local_search(list(S0), deadline - 1.0, rng)
            v2, _ = _count_violations(res2)
            if v2 == 0 and len(res2) > best_score:
                best = res2
                best_score = len(best)

    # final: if best invalid, fall back to largest valid greedy we found
    v_best, _ = _count_violations(best)
    if v_best != 0:
        # rebuild a guaranteed-valid set
        fb = _greedy_large_first()
        if len(fb) >= len(best):
            best = fb

    return best
# EVOLVE-BLOCK-END
