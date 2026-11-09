"""Rewritten program: graph-based maximum sum-free subset search with local polish."""
# EVOLVE-BLOCK-START
import time
import random
import math

_DEADLINE = 120.0
_SAFETY = 12.0  # return this many seconds before the hard limit

def _instance(size, seed, limit=10000):
    rng = random.Random(seed)
    values = set()
    while len(values) < size:
        x = rng.randint(-limit, limit)
        if x != 0:
            values.add(x)
    return sorted(values)

def _build_conflicts(S):
    """Return (adj, self_conflict) where adj[i] is set of j with S[i]+S[j] in S,
    and self_conflict[i] is True if 2*S[i] in S. We treat a vertex as 'bad' if it
    has a self conflict OR forms an edge. Actually an element c is invalid only if
    there exist a,b in set with a+b=c. So c is 'forbidden' if c = a+b for some a,b
    in the current set (a may equal b). We model via: a vertex is 'used' if it is
    the sum of two chosen vertices.
    We'll track for each element whether it is a sum of two chosen elements.
    """
    S_set = set(S)
    n = len(S)
    # For each element value, precompute pairs (i,j) with i<=j such that S[i]+S[j]==value
    # and value in S.
    # conflict_as_sum: for each value v in S, list of (i,j) pairs with S[i]+S[j]=v
    conflict_as_sum = {v: [] for v in S}
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in S_set:
                conflict_as_sum[s].append((i, j))
    return S, S_set, conflict_as_sum

def _is_sum_free(S_set, chosen_set):
    """Check sum-free: no a+b=c with all in chosen_set."""
    for a in chosen_set:
        for b in chosen_set:
            if a + b in chosen_set:
                return False
    return True

def _greedy(S, S_set, conflict_as_sum, order):
    """Greedy: iterate in `order` (list of indices), add element if it doesn't
    create a violation. A violation is created if adding element x makes some c in
    chosen become a sum of two chosen, OR x itself becomes a sum of two chosen.
    To keep it fast, we check: x is forbidden if x = a+b for some a,b in chosen
    (x is the sum). Also adding x could make an existing c = a+x or x+b.
    We maintain chosen as set.
    """
    chosen = set()
    for idx in order:
        x = S[idx]
        # x must not be a sum of two already-chosen elements
        bad = False
        for a in chosen:
            if x - a in chosen:
                bad = True
                break
        if bad:
            continue
        # adding x must not make any existing c a sum: c = a + x for some a in chosen
        # i.e. c - x in chosen for some c in chosen
        for a in chosen:
            if a + x in chosen:
                bad = True
                break
        if bad:
            continue
        chosen.add(x)
    return chosen

def _polish(S, S_set, conflict_as_sum, chosen, rng, time_budget):
    """Remove-and-retry + simulated annealing style local search.
    Returns improved chosen set.
    """
    chosen = set(chosen)
    best = chosen
    best_size = len(chosen)
    n = len(S)
    S_list = S
    deadline = time.time() + time_budget
    # Precompute for each value its "sum-of-two" pairs among ALL elements
    # so we can quickly test if a value is a sum of two chosen.
    # sum_pairs[v] = list of (i,j) with S[i]+S[j]==v
    sum_pairs = {v: [] for v in S}
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in S_set:
                sum_pairs[s].append((i, j))
    # index map value->index
    idx_of = {v: i for i, v in enumerate(S_list)}

    def is_sum_free_set(cs):
        for v in cs:
            for (i, j) in sum_pairs[v]:
                if S_list[i] in cs and S_list[j] in cs:
                    return False
        return True

    # verify initial
    if not is_sum_free_set(chosen):
        # fallback: rebuild greedily
        return _greedy(S, S_set, conflict_as_sum, list(range(n)))

    # Try removals: for each element in chosen, try removing it, then re-add as many
    # as possible.
    def try_remove_readd(chosen):
        # pick a random element to remove
        if not chosen:
            return chosen
        removed = rng.choice(list(chosen))
        new_chosen = set(chosen)
        new_chosen.discard(removed)
        # try to add back removed and other elements greedily
        candidates = [removed] + [v for v in S_list if v not in new_chosen]
        rng.shuffle(candidates)
        for v in candidates:
            if v in new_chosen:
                continue
            # check adding v keeps sum-free
            ok = True
            for (i, j) in sum_pairs[v]:
                if S_list[i] in new_chosen and S_list[j] in new_chosen:
                    ok = False
                    break
            if ok:
                for a in new_chosen:
                    if a + v in new_chosen:
                        ok = False
                        break
            if ok:
                new_chosen.add(v)
        return new_chosen

    iters = 0
    while time.time() < deadline:
        iters += 1
        cand = try_remove_readd(chosen)
        if len(cand) > len(chosen):
            chosen = cand
            if len(chosen) > best_size:
                best = set(chosen)
                best_size = len(chosen)
        elif len(cand) == len(chosen) and rng.random() < 0.3:
            # occasional neutral move to escape plateaus
            chosen = cand
        if iters % 500 == 0 and time.time() > deadline - 1.0:
            break
    return best

def _solve_instance(S, time_budget):
    S_set = set(S)
    n = len(S)
    _, _, conflict_as_sum = _build_conflicts(S)
    rng = random.Random(12345)
    best = set()
    best_size = 0
    deadline = time.time() + time_budget
    # Multiple diverse orderings
    orderings = []
    # order by value, by abs value, by index, random shuffles
    orderings.append(sorted(range(n), key=lambda i: S[i]))
    orderings.append(sorted(range(n), key=lambda i: abs(S[i])))
    orderings.append(list(range(n)))
    for _ in range(6):
        o = list(range(n))
        rng.shuffle(o)
        orderings.append(o)
    for o in orderings:
        if time.time() > deadline - 5.0:
            break
        cand = _greedy(S, S_set, conflict_as_sum, o)
        if len(cand) > best_size:
            best = cand
            best_size = len(cand)
    # Polish the best
    if time.time() < deadline - 2.0:
        polished = _polish(S, S_set, conflict_as_sum, best, rng, max(0.5, deadline - time.time() - 2.0))
        if len(polished) > best_size:
            best = polished
            best_size = len(polished)
    # Final safety: ensure sum-free
    if not _is_sum_free(S_set, best):
        # rebuild greedily as a safe fallback
        best = _greedy(S, S_set, conflict_as_sum, list(range(n)))
    return sorted(best)

def construct():
    start = time.time()
    remaining = _DEADLINE - _SAFETY
    if remaining <= 0:
        remaining = 1.0
    # Allocate time per instance: bigger instances get more time
    specs = [(60, 801), (120, 802), (240, 803)]
    weights = [1.0, 1.6, 2.5]
    total_w = sum(weights)
    budgets = [remaining * w / total_w for w in weights]
    results = []
    for (size, seed), tb in zip(specs, budgets):
        S = _instance(size, seed)
        # leave a little margin between instances
        sub = max(0.5, tb - 1.0)
        res = _solve_instance(S, sub)
        results.append(res)
        if time.time() > _DEADLINE - _SAFETY:
            # not enough time left; finish remaining with a fast greedy
            for (size2, seed2) in specs[len(results):]:
                S2 = _instance(size2, seed2)
                _, _, caf = _build_conflicts(S2)
                res2 = _greedy(S2, set(S2), caf, list(range(len(S2))))
                results.append(res2)
            break
    return results
# EVOLVE-BLOCK-END
