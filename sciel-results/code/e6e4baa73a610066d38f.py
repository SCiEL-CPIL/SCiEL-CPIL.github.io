"""Suite task: maximize size of a set in 2..256 with no a | b*c for distinct a,b,c."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import combinations

P = {'n': 256}


def _prime_factors(n):
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors


def _precompute(maxn):
    pf = [set() for _ in range(maxn + 1)]
    for i in range(2, maxn + 1):
        pf[i] = _prime_factors(i)
    return pf


def _valid_add(pf, S, x):
    # Check that no existing a in S divides x*b for any other b in S (b != x, b != a)
    fx = pf[x]
    for a in S:
        if a == x:
            continue
        fa = pf[a]
        for b in S:
            if b == x or b == a:
                continue
            # a | x*b  <=>  every prime factor of a divides x*b
            fb = pf[b]
            ok = True
            for p in fa:
                if p not in fx and p not in fb:
                    ok = False
                    break
            if ok:
                return False
    return True


def _build_greedy(pf, maxn):
    # Try several orderings, keep the best resulting set
    best = None
    orderings = [
        list(range(2, maxn + 1)),
        list(range(maxn, 1, -1)),
        sorted(range(2, maxn + 1), key=lambda v: (len(pf[v]), v)),
        sorted(range(2, maxn + 1), key=lambda v: (len(pf[v]), -v)),
    ]
    for order in orderings:
        S = []
        for x in order:
            if _valid_add(pf, S, x):
                S.append(x)
        if best is None or len(S) > len(best):
            best = S
    return best


def _build_conflict_matrix(pf, maxn):
    """Precompute conflict[x][a] = set of b values such that a | x*b (with x != a, x != b, a != b).
    For a given set S, x can be added iff for all a in S\{x}, there exists b in S\{x,a} with a NOT dividing x*b.
    Equivalently: x is incompatible with a (given S) if ALL b in S\{x,a} have a | x*b.
    """
    # conflict[x][a] = frozenset of b such that a divides x*b
    # We need: for x to be addable to S, for each a in S (a != x), NOT(all b in S\{x,a}: a|x*b)
    # i.e., for each a in S, there must exist some b in S\{x,a} such that a does NOT divide x*b
    
    # Precompute: for each pair (x, a) with x != a, the set of b (b != x, b != a) where a | x*b
    conflict = [dict() for _ in range(maxn + 1)]
    for x in range(2, maxn + 1):
        fx = pf[x]
        for a in range(2, maxn + 1):
            if a == x:
                continue
            fa = pf[a]
            bad_bs = set()
            for b in range(2, maxn + 1):
                if b == x or b == a:
                    continue
                fb = pf[b]
                ok = True
                for p in fa:
                    if p not in fx and p not in fb:
                        ok = False
                        break
                if ok:
                    bad_bs.add(b)
            conflict[x][a] = bad_bs
    return conflict


def _can_add_bitset(conflict, S_set, S_list, x):
    """Check if x can be added to S. Uses precomputed conflict matrix."""
    for a in S_list:
        if a == x:
            continue
        bad_bs = conflict[x][a]
        # Need: exists b in S\{x,a} such that b NOT in bad_bs
        found_good = False
        for b in S_list:
            if b == x or b == a:
                continue
            if b not in bad_bs:
                found_good = True
                break
        if not found_good:
            return False
    return True


def _swap_search_bitset(conflict, S_list, candidates, deadline, max_iters):
    """Local search using precomputed conflict matrix for speed."""
    S = list(S_list)
    S_set = set(S)
    cand = [c for c in candidates if c not in S_set]
    cand.sort()
    improved = True
    it = 0
    while improved and it < max_iters and time.time() < deadline:
        improved = False
        it += 1
        for c in cand:
            if time.time() > deadline:
                break
            if c in S_set:
                continue
            # Try adding c directly
            if _can_add_bitset(conflict, S_set, S, c):
                S.append(c)
                S_set.add(c)
                improved = True
                continue
            # Try 1-1 swap: remove r, add c
            for r in S:
                if time.time() > deadline:
                    return S, S_set
                newS = [y for y in S if y != r]
                newS_set = S_set - {r}
                newS.append(c)
                newS_set.add(c)
                if _can_add_bitset(conflict, newS_set, newS, c):
                    S = newS
                    S_set = newS_set
                    improved = True
                    break
    return S, S_set


def _two_swap_search(conflict, S_list, candidates, deadline, max_iters):
    """Try 2-2 exchanges: remove two elements, add two new ones."""
    S = list(S_list)
    S_set = set(S)
    cand = [c for c in candidates if c not in S_set]
    cand.sort()
    improved = True
    it = 0
    while improved and it < max_iters and time.time() < deadline:
        improved = False
        it += 1
        # Try pairs of candidates
        for i in range(len(cand)):
            if time.time() > deadline:
                break
            c1 = cand[i]
            if c1 in S_set:
                continue
            for j in range(i + 1, len(cand)):
                if time.time() > deadline:
                    break
                c2 = cand[j]
                if c2 in S_set:
                    continue
                # Try removing two elements r1, r2 and adding c1, c2
                for r1_idx in range(len(S)):
                    if time.time() > deadline:
                        return S, S_set
                    r1 = S[r1_idx]
                    for r2_idx in range(r1_idx + 1, len(S)):
                        if time.time() > deadline:
                            return S, S_set
                        r2 = S[r2_idx]
                        newS = [y for y in S if y != r1 and y != r2]
                        newS_set = S_set - {r1, r2}
                        newS.append(c1)
                        newS.append(c2)
                        newS_set.add(c1)
                        newS_set.add(c2)
                        # Check validity
                        valid = True
                        for x in newS:
                            if not _can_add_bitset(conflict, newS_set - {x}, newS, x):
                                valid = False
                                break
                        if valid:
                            S = newS
                            S_set = newS_set
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
    return S, S_set


def construct():
    t0 = time.time()
    deadline = t0 + 50.0  # safety margin vs 60s/90s limits
    maxn = P['n']
    pf = _precompute(maxn)

    # Build conflict matrix
    conflict = _build_conflict_matrix(pf, maxn)

    # Build initial set with greedy
    best = None
    orderings = [
        list(range(2, maxn + 1)),
        list(range(maxn, 1, -1)),
        sorted(range(2, maxn + 1), key=lambda v: (len(pf[v]), v)),
        sorted(range(2, maxn + 1), key=lambda v: (len(pf[v]), -v)),
    ]
    for order in orderings:
        S = []
        S_set = set()
        for x in order:
            if _can_add_bitset(conflict, S_set, S, x):
                S.append(x)
                S_set.add(x)
        if best is None or len(S) > len(best):
            best = S

    S = list(best)
    S_set = set(S)

    # Add-only pass
    if time.time() < deadline:
        S, S_set = _swap_search_bitset(conflict, S, list(range(2, maxn + 1)), deadline, max_iters=50)

    # More swap rounds with different orderings
    if time.time() < deadline - 5.0:
        for _ in range(3):
            if time.time() > deadline - 5.0:
                break
            S, S_set = _swap_search_bitset(conflict, S, list(range(maxn, 1, -1)), deadline, max_iters=20)

    # Try 2-2 exchanges if time remains
    if time.time() < deadline - 10.0:
        S, S_set = _two_swap_search(conflict, S, list(range(2, maxn + 1)), deadline, max_iters=10)

    # Final add-only pass
    if time.time() < deadline - 2.0:
        S, S_set = _swap_search_bitset(conflict, S, list(range(2, maxn + 1)), deadline - 2.0, max_iters=20)

    # Ensure Python ints in a list
    return [int(x) for x in S]
# EVOLVE-BLOCK-END
