"""
Stronger algorithm for equal-gcd-triple-free subsets of [1,128].

Key insight: The constraint is that no three distinct elements a,b,c in S
have gcd(a,b) == gcd(a,c) == gcd(b,c).

Strategy:
1. Precompute all pairwise gcds.
2. Use a smarter initial construction: start with numbers that are "structurally
   safe" (prime powers, numbers with unique prime factorizations).
3. Use a fast incremental validity check when adding elements.
4. Use simulated annealing / local search with swaps to maximize cardinality.
5. Multiple random restarts with different orderings.

The key optimization is to maintain for each pair (a,b) in S, the gcd value,
and when considering adding x, check all pairs (a,b) to see if
gcd(x,a) == gcd(x,b) == gcd(a,b).

For speed, we can precompute for each x, the list of elements a where gcd(x,a) = g
for each g. Then when adding x, for each existing pair (a,b), we just check
if gcd(x,a) == gcd(x,b) == gcd(a,b).

Actually the most efficient approach:
- Maintain S as a set.
- To check if x can be added: for each pair (a,b) in S, check if gcd(x,a)==gcd(x,b)==gcd(a,b).
  This is O(n^2) per insertion where n = |S|.

For local search (swap): remove r, add x. Check validity of S \ {r} ∪ {x}.
This requires checking all triples in the new set, which is O(n^3). Too slow.

Better: when swapping r->x, the only new triples are those involving x (and not r).
So we need to check: for all pairs (a,b) in S\{r}, is gcd(x,a)==gcd(x,b)==gcd(a,b)?
That's O(n^2) per swap attempt.

Let's implement this carefully with time management.
"""
import time
import random
from math import gcd
from itertools import combinations

_DEADLINE = 85.0  # safety margin under 90s kill
_START = None
_CAND = list(range(1, 129))
_PRE = None
_PRIMEP = None


def _ensure_pre():
    global _PRE, _PRIMEP
    if _PRE is not None:
        return
    n = 128
    PRE = [[1] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            PRE[i][j] = PRE[j][i] = gcd(i, j)
    _PRE = PRE
    # prime-power flags
    flag = [False] * (n + 1)
    for x in range(2, n + 1):
        v = x
        f0 = None
        ok = True
        p = 2
        while p * p <= v:
            if v % p == 0:
                if f0 is None:
                    f0 = p
                else:
                    ok = False
                    break
                while v % p == 0:
                    v //= p
            p += 1
        if ok and (v == 1 or v == f0):
            flag[x] = True
    _PRIMEP = flag


def _bad_triples_count(S):
    """Count equal-gcd triples in S."""
    P = _PRE
    s = list(S)
    n = len(s)
    cnt = 0
    for i in range(n):
        a = s[i]
        for j in range(i + 1, n):
            b = s[j]
            g_ab = P[a][b]
            for k in range(j + 1, n):
                c = s[k]
                if P[a][c] == g_ab and P[b][c] == g_ab:
                    cnt += 1
    return cnt


def _can_add_fast(Slist, x, P):
    """Check if x can be added to Slist without creating an equal-gcd triple."""
    n = len(Slist)
    for i in range(n):
        a = Slist[i]
        gxa = P[x][a]
        for j in range(i + 1, n):
            b = Slist[j]
            if P[x][b] == gxa and P[a][b] == gxa:
                return False
    return True


def _swap_valid(Slist, remove_idx, x, P):
    """Check if removing Slist[remove_idx] and adding x keeps the set valid.
    Only need to check triples involving x (new element) and pairs from Slist minus the removed one.
    Also need to ensure no triple was "only" invalid because of the removed element - but since
    the original set is valid, removing an element can only make it more valid. So we only
    need to check triples involving x.
    """
    n = len(Slist)
    r = Slist[remove_idx]
    for i in range(n):
        if i == remove_idx:
            continue
        a = Slist[i]
        gxa = P[x][a]
        for j in range(n):
            if j == remove_idx or j <= i:
                continue
            b = Slist[j]
            if P[x][b] == gxa and P[a][b] == gxa:
                return False
    return True


def _greedy_build(order, P, PRIMEP):
    """Greedy build following the given order."""
    S = []
    for x in order:
        if _can_add_fast(S, x, P):
            S.append(x)
    return S


def _local_search(S, P, deadline):
    """Local search: try swaps to improve (but size is fixed in swap).
    Actually for maximizing size, we should try to add elements.
    Let's do: repeatedly try to add any excluded element. If none can be added,
    try swaps that might allow a subsequent add.
    
    Simpler: just do hill-climbing on size via additions, with random restarts.
    """
    S = list(S)
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for x in _CAND:
            if x in S:
                continue
            if _can_add_fast(S, x, P):
                S.append(x)
                improved = True
                break
    return S


def _swap_local_search(S, P, deadline, rng):
    """Try swaps: remove one, add another, then try to add more.
    This can increase size if the swap enables a new addition."""
    S = list(S)
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        n = len(S)
        # Try all swaps
        for i in range(n):
            if time.time() > deadline:
                break
            r = S[i]
            for x in _CAND:
                if x in S or x == r:
                    continue
                if _swap_valid(S, i, x, P):
                    # Perform swap
                    S[i] = x
                    # Now try to add more
                    added = True
                    while added:
                        if time.time() > deadline:
                            break
                        added = False
                        for y in _CAND:
                            if y in S:
                                continue
                            if _can_add_fast(S, y, P):
                                S.append(y)
                                added = True
                                break
                    improved = True
                    break
            if improved:
                break
    return S


def _construct():
    global _START
    _ensure_pre()
    _START = time.time()
    P = _PRE
    PRIMEP = _PRIMEP

    # Build multiple orderings and greedy-build with each
    prime_powers = [x for x in range(1, 129) if PRIMEP[x]]
    
    # Ordering 1: prime powers first, then by number of distinct prime factors
    def num_distinct_pf(x):
        v = x
        cnt = 0
        p = 2
        while p * p <= v:
            if v % p == 0:
                cnt += 1
                while v % p == 0:
                    v //= p
            p += 1
        if v > 1:
            cnt += 1
        return cnt

    order1 = sorted(_CAND, key=lambda x: (0 if PRIMEP[x] else 1, num_distinct_pf(x), x))
    S1 = _greedy_build(order1, P, PRIMEP)

    # Ordering 2: random
    rng = random.Random(42)
    best = S1[:]
    best_size = len(best)

    # Try several random orderings
    deadline_inner = _START + _DEADLINE - 10
    for trial in range(200):
        if time.time() > deadline_inner:
            break
        order = _CAND[:]
        rng.shuffle(order)
        # Bias: put prime powers earlier with some probability
        S = _greedy_build(order, P, PRIMEP)
        if len(S) > best_size:
            best = S[:]
            best_size = len(S)

    # Local search on best
    if time.time() < _START + _DEADLINE - 5:
        best = _local_search(best, P, _START + _DEADLINE - 5)
        best_size = len(best)

    # Swap-based local search
    if time.time() < _START + _DEADLINE - 3:
        best = _swap_local_search(best, P, _START + _DEADLINE - 3, rng)
        best_size = len(best)

    # More random restarts with local search
    if time.time() < _START + _DEADLINE - 2:
        for trial in range(100):
            if time.time() > _START + _DEADLINE - 2:
                break
            order = _CAND[:]
            rng.shuffle(order)
            S = _greedy_build(order, P, PRIMEP)
            S = _local_search(S, P, _START + _DEADLINE - 2)
            if len(S) > best_size:
                best = S[:]
                best_size = len(S)

    # Final validation
    if _bad_triples_count(best) != 0:
        # Fallback: rebuild greedily
        order = sorted(_CAND, key=lambda x: (0 if PRIMEP[x] else 1, num_distinct_pf(x), x))
        best = _greedy_build(order, P, PRIMEP)

    if not best:
        best = [2]

    return best


def construct():
    try:
        res = _construct()
    except Exception:
        # Guaranteed valid fallback
        _ensure_pre()
        P = _PRE
        PRIMEP = _PRIMEP
        S = []
        for x in range(1, 129):
            if _can_add_fast(S, x, P):
                S.append(x)
        if not S:
            S = [2]
        res = S
    return res
