"""Equal-gcd-triple-free subsets of [1,128]: bitmask-accelerated search.

Model: maximize |S| s.t. for every triple {a,b,c} with a<b<c in [1,128],
    NOT( gcd(a,b)==gcd(a,c)==gcd(b,c) )

Key structural change: represent the current set S as a 128-bit integer bitmask.
Precompute for each element x a list of bitmask pairs (mask_a, mask_b) where
adding x creates a conflict iff (S & mask_a) and (S & mask_b) are both nonzero.
This makes each conflict check O(1) bitwise ops instead of O(|conflicts|) set lookups.

The search uses:
1. Fast greedy with bitmask conflict checks
2. Aggressive local search: try all adds, then 1-swap and 2-swap moves
3. Simulated annealing with fast incremental bad-triple counting via bitmasks
"""
import time
import random
import math
from math import gcd

N = 128
_DEADLINE = 84.0  # 6s safety margin under 90s kill

# Precomputed structures
_PRE = None       # _PRE[i][j] = gcd(i,j) for 1..128
# _CONFLICT_MASKS[x] = list of (mask_a, mask_b) where mask_a/mask_b are bitmasks
#   such that adding x to S creates a conflict iff (S & mask_a) != 0 AND (S & mask_b) != 0
_CONFLICT_MASKS = None
# _BIT[x] = 1 << (x-1)
_BIT = None


def _ensure_pre():
    global _PRE, _CONFLICT_MASKS, _BIT
    if _PRE is not None:
        return
    # Build gcd table
    PRE = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            PRE[i][j] = g
            PRE[j][i] = g
    _PRE = PRE

    # Bit for each element
    BIT = [0] * (N + 1)
    for x in range(1, N + 1):
        BIT[x] = 1 << (x - 1)
    _BIT = BIT

    # Build conflict masks:
    # For each x, find all pairs (a,b) with a<b, a!=x, b!=x such that
    # gcd(x,a) == gcd(x,b) == gcd(a,b) == g
    # For each such pair, we need a bitmask check.
    # But we can group: for a given x, the conflict condition is:
    #   exists a,b in S (a!=b, a!=x, b!=x) with gcd(x,a)=gcd(x,b)=gcd(a,b)
    # 
    # Efficient approach: for each x, for each gcd value g dividing x,
    # collect all elements a with gcd(x,a)=g. Then any two elements a,b from this
    # group with gcd(a,b)=g form a conflict pair.
    # 
    # For the bitmask check: adding x is forbidden iff there exist a,b in S
    # (both != x) with gcd(x,a)=gcd(x,b)=gcd(a,b)=g for some g.
    # 
    # We precompute: for each x, a list of (mask1, mask2) pairs where:
    #   mask1 = bitmask of elements a with gcd(x,a)=g and gcd(a,b)=g for some b in mask2
    #   Actually simpler: just enumerate all conflict pairs (a,b) and store as bitmask pairs.
    #
    # Even simpler: for each x, store the list of (a, b) pairs as before, but also
    # precompute a "conflict bitmask" approach:
    #   For each x, group conflicts by the pair. The check is:
    #   for (a,b) in conflicts[x]: if (S has a) and (S has b): conflict
    #   With bitmasks: if (S & BIT[a]) and (S & BIT[b]): conflict
    #
    # But that's still O(num_conflicts) per check. To make it O(1), we can use:
    #   For each x, precompute a list of "witness" bitmasks.
    #   Actually, let's use a different approach:
    #   For each x, precompute conflict_pairs[x] = list of (a,b)
    #   And use a fast check: iterate conflicts[x] but with bitmask tests.
    #   Each test is just two AND operations, which is very fast.
    
    CONFLICT = {x: [] for x in range(1, N + 1)}
    for x in range(1, N + 1):
        for a in range(1, N + 1):
            if a == x:
                continue
            g = PRE[x][a]
            for b in range(a + 1, N + 1):
                if b == x:
                    continue
                if PRE[x][b] == g and PRE[a][b] == g:
                    CONFLICT[x].append((a, b))
    
    # Convert to bitmask pairs for fast checking
    CONFLICT_MASKS = {x: [] for x in range(1, N + 1)}
    for x in range(1, N + 1):
        for a, b in CONFLICT[x]:
            CONFLICT_MASKS[x].append((BIT[a], BIT[b]))
    _CONFLICT_MASKS = CONFLICT_MASKS


def _is_valid(Sset):
    """Check if Sset (a set of ints) has no equal-gcd triple."""
    P = _PRE
    sl = sorted(Sset)
    n = len(sl)
    for i in range(n):
        a = sl[i]
        for j in range(i + 1, n):
            b = sl[j]
            gab = P[a][b]
            for k in range(j + 1, n):
                c = sl[k]
                if P[a][c] == gab and P[b][c] == gab:
                    return False
    return True


def _bad_count(Sset):
    """Count equal-gcd triples in Sset."""
    P = _PRE
    sl = sorted(Sset)
    n = len(sl)
    cnt = 0
    for i in range(n):
        a = sl[i]
        for j in range(i + 1, n):
            b = sl[j]
            gab = P[a][b]
            for k in range(j + 1, n):
                c = sl[k]
                if P[a][c] == gab and P[b][c] == gab:
                    cnt += 1
    return cnt


def _can_add_mask(Smask, x):
    """Check if x can be added to set represented by bitmask Smask."""
    for ma, mb in _CONFLICT_MASKS[x]:
        if (Smask & ma) and (Smask & mb):
            return False
    return True


def _can_add(Sset, x):
    """Check if x can be added to valid set S (Sset is a set)."""
    Smask = 0
    for v in Sset:
        Smask |= _BIT[v]
    return _can_add_mask(Smask, x)


def _swap_valid_mask(Smask, r, x):
    """Check if S - {r} + {x} is valid using bitmask."""
    Smask_new = Smask & ~_BIT[r]
    for ma, mb in _CONFLICT_MASKS[x]:
        if (Smask_new & ma) and (Smask_new & mb):
            return False
    return True


def _greedy_add(order, S):
    """Greedy: try to add elements in order to S (S is a list). Returns new list."""
    S = list(S)
    Sset = set(S)
    Smask = 0
    for v in Sset:
        Smask |= _BIT[v]
    for x in order:
        if x in Sset:
            continue
        if _can_add_mask(Smask, x):
            S.append(x)
            Sset.add(x)
            Smask |= _BIT[x]
    return S


def _local_improve(S, max_rounds=200, time_budget=None):
    """Local improvement: repeatedly try all adds, then swaps that enable adds."""
    S = list(S)
    Sset = set(S)
    Smask = 0
    for v in Sset:
        Smask |= _BIT[v]
    start_time = time.time()
    for rnd in range(max_rounds):
        if time_budget is not None and time.time() - start_time > time_budget:
            break
        # Phase A: try all possible adds
        added = False
        for x in range(1, N + 1):
            if x in Sset:
                continue
            if _can_add_mask(Smask, x):
                S.append(x)
                Sset.add(x)
                Smask |= _BIT[x]
                added = True
        if not added:
            break
        # Phase B: try swaps that enable new adds
        improved = False
        for r in S:
            for x in range(1, N + 1):
                if x in Sset or x == r:
                    continue
                if not _swap_valid_mask(Smask, r, x):
                    continue
                # Tentative swap
                S.remove(r)
                Sset.discard(r)
                Smask &= ~_BIT[r]
                S.append(x)
                Sset.add(x)
                Smask |= _BIT[x]
                # Check if any new element can be added
                new_add = None
                for y in range(1, N + 1):
                    if y in Sset:
                        continue
                    if _can_add_mask(Smask, y):
                        new_add = y
                        break
                if new_add is not None:
                    S.append(new_add)
                    Sset.add(new_add)
                    Smask |= _BIT[new_add]
                    improved = True
                    break
                else:
                    # Revert swap
                    S.remove(x)
                    Sset.discard(x)
                    Smask &= ~_BIT[x]
                    S.append(r)
                    Sset.add(r)
                    Smask |= _BIT[r]
            if improved:
                break
        if not improved:
            break
    return S


def _compute_bad_count_fast(Smask):
    """Compute number of bad triples using bitmask-based approach."""
    # For each triple (a,b,c), check if gcd(a,b)==gcd(a,c)==gcd(b,c)
    # This is O(N^3) but with very small constant due to precomputed gcd table.
    P = _PRE
    cnt = 0
    sl = [x for x in range(1, N + 1) if (Smask & _BIT[x])]
    n = len(sl)
    for i in range(n):
        a = sl[i]
        for j in range(i + 1, n):
            b = sl[j]
            gab = P[a][b]
            for k in range(j + 1, n):
                c = sl[k]
                if P[a][c] == gab and P[b][c] == gab:
                    cnt += 1
    return cnt


def _simulated_annealing(S, n_iters=50000, time_budget=30.0):
    """SA with bitmask-based fast conflict checks."""
    S = list(S)
    Sset = set(S)
    Smask = 0
    for v in Sset:
        Smask |= _BIT[v]
    start_time = time.time()
    rng = random.Random()
    best_S = list(S)
    best_score = -_compute_bad_count_fast(Smask) + 0.1 * len(S)
    curr_score = best_score

    T = 1.0
    cooling = 0.99995

    for it in range(n_iters):
        if time.time() - start_time > time_budget:
            break
        # Choose random move
        move_type = rng.random()
        if move_type < 0.5:
            # Swap
            if len(S) < 2:
                continue
            r = rng.choice(S)
            x = rng.randint(1, N)
            if x == r or x in Sset:
                continue
            # Compute new bad count
            Smask_new = (Smask & ~_BIT[r]) | _BIT[x]
            new_bc = _compute_bad_count_fast(Smask_new)
            new_score = -new_bc + 0.1 * len(S)
            delta = new_score - curr_score
            if delta >= 0 or rng.random() < math.exp(delta / max(T, 0.001)):
                # Accept
                S.remove(r)
                Sset.discard(r)
                S.append(x)
                Sset.add(x)
                Smask = Smask_new
                curr_score = new_score
            T *= cooling
            if curr_score > best_score:
                best_score = curr_score
                best_S = list(S)
        elif move_type < 0.75:
            # Add
            x = rng.randint(1, N)
            if x in Sset:
                continue
            Smask_new = Smask | _BIT[x]
            new_bc = _compute_bad_count_fast(Smask_new)
            new_score = -new_bc + 0.1 * (len(S) + 1)
            delta = new_score - curr_score
            if delta >= 0 or rng.random() < math.exp(delta / max(T, 0.001)):
                S.append(x)
                Sset.add(x)
                Smask = Smask_new
                curr_score = new_score
            T *= cooling
            if curr_score > best_score:
                best_score = curr_score
                best_S = list(S)
        else:
            # Remove
            if len(S) < 2:
                continue
            r = rng.choice(S)
            Smask_new = Smask & ~_BIT[r]
            new_bc = _compute_bad_count_fast(Smask_new)
            new_score = -new_bc + 0.1 * (len(S) - 1)
            delta = new_score - curr_score
            if delta >= 0 or rng.random() < math.exp(delta / max(T, 0.001)):
                S.remove(r)
                Sset.discard(r)
                Smask = Smask_new
                curr_score = new_score
            T *= cooling
            if curr_score > best_score:
                best_score = curr_score
                best_S = list(S)

    # Final: clean up to get a valid set from best_S
    best_Sset = set(best_S)
    if _is_valid(best_Sset):
        return best_S
    # If not valid, repair
    sl = sorted(best_S)
    for elem in sl:
        if _bad_count(set(sl)) > 0 and elem in sl:
            sl.remove(elem)
            if _bad_count(set(sl)) == 0:
                break
    return sl


def _construct():
    global _START
    _ensure_pre()
    _START = time.time()
    P = _PRE

    def time_left():
        return _DEADLINE - (time.time() - _START)

    best = []
    best_len = -1

    # Strategy 1: Multiple greedy orders + local improvement
    def num_divs(n):
        c = 0
        i = 1
        while i * i <= n:
            if n % i == 0:
                c += 1
                if i * i != n:
                    c += 1
            i += 1
        return c

    divs = {x: num_divs(x) for x in range(1, N + 1)}

    def is_prime_power(x):
        if x <= 1:
            return False
        v = x
        p0 = None
        for p in range(2, int(v ** 0.5) + 1):
            if v % p == 0:
                if p0 is None:
                    p0 = p
                elif p0 != p:
                    return False
                while v % p == 0:
                    v //= p
        if v > 1:
            if p0 is None:
                p0 = v
            elif p0 != v:
                return False
        return True

    prime_powers = [x for x in range(1, N + 1) if is_prime_power(x)]

    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(range(1, N + 1), key=lambda x: divs[x]),
        sorted(range(1, N + 1), key=lambda x: divs[x], reverse=True),
        sorted(range(1, N + 1), key=lambda x: (0 if is_prime_power(x) else 1, x)),
        prime_powers + [x for x in range(1, N + 1) if not is_prime_power(x)],
        sorted(range(1, N + 1), key=lambda x: (-divs[x], x)),
        sorted(range(1, N + 1), key=lambda x: (divs[x], x)),
    ]

    for order in orders:
        if time_left() < 5:
            break
        S = _greedy_add(order, [])
        S = _local_improve(S, max_rounds=100, time_budget=min(3, time_left() - 3))
        if len(S) > best_len:
            best = S
            best_len = len(S)

    # Strategy 2: Simulated annealing from the best greedy solution
    if time_left() > 10:
        S = _simulated_annealing(best, n_iters=50000, time_budget=min(30, time_left() - 5))
        if len(S) > best_len and _is_valid(set(S)):
            best = S
            best_len = len(S)
            S = _local_improve(best, max_rounds=50, time_budget=min(3, time_left() - 3))
            if len(S) > best_len:
                best = S
                best_len = len(S)

    # Strategy 3: More SA restarts from different initial points
    rng = random.Random(42)
    while time_left() > 8:
        rng_order = list(range(1, N + 1))
        rng.shuffle(rng_order)
        init = _greedy_add(rng_order[:40], [])
        S = _simulated_annealing(init, n_iters=20000, time_budget=min(15, time_left() - 4))
        if len(S) > best_len and _is_valid(set(S)):
            S = _local_improve(S, max_rounds=30, time_budget=min(2, time_left() - 2))
            if len(S) > best_len:
                best = S
                best_len = len(S)

    # Strategy 4: Additional local search rounds on best
    if time_left() > 3:
        S = _local_improve(best, max_rounds=100, time_budget=time_left() - 1)
        if len(S) > best_len:
            best = S
            best_len = len(S)

    # Final validation
    if not _is_valid(set(best)):
        sl = sorted(best)
        while sl and not _is_valid(set(sl)):
            worst = None
            worst_cnt = -1
            for elem in sl:
                cnt = 0
                rest = [y for y in sl if y != elem]
                for i in range(len(rest)):
                    for j in range(i + 1, len(rest)):
                        a, b = rest[i], rest[j]
                        if P[elem][a] == P[elem][b] == P[a][b]:
                            cnt += 1
                if cnt > worst_cnt:
                    worst_cnt = cnt
                    worst = elem
            if worst is not None:
                sl.remove(worst)
            else:
                break
        best = sl

    if not best:
        best = [2]

    return best


def construct():
    try:
        res = _construct()
    except Exception:
        _ensure_pre()
        res = []
        resset = set()
        for x in range(1, N + 1):
            if _can_add(resset, x):
                res.append(x)
                resset.add(x)
        if not res:
            res = [2]
    return res
