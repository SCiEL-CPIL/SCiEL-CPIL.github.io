"""Maximum independent set in 3-uniform hypergraph via branch-and-bound."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

_TIME_LIMIT = 84.0

_N = 128
_gcd_matrix = [[0] * (_N + 1) for _ in range(_N + 1)]
for _i in range(1, _N + 1):
    for _j in range(_i, _N + 1):
        _g = gcd(_i, _j)
        _gcd_matrix[_i][_j] = _g
        _gcd_matrix[_j][_i] = _g


def _can_add(S, x, gm):
    """Check if x can be added to S without creating a bad triple."""
    n = len(S)
    if n < 2:
        return True
    for i in range(n):
        a = S[i]
        ga = gm[a][x]
        for j in range(i + 1, n):
            b = S[j]
            if gm[b][x] == ga and gm[a][b] == ga:
                return False
    return True


def _greedy(order, deadline):
    """Greedy construction following given order."""
    S = []
    Sset = set()
    gm = _gcd_matrix
    for x in order:
        if time.time() > deadline:
            break
        if x in Sset:
            continue
        if _can_add(S, x, gm):
            S.append(x)
            Sset.add(x)
    return S


def _greedy_fill(S, Sset, universe, deadline):
    """Greedily add all possible elements."""
    improved = True
    gm = _gcd_matrix
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for x in universe:
            if time.time() > deadline:
                break
            if x in Sset:
                continue
            if _can_add(S, x, gm):
                S.append(x)
                Sset.add(x)
                improved = True
    return S, Sset


def _branch_and_bound(remaining, chosen, deadline, best_size):
    """
    Branch and bound to find maximum independent set.
    remaining: sorted list of candidates
    chosen: set of already chosen elements
    deadline: time limit
    best_size: current best size found
    Returns: list of chosen elements if improved, else None
    """
    gm = _gcd_matrix
    chosen = list(chosen)
    
    # Upper bound: |chosen| + |remaining|
    if len(chosen) + len(remaining) <= best_size:
        return None
    
    # If no remaining, return chosen
    if not remaining:
        return chosen if len(chosen) > best_size else None
    
    # Try to greedily fill from remaining to get a quick solution
    trial = chosen[:]
    trial_set = set(chosen)
    gm_local = gm
    for x in remaining:
        if time.time() > deadline:
            break
        if x in trial_set:
            continue
        if _can_add(trial, x, gm_local):
            trial.append(x)
            trial_set.add(x)
    
    if len(trial) > best_size:
        best_size = len(trial)
    
    # Branch: pick the first element in remaining
    x = remaining[0]
    rest = remaining[1:]
    
    # Option 1: include x
    if _can_add(chosen, x, gm):
        result = _branch_and_bound(rest, chosen + [x], deadline, best_size)
        if result and len(result) > best_size:
            return result
    
    # Option 2: exclude x
    result = _branch_and_bound(rest, chosen, deadline, best_size)
    if result and len(result) > best_size:
        return result
    
    return None


def _local_search(S, deadline, rng):
    """Local search with multiple strategies."""
    S = list(S)
    Sset = set(S)
    universe = list(range(1, 129))
    d = deadline - 3
    gm = _gcd_matrix
    
    # Phase 1: greedy fill
    S, Sset = _greedy_fill(S, Sset, universe, d)
    
    # Phase 2: 1-swap search
    if time.time() < d - 5:
        for _iter in range(50):
            if time.time() > d - 5:
                break
            progress = False
            for idx in range(len(S)):
                if time.time() > d - 5:
                    break
                removed = S[idx]
                S2 = S[:idx] + S[idx + 1:]
                S2set = set(S2)
                best_gain = 0
                best_x = None
                best_added = []
                for x in universe:
                    if x in S2set:
                        continue
                    if _can_add(S2, x, gm):
                        S3 = S2 + [x]
                        S3set = set(S3)
                        added = []
                        for y in universe:
                            if y in S3set:
                                continue
                            if _can_add(S3, y, gm):
                                S3.append(y)
                                S3set.add(y)
                                added.append(y)
                        gain = len(added)
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                            best_added = added
                            if gain >= 2:
                                break
                if best_x is not None and best_gain > 0:
                    S = S2 + [best_x]
                    Sset = set(S)
                    for y in best_added:
                        S.append(y)
                        Sset.add(y)
                    progress = True
                    break
            if not progress:
                break
    
    # Phase 3: greedy fill again
    S, Sset = _greedy_fill(S, Sset, universe, d)
    
    # Phase 4: 2-swap search
    if time.time() < d - 4:
        for _iter in range(20):
            if time.time() > d - 4:
                break
            progress = False
            n = len(S)
            for i in range(n):
                if time.time() > d - 4:
                    break
                for j in range(i + 1, n):
                    if time.time() > d - 4:
                        break
                    S2 = S[:i] + S[i+1:j] + S[j+1:]
                    S2set = set(S2)
                    best_gain = 0
                    best_x = None
                    best_added = []
                    for x in universe:
                        if time.time() > d - 4:
                            break
                        if x in S2set:
                            continue
                        if _can_add(S2, x, gm):
                            S3 = S2 + [x]
                            S3set = set(S3)
                            added = []
                            for y in universe:
                                if y in S3set:
                                    continue
                                if _can_add(S3, y, gm):
                                    S3.append(y)
                                    S3set.add(y)
                                    added.append(y)
                                    if len(added) >= 2:
                                        break
                            gain = len(added)
                            if gain > best_gain:
                                best_gain = gain
                                best_x = x
                                best_added = added
                                if gain >= 2:
                                    break
                    if best_x is not None and best_gain > 0:
                        S = S2 + [best_x]
                        Sset = set(S)
                        for y in best_added:
                            S.append(y)
                            Sset.add(y)
                        progress = True
                        break
                if progress:
                    break
            if not progress:
                break
    
    # Phase 5: greedy fill
    S, Sset = _greedy_fill(S, Sset, universe, d)
    
    # Phase 6: Kick strategy
    if time.time() < d - 3:
        for _kick in range(15):
            if time.time() > d - 3:
                break
            S_kick = S[:]
            S_kick_set = set(S_kick)
            n_remove = 1 if len(S) > 40 else 1
            if len(S) > 50 and rng.random() < 0.5:
                n_remove = 2
            to_remove = rng.sample(range(len(S_kick)), n_remove)
            for idx in sorted(to_remove, reverse=True):
                S_kick_set.discard(S_kick[idx])
                S_kick.pop(idx)
            S_kick, S_kick_set = _greedy_fill(S_kick, S_kick_set, universe, d)
            if len(S_kick) > len(S):
                S = S_kick
                Sset = S_kick_set
    
    # Phase 7: Final polish - try removing each element and refilling
    if time.time() < d - 2:
        polish_improved = True
        while polish_improved:
            if time.time() > d - 2:
                break
            polish_improved = False
            for idx in range(len(S)):
                if time.time() > d - 2:
                    break
                S2 = S[:idx] + S[idx + 1:]
                S2set = set(S2)
                S2, S2set = _greedy_fill(S2, S2set, universe, d)
                if len(S2) > len(S):
                    S = S2
                    Sset = S2set
                    polish_improved = True
                    break
    
    # Phase 8: Final greedy fill
    S, Sset = _greedy_fill(S, Sset, universe, d)
    
    return S


def _multi_greedy_local(deadline):
    """Multi-start greedy with local search."""
    rng = random.Random(42)
    universe = list(range(1, 129))
    best = []
    gm = _gcd_matrix
    
    # Generate diverse orderings
    orders = [
        list(range(1, 129)),
        list(range(128, 0, -1)),
        [x for x in range(1, 129) if x % 2 == 1] + [x for x in range(1, 129) if x % 2 == 0],
        [x for x in range(1, 129) if x % 2 == 0] + [x for x in range(1, 129) if x % 2 == 1],
        sorted(range(1, 129), key=lambda x: -x.bit_count()),
        sorted(range(1, 129), key=lambda x: x.bit_count()),
        [x for x in range(1, 129) if x % 3 != 0] + [x for x in range(1, 129) if x % 3 == 0],
        [x for x in range(1, 129) if x % 3 == 0] + [x for x in range(1, 129) if x % 3 != 0],
        sorted(range(1, 129), key=lambda x: (x % 2, x)),
        sorted(range(1, 129), key=lambda x: (-x.bit_count(), x)),
    ]
    
    # Add random orderings
    for _ in range(25):
        o = universe[:]
        rng.shuffle(o)
        orders.append(o)
    
    # Run greedy on each ordering
    candidates = []
    for order in orders:
        if time.time() > deadline - 25:
            break
        cand = _greedy(order, deadline)
        candidates.append(cand)
        if len(cand) > len(best):
            best = cand
    
    # Local search on top candidates
    candidates.sort(key=len, reverse=True)
    for cand in candidates[:12]:
        if time.time() > deadline - 8:
            break
        improved = _local_search(cand, deadline, rng)
        if len(improved) > len(best):
            best = improved
    
    # Additional local search passes with different seeds
    if time.time() < deadline - 5:
        for seed in [123, 456, 789, 101, 202]:
            if time.time() > deadline - 5:
                break
            rng2 = random.Random(seed)
            improved = _local_search(best[:], deadline, rng2)
            if len(improved) > len(best):
                best = improved
    
    return best


def construct():
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT
    try:
        result = _multi_greedy_local(_DEADLINE)
    except Exception:
        # Fallback: simple ascending greedy
        result = []
        for x in range(1, 129):
            ok = True
            for i in range(len(result)):
                a = result[i]
                ga = _gcd_matrix[a][x]
                for j in range(i + 1, len(result)):
                    b = result[j]
                    if ga == _gcd_matrix[b][x] == _gcd_matrix[a][b]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                result.append(x)
    if not result:
        result = [1]
    return result
# EVOLVE-BLOCK-END
