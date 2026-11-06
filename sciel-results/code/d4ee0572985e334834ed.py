"""Rewritten program for the reciprocal-subsum-free set task.

Direct approach:
  * Build candidate subsets of [1,24] with explicit structural heuristics
    (geometric "superincreasing reciprocal" sequences, top-half anchors,
    greedy-add in several orders, and greedy-removal from the full set).
  * Validate each candidate EXACTLY with rational arithmetic (Fraction) and
    a meet-in-the-middle subset-sum test, so no floating-point false positives.
  * Keep the largest valid set found, with a wall-clock deadline guard.
"""
import time
import random
import math
from fractions import Fraction
from math import gcd
from functools import reduce

DEADLINE = 85.0  # safety margin under the 90 s wall-clock kill
_START = None

def _deadline_hit():
    global _START
    if _START is None:
        return False
    return (time.time() - _START) > DEADLINE


def _subset_sum_possible(vals, target):
    """True iff some subset of `vals` (Fractions) sums exactly to `target`.
    Meet-in-the-middle, O(2^(n/2))."""
    n = len(vals)
    if n == 0:
        return target == Fraction(0)
    if target == Fraction(0):
        return False  # non-empty subset required? No, empty subset sums to 0
    # Actually the problem says "subset of the OTHER selected elements"
    # A subset can be empty, but sum of empty set is 0, and 1/v > 0, so
    # target > 0 means we need a non-empty subset.
    
    mid = n // 2
    left, right = vals[:mid], vals[mid:]

    L = {Fraction(0)}
    for v in left:
        L = L | {x + v for x in L}
    R = {Fraction(0)}
    for v in right:
        R = R | {x + v for x in R}

    # check whether target = l + r for some l in L, r in R
    if len(L) <= len(R):
        for l in L:
            if (target - l) in R:
                return True
    else:
        for r in R:
            if (target - r) in L:
                return True
    return False


def is_valid(S):
    """Exact reciprocal-subsum-free check for a nonempty set S of ints."""
    S = list(set(S))
    if not S:
        return False
    r = {v: Fraction(1, v) for v in S}
    for a in S:
        others = [r[v] for v in S if v != a]
        if _subset_sum_possible(others, r[a]):
            return False
    return True


def is_valid_fast(S, cache=None):
    """Faster validation using integer arithmetic.
    Multiply all reciprocals by LCM of elements to work with integers."""
    S = list(set(S))
    if not S:
        return False
    if len(S) <= 1:
        return True
    
    # Compute LCM of all elements
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, S)
    # Convert reciprocals to integers: 1/v * L = L/v
    int_vals = {v: L // v for v in S}
    
    for a in S:
        target = int_vals[a]
        others = [int_vals[v] for v in S if v != a]
        if _subset_sum_possible_int(others, target):
            return False
    return True


def _subset_sum_possible_int(vals, target):
    """True iff some subset of integer `vals` sums exactly to `target`.
    Meet-in-the-middle, O(2^(n/2))."""
    n = len(vals)
    if n == 0:
        return target == 0
    if target <= 0:
        return target == 0  # only empty subset can sum to 0
    
    mid = n // 2
    left, right = vals[:mid], vals[mid:]

    L = {0}
    for v in left:
        L = L | {x + v for x in L}
    R = {0}
    for v in right:
        R = R | {x + v for x in R}

    if len(L) <= len(R):
        for l in L:
            if (target - l) in R:
                return True
    else:
        for r in R:
            if (target - r) in L:
                return True
    return False


def _precompute_conflicts():
    """Precompute all conflict constraints: for each a in [1,24], find all
    subsets T of [1,24] \\ {a} such that sum_{t in T} 1/t = 1/a.
    Each conflict is stored as (a, frozenset(T))."""
    conflicts = []
    for a in range(1, 25):
        # LCM of 1..24
        L = 5354228880  # lcm(1..24)
        target = L // a
        others = list(range(1, 25))
        others.remove(a)
        int_vals = [L // v for v in others]
        n = len(int_vals)
        mid = n // 2
        left, right = int_vals[:mid], int_vals[mid:]
        
        Lsums = {0: (0, frozenset())}
        for i, v in enumerate(left):
            new_Lsums = {}
            for s, (mask, fs) in Lsums.items():
                if s + v <= target:
                    new_Lsums[s + v] = (mask | (1 << i), fs | frozenset([others[i]]))
            Lsums.update(new_Lsums)
        
        Rsums = {0: (0, frozenset())}
        for i, v in enumerate(right):
            new_Rsums = {}
            for s, (mask, fs) in Rsums.items():
                if s + v <= target:
                    new_Rsums[s + v] = (mask | (1 << i), fs | frozenset([others[mid + i]]))
            Rsums.update(new_Rsums)
        
        for ls, (lmask, lfs) in Lsums.items():
            need = target - ls
            if need in Rsums:
                rms, rfs = Rsums[need]
                T = lfs | rfs
                if T:  # non-empty subset
                    conflicts.append((a, T))
    return conflicts


_CONFLICTS = None

def _get_conflicts():
    global _CONFLICTS
    if _CONFLICTS is None:
        _CONFLICTS = _precompute_conflicts()
    return _CONFLICTS


def _count_conflicts(selected):
    """Count active conflict constraints for a given selected set."""
    selected_set = set(selected)
    cnt = 0
    for a, T in _get_conflicts():
        if a in selected_set and T.issubset(selected_set):
            cnt += 1
    return cnt


# ---- candidate generators ------------------------------------------------

def cand_seed():
    return list(range(13, 25))


def cand_geometric(base, ratio, lo=1, hi=24):
    out, x = [], float(base)
    while x <= hi:
        v = int(round(x))
        if v < lo:
            v = lo
        if v not in out:
            out.append(v)
        x *= ratio
    return out


def cand_top_half(start):
    return list(range(start, 25))


def greedy_add(order, best_size):
    """Grow a set by adding elements in `order`, keeping validity."""
    chosen = []
    for v in order:
        if v in chosen:
            continue
        if len(chosen) <= best_size:
            continue
        trial = chosen + [v]
        # Quick necessary check: 1/v must not be a subset sum of current recips
        if len(chosen) > 0:
            S = chosen + [v]
            if not is_valid_fast(S):
                continue
        else:
            chosen.append(v)
            continue
        chosen = trial
    return chosen


def local_search(S, max_steps=500):
    """Try to improve set S by single additions and swaps."""
    S = list(set(S))
    if not S:
        return S
    
    all_elems = set(range(1, 25))
    cur = set(S)
    best = set(S)
    
    for step in range(max_steps):
        if _deadline_hit():
            break
        
        improved = False
        
        # Try adding an element
        for v in sorted(all_elems - cur, reverse=True):
            if len(cur) >= len(best):
                continue
            trial = cur | {v}
            if is_valid_fast(list(trial)):
                cur = trial
                best = set(trial)
                improved = True
                break
        
        if improved:
            continue
        
        # Try swaps: remove one, add one
        if len(cur) > 0:
            for rem in sorted(cur, key=lambda x: -x):  # try removing larger first
                for add in sorted(all_elems - cur, reverse=True):
                    trial = (cur - {rem}) | {add}
                    if len(trial) > len(best):
                        if is_valid_fast(list(trial)):
                            cur = trial
                            best = set(trial)
                            improved = True
                            break
                if improved:
                    break
        
        if not improved:
            break
    
    return list(best)


def greedy_remove(best_size):
    """Start from [1,24]; remove elements until valid, prefer removing the
    element that 'causes' the most violations (heuristic)."""
    cur = set(range(1, 25))
    
    while not is_valid_fast(list(cur)):
        if _deadline_hit():
            break
        
        # Find which element to remove: try removing each and see if it helps
        # Use a heuristic: remove the element whose reciprocal is most "easy" to form
        # i.e., the one with the smallest reciprocal (largest element)
        # Actually, let's try removing each element and pick the one that
        # results in the fewest remaining violations
        
        best_el = None
        best_count = len(cur) + 1
        
        for v in sorted(cur, reverse=True):  # try removing large first
            trial = cur - {v}
            if len(trial) <= len(best_size) if isinstance(best_size, list) else len(trial) <= best_size:
                continue
            # Count violations in trial
            cnt = 0
            S = list(trial)
            r = {x: Fraction(1, x) for x in S}
            for a in S:
                others = [r[x] for x in S if x != a]
                if _subset_sum_possible(others, r[a]):
                    cnt += 1
                    if cnt >= best_count:
                        break
            if cnt < best_count:
                best_count = cnt
                best_el = v
                if cnt == 0:
                    break
        
        if best_el is None:
            # Fallback: just remove the largest element
            best_el = max(cur)
        
        cur.remove(best_el)
    
    return list(cur)


def simulated_annealing(initial, max_iters=50000, t0=1.0, t1=0.01, rng=None):
    """Simulated annealing to minimize conflicts while maximizing size."""
    if rng is None:
        rng = random.Random()
    
    cur = set(initial)
    cur_size = len(cur)
    cur_conf = _count_conflicts(cur)
    cur_score = cur_size - cur_conf  # maximize size, minimize conflicts
    
    best = set(cur)
    best_score = cur_score
    
    all_elems = list(range(1, 25))
    n_iters = max_iters
    
    for it in range(n_iters):
        if _deadline_hit():
            break
        
        T = t0 * (t1 / t0) ** (it / max(1, n_iters - 1))
        
        # Generate neighbor: add, remove, or swap
        r = rng.random()
        if r < 0.33:
            # Add a random element
            candidates = [v for v in all_elems if v not in cur]
            if not candidates:
                continue
            v = rng.choice(candidates)
            new_set = cur | {v}
        elif r < 0.66:
            # Remove a random element
            if not cur:
                continue
            v = rng.choice(list(cur))
            new_set = cur - {v}
        else:
            # Swap: remove one, add one
            if not cur:
                continue
            candidates = [v for v in all_elems if v not in cur]
            if not candidates:
                continue
            rem = rng.choice(list(cur))
            add = rng.choice(candidates)
            new_set = (cur - {rem}) | {add}
        
        new_size = len(new_set)
        new_conf = _count_conflicts(new_set)
        new_score = new_size - new_conf
        
        delta = new_score - cur_score
        if delta >= 0 or rng.random() < math.exp(delta / max(T, 1e-10)):
            cur = new_set
            cur_size = new_size
            cur_conf = new_conf
            cur_score = new_score
            
            if cur_conf == 0 and cur_size > len(best):
                best = set(cur)
                best_score = cur_score
        else:
            # Still track best valid
            if new_conf == 0 and new_size > len(best):
                best = set(new_set)
                best_score = new_score
    
    return list(best)


def construct():
    global _START
    _START = time.time()

    rng = random.Random(12345)
    best = []

    def consider(c):
        nonlocal best
        c = list(set(c))
        if not c:
            return
        if len(c) <= len(best):
            return
        if is_valid_fast(c):
            best = c

    # 1) structural anchors
    consider(cand_seed())
    for start in range(2, 25):
        if _deadline_hit():
            break
        consider(cand_top_half(start))
    for base in range(1, 6):
        for ratio in (2.1, 2.3, 2.5, 2.8, 3.0):
            consider(cand_geometric(base, ratio))
    
    # 2) greedy add from several orders
    for order in (list(range(24, 0, -1)),
                  list(range(1, 25)),
                  list(range(1, 25, 2)) + list(range(2, 25, 2)),
                  [24, 12, 23, 11, 22, 10, 21, 9, 20, 8, 19, 7, 18, 6, 17, 5, 16, 4, 15, 3, 14, 2, 13, 1]):
        if _deadline_hit():
            break
        consider(greedy_add(order, len(best)))
    
    # 3) random-shuffle greedy adds
    for _ in range(300):
        if _deadline_hit():
            break
        order = list(range(1, 25))
        rng.shuffle(order)
        consider(greedy_add(order, len(best)))
    
    # 4) greedy removal from full set
    if not _deadline_hit():
        consider(greedy_remove(len(best)))
    
    # 5) Local search on best found so far
    if best and not _deadline_hit():
        improved = local_search(best, max_steps=1000)
        consider(improved)
    
    # 6) More random starts with local search
    for _ in range(50):
        if _deadline_hit():
            break
        # Random initial set
        size = rng.randint(8, 16)
        init = rng.sample(range(1, 25), size)
        # Make it valid by removing elements
        cur = set(init)
        while not is_valid_fast(list(cur)) and cur:
            # Remove random element
            to_remove = rng.choice(list(cur))
            cur.remove(to_remove)
        if cur and len(cur) > len(best):
            consider(list(cur))
        if cur and len(cur) >= len(best) - 2:
            improved = local_search(list(cur), max_steps=200)
            consider(improved)
    
    # 7) Simulated annealing with precomputed conflicts
    if not _deadline_hit():
        # Precompute conflicts once
        _get_conflicts()
        
        # Multiple SA restarts from different initial sets
        sa_rng = random.Random(98765)
        for restart in range(20):
            if _deadline_hit():
                break
            
            # Start from current best or a random valid set
            if best and rng.random() < 0.5:
                initial = list(best)
            else:
                size = rng.randint(10, 18)
                initial = rng.sample(range(1, 25), size)
                # Make valid
                cur = set(initial)
                while not is_valid_fast(list(cur)) and cur:
                    to_remove = rng.choice(list(cur))
                    cur.remove(to_remove)
                initial = list(cur) if cur else [12]
            
            result = simulated_annealing(initial, max_iters=20000, t0=1.0, t1=0.005, rng=sa_rng)
            consider(result)
            
            # Also try local search on SA result
            if result and len(result) >= len(best):
                improved = local_search(result, max_steps=300)
                consider(improved)
    
    # 8) Final polish: local search on best
    if best and not _deadline_hit():
        improved = local_search(best, max_steps=500)
        consider(improved)
    
    # final safety: ensure nonempty and valid
    if not best:
        best = cand_seed()
    if not is_valid_fast(best):
        best = cand_seed()
    return list(best)


if __name__ == "__main__":
    print(construct())
