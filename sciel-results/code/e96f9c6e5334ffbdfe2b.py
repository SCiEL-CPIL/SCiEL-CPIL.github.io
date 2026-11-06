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
    
    # final safety: ensure nonempty and valid
    if not best:
        best = cand_seed()
    if not is_valid_fast(best):
        best = cand_seed()
    return list(best)


if __name__ == "__main__":
    print(construct())
