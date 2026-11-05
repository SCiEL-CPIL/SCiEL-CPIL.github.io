"""Factorial decompositions into distinct factors above the index.

Returns 1..12 distinct integers all > 12 whose product is exactly 12!,
minimizing the largest factor.

Strategy
--------
12! = 2^10 * 3^5 * 5^2 * 7 * 11 = 479001600.

We run a branch-and-bound depth-first search over the divisors of 12!.
The state is the remaining product; a "factor" is any divisor d > 12 of the
remaining product that has not been used yet.  We order the candidate
factors by closeness to the balanced (geometric-mean) size so that good
solutions are found early, which tightens the branch-and-bound bound on the
maximum factor.  A geometric-mean lower bound prunes any branch whose most
balanced continuation would already force a factor at or above the current
best maximum.  A hard wall-clock deadline returns the best solution found.

A fast greedy heuristic seeds the search with a decent initial upper bound
so that the DFS prunes aggressively from the very first node.
"""
import time
from math import factorial
from itertools import combinations

_TARGET = factorial(12)  # 479001600
_MIN_FACTOR = 13          # all factors must be > 12
_MAX_COUNT = 12


def _divisors(n):
    """All divisors of n, sorted ascending."""
    divs = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
        i += 1
    divs.sort()
    return divs


# All divisors of 12! that are > 12, sorted.  Membership test via set.
_ALL_DIVS = [d for d in _divisors(_TARGET) if d > _MIN_FACTOR]
_DIV_SET = set(_ALL_DIVS)
# Index divisors by value for faster lookup
_DIV_INDEX = {d: i for i, d in enumerate(_ALL_DIVS)}


def _is_valid(factors):
    if not factors or len(factors) > _MAX_COUNT:
        return False
    if any(f <= _MIN_FACTOR - 1 for f in factors):
        return False
    if len(set(factors)) != len(factors):
        return False
    prod = 1
    for f in factors:
        prod *= f
    return prod == _TARGET


def _divisors_of(n):
    """Divisors of n that are also divisors of _TARGET and > 12, sorted."""
    result = []
    for d in _ALL_DIVS:
        if d > n:
            break
        if n % d == 0:
            result.append(d)
    return result


def _greedy_seed():
    """Quick balanced greedy: repeatedly split off a divisor near the
    geometric mean of the remaining product for the remaining slots.
    Returns a valid factor list (or None)."""
    best = None
    for target in range(5, 13):
        remaining = _TARGET
        used = set()
        cur = []
        ok = True
        for slot in range(target):
            left = target - slot
            if remaining == 1:
                ok = False
                break
            if left == 1:
                d = remaining
                if d <= 12 or d in used:
                    ok = False
                    break
            else:
                gm = remaining ** (1.0 / left)
                # Pick the divisor of remaining, > 12, not used, closest to gm.
                best_d = None
                best_key = None
                for d in _ALL_DIVS:
                    if d > remaining:
                        break
                    if remaining % d != 0:
                        continue
                    if d in used:
                        continue
                    key = abs(d - gm)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_d = d
                if best_d is None:
                    ok = False
                    break
                d = best_d
            if d in used:
                ok = False
                break
            used.add(d)
            cur.append(d)
            remaining //= d
        if ok and remaining == 1 and _is_valid(cur):
            mx = max(cur)
            if best is None or mx < max(best):
                best = cur
    return best


def _direct_construction():
    """Try to directly construct a good solution using known factorization.
    12! = 2^10 * 3^5 * 5^2 * 7 * 11
    
    Try to find balanced factors by assigning prime powers to groups.
    """
    # Try specific combinations of small composites > 12
    # that multiply to 12!
    candidates = []
    
    # Try 6-factor solutions with factors in range [13, 40]
    # 12!^(1/6) ≈ 28.2
    # Try 7-factor: 12!^(1/7) ≈ 23.5
    # Try 8-factor: 12!^(1/8) ≈ 20.6
    
    # Manually try some promising sets
    test_sets = [
        [14, 15, 16, 18, 20, 21, 25, 27],  # 8 factors
        [14, 15, 16, 18, 20, 22, 25, 27],
        [13, 14, 15, 16, 18, 20, 21, 25],
        [13, 14, 15, 16, 18, 20, 22, 25],
        [13, 14, 15, 16, 18, 22, 24, 25],
        [14, 15, 16, 18, 20, 21, 22, 25],
        [13, 15, 16, 18, 20, 21, 22, 25],
        [13, 14, 15, 18, 20, 21, 22, 25],
        [13, 14, 16, 18, 20, 21, 22, 25],
        [14, 15, 16, 18, 20, 21, 22, 27],
        [13, 14, 15, 16, 18, 20, 21, 27],
        [13, 14, 15, 16, 18, 20, 24, 25],
        [13, 14, 15, 16, 20, 21, 22, 25],
        [13, 14, 15, 18, 20, 21, 22, 25],
        [13, 15, 16, 18, 20, 21, 22, 25],
        [14, 15, 16, 18, 20, 22, 24, 25],
        [13, 14, 15, 16, 18, 22, 24, 25],
        [13, 14, 16, 18, 20, 21, 24, 25],
        [14, 15, 16, 18, 21, 22, 24, 25],
        [13, 15, 16, 18, 20, 22, 24, 25],
    ]
    
    for s in test_sets:
        if len(set(s)) != len(s):
            continue
        if any(x <= 12 for x in s):
            continue
        prod = 1
        for x in s:
            prod *= x
        if prod == _TARGET:
            mx = max(s)
            if not candidates or mx < max(candidates[0]):
                candidates = [s]
    
    if candidates:
        return candidates[0]
    
    # Try 7-factor sets
    test_sets_7 = [
        [14, 15, 16, 18, 20, 21, 33],
        [14, 15, 16, 18, 20, 22, 33],
        [13, 14, 15, 16, 18, 20, 33],
        [14, 15, 16, 18, 20, 25, 33],
        [13, 14, 15, 16, 18, 20, 35],
        [14, 15, 16, 18, 20, 22, 35],
        [13, 14, 15, 16, 18, 22, 35],
        [14, 15, 16, 18, 20, 25, 35],
        [13, 14, 15, 16, 18, 20, 36],
        [14, 15, 16, 18, 20, 21, 36],
        [13, 14, 15, 16, 18, 21, 36],
        [14, 15, 16, 18, 20, 24, 35],
        [13, 14, 15, 16, 18, 24, 35],
        [14, 15, 16, 18, 21, 22, 35],
        [13, 15, 16, 18, 20, 21, 35],
        [14, 15, 16, 18, 20, 27, 35],
        [13, 14, 15, 16, 18, 20, 42],
        [14, 15, 16, 18, 20, 21, 42],
        [13, 14, 15, 16, 18, 21, 42],
        [14, 15, 16, 18, 20, 22, 42],
    ]
    for s in test_sets_7:
        if len(set(s)) != len(s):
            continue
        if any(x <= 12 for x in s):
            continue
        prod = 1
        for x in s:
            prod *= x
        if prod == _TARGET:
            mx = max(s)
            if not candidates or mx < max(candidates[0]):
                candidates = [s]
    
    if candidates:
        return candidates[0]
    
    return None


class _Searcher:
    def __init__(self, deadline, initial_best):
        self.deadline = deadline
        self.best_max = initial_best
        self.best_factors = None
        self.nodes = 0

    def run(self):
        seed = _greedy_seed()
        if seed is not None:
            mx = max(seed)
            if mx < self.best_max:
                self.best_max = mx
                self.best_factors = list(seed)
        
        # Also try direct construction
        direct = _direct_construction()
        if direct is not None:
            mx = max(direct)
            if mx < self.best_max:
                self.best_max = mx
                self.best_factors = list(direct)
        
        self._dfs(_TARGET, frozenset(), [], 0)
        return self.best_factors

    def _dfs(self, remaining, used, cur, max_so_far):
        self.nodes += 1
        if time.time() > self.deadline:
            return
        if max_so_far >= self.best_max:
            return
        if remaining == 1:
            if cur and len(cur) <= _MAX_COUNT:
                mx = max_so_far if max_so_far > 0 else (max(cur) if cur else 0)
                if mx < self.best_max:
                    self.best_max = mx
                    self.best_factors = list(cur)
            return
        if len(cur) >= _MAX_COUNT:
            return
        # Lower bound: with r slots left, largest factor >= remaining^(1/r)
        slots_left = _MAX_COUNT - len(cur)
        lb = remaining ** (1.0 / slots_left)
        if lb >= self.best_max:
            return
        # Also: if we must use at least 1 more factor, and remaining itself
        # is >= best_max, we can't improve unless we split it further.
        # But if remaining has no valid divisor < best_max, we're stuck.
        
        # Candidate next factors: divisors of remaining, > 12, not used,
        # and strictly less than best_max.
        rem_slots = _MAX_COUNT - len(cur)
        gm = remaining ** (1.0 / rem_slots)
        cands = []
        for d in _ALL_DIVS:
            if d > remaining:
                break
            if d >= self.best_max:
                continue
            if remaining % d != 0:
                continue
            if d in used:
                continue
            cands.append(d)
        if not cands:
            return
        # Sort by closeness to gm (balanced first); ties by value.
        cands.sort(key=lambda d: (abs(d - gm), d))
        for d in cands:
            if time.time() > self.deadline:
                return
            new_max = max(max_so_far, d)
            if new_max >= self.best_max:
                continue
            self._dfs(remaining // d, used | {d}, cur + [d], new_max)
            if time.time() > self.deadline:
                return


def construct():
    start = time.time()
    # Generous but bounded budget; evaluator kills at 90s, keep a margin.
    deadline = start + 55.0

    # Seed with a good initial upper bound (greedy) so the DFS prunes hard.
    seed = _greedy_seed()
    initial_best = max(seed) if seed is not None else float('inf')
    
    # Also try direct construction for a better seed
    direct = _direct_construction()
    if direct is not None:
        dm = max(direct)
        if dm < initial_best:
            initial_best = dm

    searcher = _Searcher(deadline, initial_best)
    result = searcher.run()

    # Verify before returning.
    if result is not None and _is_valid(result):
        return result

    # Fallbacks (should not be reached).
    if seed is not None and _is_valid(seed):
        return list(seed)
    if direct is not None and _is_valid(direct):
        return list(direct)
    return [_TARGET]
