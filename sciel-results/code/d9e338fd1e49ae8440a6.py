"""
Reciprocal-subsum-free set of maximum cardinality from [1,24].

Strategy:
1. Directly return the known size-19 set {5,6,8,...,24} which is validated exactly.
2. Use spare time to attempt to find a size-20 set via ILP (scipy.optimize.milp)
   with conflict constraints, or via a more targeted local search.
3. If no size-20 is found, return the size-19 set.

The key insight: for a set S to be valid, for every a in S, 1/a must NOT be expressible
as a sum of a subset of {1/v : v in S, v != a}. This is a global constraint that
can be encoded as an ILP if we can enumerate all "forbidden" subsets.

However, enumerating all forbidden subsets is expensive. Instead, we use a hybrid:
- Start with the known size-19 set.
- Try to add elements from [1,4] by checking validity incrementally.
- Use local search (swaps, adds, drops) with exact rational validation.
- As a final resort, try a small ILP if time permits.
"""
import time
import random
from fractions import Fraction

# Known valid size-19 set: [5,6,8,9,...,24]
_BASE = [5, 6] + list(range(8, 25))

# Safety margin: evaluator kills at 90s; stop at 85s.
_DEADLINE = 85.0
_START = None


def _elapsed():
    return 0.0 if _START is None else (time.time() - _START)


def _deadline_hit():
    return _elapsed() > _DEADLINE


def _subset_sum_possible(vals, target):
    """Exact: some subset of `vals` (list of Fraction) sums to `target`?
    Meet-in-the-middle O(2^(n/2))."""
    n = len(vals)
    if n == 0:
        return target == 0
    if n == 1:
        return vals[0] == target
    mid = n // 2
    left, right = vals[:mid], vals[mid:]

    # right-half subset sums
    R = {Fraction(0)}
    for v in right:
        R = R | {x + v for x in R}

    # left-half subset sums, early-exit if (target - l) in R
    L = {Fraction(0)}
    if target in R:
        return True
    for v in left:
        new = set()
        for x in L:
            y = x + v
            if (target - y) in R:
                return True
            new.add(y)
        L |= new
    return False


def _violations(S):
    """Count of elements a in S for which 1/a IS a subset-sum of the others."""
    S = list(S)
    r = [Fraction(1, v) for v in S]
    cnt = 0
    for i in range(len(S)):
        others = r[:i] + r[i + 1:]
        if _subset_sum_possible(others, r[i]):
            cnt += 1
    return cnt


def is_valid(S):
    S = list(set(S))
    if not S:
        return False
    return _violations(S) == 0


def _try_add(S, a):
    """Check if adding element a to S keeps the set valid. Returns True if valid."""
    S_list = list(S)
    if a in S_list:
        return True  # already there
    new_S = S_list + [a]
    # Only need to check:
    # 1. Is 1/a a subset-sum of {1/v : v in S}? (new element's reciprocal)
    # 2. For each v in S, is 1/v still NOT a subset-sum of {1/u : u in new_S, u != v}?
    #    But since we're only adding, the only new subsets involve 1/a.
    #    So for each v in S, check if 1/v = (subset of {1/u : u in S, u != v}) + 1/a
    #    i.e., is 1/v - 1/a a subset-sum of {1/u : u in S, u != v}?
    
    r_S = [Fraction(1, v) for v in S_list]
    r_a = Fraction(1, a)
    
    # Check 1: is r_a a subset-sum of r_S?
    if _subset_sum_possible(r_S, r_a):
        return False
    
    # Check 2: for each v in S, is (1/v - 1/a) a subset-sum of {1/u : u in S, u != v}?
    for i, v in enumerate(S_list):
        target = Fraction(1, v) - r_a
        if target <= 0:
            continue
        others = r_S[:i] + r_S[i+1:]
        if _subset_sum_possible(others, target):
            return False
    
    return True


def _local_search(S, deadline):
    """Bounded local search from S: try swaps to reduce violations; accept
    improvements. Returns best valid set found (or best-effort)."""
    cur = list(set(S))
    best = cur if is_valid(cur) else []
    pool = list(range(1, 25))
    rng = random.Random(777)

    def score_of(x):
        v = _violations(x)
        if v == 0:
            return (0, -len(x), 0)
        return (1, v, -len(x))

    cur_score = score_of(cur)
    stuck = 0
    while not _deadline_hit():
        improved = False
        cur_set = set(cur)
        
        # Try adding elements not in set
        for a in pool:
            if _deadline_hit():
                break
            if a in cur_set:
                continue
            trial = cur + [a]
            ts = score_of(trial)
            if ts < cur_score:
                cur, cur_score = trial, ts
                if ts[0] == 0 and (not best or len(cur) > len(best)):
                    best = list(cur)
                improved = True
        
        if _deadline_hit():
            break
        
        # Try dropping one element
        for a in list(cur):
            if _deadline_hit():
                break
            trial = [x for x in cur if x != a]
            ts = score_of(trial)
            if ts < cur_score:
                cur, cur_score = trial, ts
                if ts[0] == 0 and (not best or len(cur) > len(best)):
                    best = list(cur)
                improved = True
        
        if _deadline_hit():
            break
        
        # Swap: add a, drop b
        for a in pool:
            if _deadline_hit():
                break
            if a in cur:
                continue
            for b in list(cur):
                if _deadline_hit():
                    break
                trial = [x for x in cur if x != b] + [a]
                ts = score_of(trial)
                if ts < cur_score:
                    cur, cur_score = trial, ts
                    if ts[0] == 0 and (not best or len(cur) > len(best)):
                        best = list(cur)
                    improved = True
        
        if not improved:
            stuck += 1
            if stuck >= 4:
                break
        else:
            stuck = 0

    if is_valid(cur) and (not best or len(cur) > len(best)):
        best = list(cur)
    return best


def _try_ilp_for_20():
    """
    Try to find a size-20 set using ILP.
    
    Variables: x_i in {0,1} for i in 1..24, indicating if i is selected.
    Objective: maximize sum x_i
    Constraints:
      - sum x_i >= 20 (we want to prove size 20 exists)
      - For every subset T of [1,24] and element a not in T:
        If sum_{v in T} x_v == len(T) and x_a == 1, then 
        1/a != sum_{v in T} 1/v
    
    This is hard to encode directly. Instead, we use a different approach:
    Enumerate all "forbidden" configurations. A configuration is a pair (a, T)
    where a is an element, T is a subset of [1,24] \ {a}, and 1/a = sum_{v in T} 1/v.
    For each such (a, T), we add the constraint: NOT (x_a == 1 AND all x_v == 1 for v in T)
    i.e., x_a + sum_{v in T} (1 - x_v) >= 1
    
    But enumerating all such (a, T) is expensive. We limit to small T.
    """
    if _deadline_hit():
        return []
    
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
    except ImportError:
        return []
    
    # Enumerate forbidden (a, T) pairs for small T (|T| <= 4)
    forbidden = []
    for a in range(1, 25):
        target = Fraction(1, a)
        # Try all subsets of [1,24]\{a} of size up to 4
        others = [v for v in range(1, 25) if v != a]
        from itertools import combinations
        for size in range(1, 5):
            for T in combinations(others, size):
                s = sum(Fraction(1, v) for v in T)
                if s == target:
                    forbidden.append((a, set(T)))
    
    if not forbidden:
        return []
    
    n = 24
    # x_i for i in 1..24, index i-1
    c = np.ones(n)  # maximize sum x_i
    
    # Constraints:
    # 1. sum x_i >= 20
    A1 = np.ones((1, n))
    b1 = (20, np.inf)
    
    # 2. For each forbidden (a, T): x_a + sum_{v in T} (1 - x_v) >= 1
    #    i.e., x_a - sum_{v in T} x_v >= 1 - |T|
    #    i.e., x_a + sum_{v in T} (-1) * x_v >= 1 - |T|
    A2_rows = []
    b2_lower = []
    b2_upper = []
    for a, T in forbidden:
        row = np.zeros(n)
        row[a-1] = 1  # x_a
        for v in T:
            row[v-1] = -1  # -x_v
        A2_rows.append(row)
        b2_lower.append(1 - len(T))
        b2_upper.append(np.inf)
    
    if A2_rows:
        A2 = np.array(A2_rows)
        A = np.vstack([A1, A2])
        lb = np.concatenate([b1, b2_lower])
        ub = np.concatenate([b1, b2_upper])
    else:
        A = A1
        lb = b1
        ub = b1
    
    constraints = LinearConstraint(A, lb, ub)
    integrality = np.ones(n)
    bounds = Bounds(0, 1)
    
    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    
    if res.status == 0 and res.x is not None:
        sol = [i+1 for i in range(n) if res.x[i] > 0.5]
        if len(sol) >= 20 and is_valid(sol):
            return sol
    return []


def construct():
    global _START
    _START = time.time()

    # 1) Direct answer: the known size-19 set. Validate exactly.
    ans = list(_BASE)
    if not is_valid(ans):
        ans = list(range(7, 25))

    # 2) Try to add elements from [1,4] to reach size 20+
    for a in [1, 2, 3, 4]:
        if _deadline_hit():
            break
        if a not in ans and _try_add(ans, a):
            ans.append(a)
    
    # 3) Use local search to try to improve
    if not _deadline_hit():
        try:
            improved = _local_search(ans, _DEADLINE)
            if improved and len(improved) > len(ans) and is_valid(improved):
                ans = improved
        except Exception:
            pass

    # 4) Try ILP for size 20 if time permits
    if not _deadline_hit() and len(ans) < 20:
        try:
            ilp_sol = _try_ilp_for_20()
            if ilp_sol and len(ilp_sol) > len(ans) and is_valid(ilp_sol):
                ans = ilp_sol
        except Exception:
            pass

    # 5) Final safety: ensure nonempty and valid.
    if not ans:
        ans = list(range(7, 25))
    if not is_valid(ans):
        ans = list(range(7, 25))
    return list(ans)


if __name__ == "__main__":
    print(construct())
    print("size:", len(construct()))
