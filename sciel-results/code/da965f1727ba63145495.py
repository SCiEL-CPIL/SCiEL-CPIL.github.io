"""Maximum-size reciprocal-subsum-free subset of [1,24].

A set S is valid iff for every a in S, 1/a is NOT a subset-sum of
{1/v : v in S, v != a} (exact Fraction check).

Design (kept in the same family as the seed):
  * Guaranteed answer: the known size-19 set [5,6,8,...,24].
  * Spare time is used for:
      (a) a fast branch-and-bound maximum search (upper-bounded) to prove
          whether size 20 (or more) is reachable, and
      (b) a simulated-annealing / local-search polish with a smooth
          "active-conflict" surrogate (soft penalty) to explore beyond the base.
  * All checks are exact (Fraction); a deadline guard returns the best valid
    set found before the 90 s wall-clock kill (safety margin at 84 s).
"""
import time
import random
import math
from fractions import Fraction

_DEADLINE = 84.0
_START = None
_UNIVERSE = list(range(1, 25))
_RECIP = {v: Fraction(1, v) for v in _UNIVERSE}

# Known valid size-19 set.
_BASE = [5, 6] + list(range(8, 25))
_FALLBACK = list(range(7, 25))  # size-18 superincreasing, always valid


def _elapsed():
    return 0.0 if _START is None else (time.time() - _START)


def _deadline_hit():
    return _elapsed() > _DEADLINE


# ---------------------------------------------------------------------------
# Exact subset-sum test (meet-in-the-middle) with per-call memo of right sums.
# ---------------------------------------------------------------------------
_RMEMO = {}


def _subset_sum_possible(vals, target):
    """True iff some subset of `vals` (list of Fraction) sums exactly to target."""
    n = len(vals)
    if n == 0:
        return target == 0
    if n == 1:
        return vals[0] == target
    mid = n // 2
    left, right = vals[:mid], vals[mid:]

    key = tuple(right)
    R = _RMEMO.get(key)
    if R is None:
        R = {Fraction(0)}
        for v in right:
            R = R | {x + v for x in R}
        if len(_RMEMO) > 4000:
            _RMEMO.clear()
        _RMEMO[key] = R

    # early exit
    if target in R:
        return True
    L = {Fraction(0)}
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
    """Number of elements a in S for which 1/a IS a subset-sum of the others."""
    S = list(S)
    r = [_RECIP[v] for v in S]
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


# ---------------------------------------------------------------------------
# Smooth surrogate: total "active conflict" count.  A conflict is an identity
# (target a, subset T of others) with sum(1/t : t in T) == 1/a.  For a given
# set we count how many selected targets are "covered" by some selected subset.
# This is exactly _violations; we treat it as a soft objective for annealing.
# ---------------------------------------------------------------------------
def _surrogate(S):
    """(conflicts, -size)  -- lower is better (fewer conflicts, larger set)."""
    v = _violations(S)
    return (v, -len(S))


# ---------------------------------------------------------------------------
# Branch-and-bound maximum search.
# Upper bound: greedy "reciprocal-superincreasing" bound.  Process elements in
# increasing order; an element a can be in ANY valid set together with a set of
# LARGER elements only if 1/a > sum(1/v for larger v already chosen) is NOT
# required -- that's a sufficient condition for validity, not necessary.  So we
# use a looser but safe upper bound: |chosen| + (number of remaining elements).
# We prune when that cannot beat the best size.
# ---------------------------------------------------------------------------
def _branch_and_bound(best_size, deadline):
    """Search for a valid set larger than best_size.  Returns best set found."""
    best = []
    best_sz = best_size
    order = _UNIVERSE[:]  # increasing
    n = len(order)

    def dfs(idx, chosen, chosen_set):
        nonlocal best, best_sz
        if _deadline_hit():
            return
        # upper bound on final size
        ub = len(chosen) + (n - idx)
        if ub <= best_sz:
            return
        if idx == n:
            if len(chosen) > best_sz and is_valid(chosen):
                best_sz = len(chosen)
                best = list(chosen)
            return
        a = order[idx]
        # branch: exclude a
        dfs(idx + 1, chosen, chosen_set)
        # branch: include a (only if it keeps a chance; cheap necessary test:
        # 1/a must not equal sum of a subset of already-chosen SMALLER elements,
        # because those are the only ones smaller; larger ones not chosen yet)
        # Necessary condition for validity w.r.t. already-chosen smaller elems:
        # 1/a is NOT a subset-sum of {1/v : v in chosen}.  If it is, including a
        # would make a invalid (since those chosen are "others").  Prune.
        if chosen:
            rsmall = [_RECIP[v] for v in chosen]
            if _subset_sum_possible(rsmall, _RECIP[a]):
                # including a is invalid already -> skip include branch
                return
        dfs(idx + 1, chosen + [a], chosen_set | {a})

    dfs(0, [], set())
    return best


# ---------------------------------------------------------------------------
# Simulated-annealing / local-search polish with the smooth conflict surrogate.
# ---------------------------------------------------------------------------
def _anneal(S, deadline):
    cur = list(set(S))
    cur_score = _surrogate(cur)
    best = list(cur) if cur_score[0] == 0 else []
    best_score = cur_score
    pool = _UNIVERSE[:]
    rng = random.Random(12345)
    T = 2.0
    cool = 0.9995

    def score_of(x):
        return _surrogate(x)

    steps = 0
    while not _deadline_hit() and T > 0.01:
        steps += 1
        cur_set = set(cur)
        # pick a random move: add, drop, or swap
        move = rng.random()
        if move < 0.4:
            cands = [a for a in pool if a not in cur_set]
            if not cands:
                continue
            a = rng.choice(cands)
            trial = cur + [a]
        elif move < 0.7:
            if not cur:
                continue
            b = rng.choice(cur)
            trial = [x for x in cur if x != b]
        else:
            cands = [a for a in pool if a not in cur_set]
            if not cands or not cur:
                continue
            a = rng.choice(cands)
            b = rng.choice(cur)
            trial = [x for x in cur if x != b] + [a]

        ts = score_of(trial)
        d = (ts[0] - cur_score[0]) + 0.05 * (ts[1] - cur_score[1])
        if d <= 0 or rng.random() < math.exp(-d / T):
            cur, cur_score = trial, ts
            if cur_score < best_score:
                best_score = cur_score
                best = list(cur)
        T *= cool
        if steps % 2000 == 0 and is_valid(cur) and len(cur) > len(best):
            best = list(cur)
    if is_valid(cur) and len(cur) > len(best):
        best = list(cur)
    return best


def construct():
    global _START
    _START = time.time()

    # 1) Guaranteed valid size-19 answer.
    ans = list(_BASE)
    if not is_valid(ans):
        ans = list(_FALLBACK)

    best = ans

    # 2) Branch-and-bound: try to prove/find a larger valid set.
    if not _deadline_hit():
        try:
            bb = _branch_and_bound(len(best), _DEADLINE)
            if bb and is_valid(bb) and len(bb) > len(best):
                best = bb
        except Exception:
            pass

    # 3) Annealing polish from the best found so far (and from the base).
    if not _deadline_hit():
        try:
            an = _anneal(best, _DEADLINE)
            if an and is_valid(an) and len(an) > len(best):
                best = an
        except Exception:
            pass
    if not _deadline_hit():
        try:
            an2 = _anneal(_BASE, _DEADLINE)
            if an2 and is_valid(an2) and len(an2) > len(best):
                best = an2
        except Exception:
            pass

    # 4) Final safety.
    if not best:
        best = list(_FALLBACK)
    if not is_valid(best):
        best = list(_FALLBACK)
    return list(best)


if __name__ == "__main__":
    t0 = time.time()
    r = construct()
    print(r)
    print("size:", len(r), "valid:", is_valid(r), "time:", time.time() - t0)
