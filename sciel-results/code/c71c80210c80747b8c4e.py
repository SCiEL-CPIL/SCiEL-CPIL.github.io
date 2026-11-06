"""Reciprocal-subsum-free set task: maximize |S| for S subset of [1,24] such
that for every a in S, 1/a is NOT a subset sum of {1/b : b in S, b != a}.

Strategy (combined):
  * Structural floor: the seed top-half [13..24] is a valid baseline (12).
  * Fast incremental validity: maintain the set of achievable subset sums of
    reciprocals; adding an element is checked against the *current* sum-set,
    then the set is updated.  This makes greedy-add very cheap.
  * Local search (hill-climb): swap an in-set element for an out-set element
    when it keeps validity and (ideally) keeps size, used to escape plateaus.
  * Multiple randomized restarts with a wall-clock deadline guard.
  * Exact Fraction arithmetic throughout (no floating-point false positives).
"""
import time
import random
from fractions import Fraction

DEADLINE = 82.0  # safety margin under the 90 s wall-clock kill
_START = None


def _deadline_hit():
    global _START
    if _START is None:
        return False
    return (time.time() - _START) > DEADLINE


# ---------- exact subset-sum (meet-in-the-middle) ----------

def _subset_sum_possible(vals, target):
    """True iff some subset of `vals` (Fractions) sums exactly to `target`."""
    n = len(vals)
    if n == 0:
        return target == Fraction(0)
    mid = n // 2
    left, right = vals[:mid], vals[mid:]

    L = {Fraction(0)}
    for v in left:
        L = L | {x + v for x in L}
    R = {Fraction(0)}
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


# ---------- fast incremental validity machinery ----------

def _sums(vals):
    """All subset sums of a list of Fractions (set)."""
    out = {Fraction(0)}
    for v in vals:
        out = out | {x + v for x in out}
    return out


def _valid_with_sums(vals):
    """Return (ok, sums_set) where ok means: for every i, vals[i] is not a
    subset sum of vals[:i] + vals[i+1:].  sums_set = all subset sums of vals."""
    n = len(vals)
    pref = [None] * (n + 1)
    pref[0] = {Fraction(0)}
    for i in range(n):
        pref[i + 1] = pref[i] | {x + vals[i] for x in pref[i]}
    suff = [None] * (n + 1)
    suff[n] = {Fraction(0)}
    for i in range(n - 1, -1, -1):
        suff[i] = suff[i + 1] | {x + vals[i] for x in suff[i + 1]}
    for i in range(n):
        target = vals[i]
        # subset sums of all elements except i  =  combine pref[i] and suff[i+1]
        L, R = pref[i], suff[i + 1]
        if len(L) <= len(R):
            ok = all((target - l) not in R for l in L)
        else:
            ok = all((target - r) not in L for r in R)
        if not ok:
            return False, None
    return True, pref[n]


# ---------- candidate generators ----------

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


def greedy_add(order):
    """Grow a set by adding elements in `order`, using fast incremental check."""
    chosen = []
    recips = []
    sums = {Fraction(0)}
    for v in order:
        if v in chosen:
            continue
        rv = Fraction(1, v)
        # necessary: 1/v not a subset sum of current reciprocals
        if rv in sums:
            continue
        # sufficient: full incremental validity of the new list
        trial = recips + [rv]
        ok, new_sums = _valid_with_sums(trial)
        if ok:
            chosen.append(v)
            recips = trial
            sums = new_sums
    return chosen


def local_search(S, rng, iters=2000):
    """Hill-climb: swap an in-set element for an out-set element when valid.
    Returns a (possibly larger) valid set."""
    S = list(set(S))
    for _ in range(iters):
        if _deadline_hit():
            break
        recips = [Fraction(1, v) for v in S]
        ok, sums = _valid_with_sums(recips)
        if not ok:
            break
        in_set = set(S)
        out_set = [v for v in range(1, 25) if v not in in_set]
        if not out_set:
            break
        # try a random swap; accept if result is valid (size preserved)
        i = rng.randrange(len(S))
        j = rng.randrange(len(out_set))
        newS = S[:i] + S[i + 1:] + [out_set[j]]
        newrecips = [Fraction(1, v) for v in newS]
        ok2, _ = _valid_with_sums(newrecips)
        if ok2:
            S = newS
    return S


def greedy_remove(best_size):
    """Start from [1,24]; remove elements until valid (heuristic)."""
    cur = set(range(1, 25))
    changed = True
    while changed and (not is_valid(cur)):
        changed = False

        def violations(T):
            T = list(T)
            r = {v: Fraction(1, v) for v in T}
            cnt = 0
            for a in T:
                others = [r[v] for v in T if v != a]
                if _subset_sum_possible(others, r[a]):
                    cnt += 1
            return cnt

        base_v = violations(cur)
        if base_v == 0:
            break
        best_el, best_after = None, base_v
        for v in sorted(cur, reverse=True):
            trial = cur - {v}
            a = violations(trial)
            if a < best_after:
                best_after, best_el = a, v
                if a == 0:
                    break
        if best_el is None:
            break
        cur.remove(best_el)
        changed = True
        if _deadline_hit():
            break
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
        if is_valid(c):
            best = c

    # 1) structural anchors (guaranteed floor = seed, 12)
    consider(cand_seed())
    for start in range(2, 25):
        if _deadline_hit():
            break
        consider(cand_top_half(start))
    for base in range(1, 6):
        for ratio in (2.1, 2.3, 2.5, 2.8, 3.0):
            consider(cand_geometric(base, ratio))

    # 2) greedy add from several fixed orders (fast incremental)
    for order in (list(range(24, 0, -1)),
                  list(range(1, 25)),
                  list(range(1, 25, 2)) + list(range(2, 25, 2)),
                  [24, 12, 23, 11, 22, 10, 21, 9, 20, 8, 19, 7, 18, 6, 17, 5, 16, 4, 15, 3, 14, 2, 13, 1]):
        if _deadline_hit():
            break
        consider(greedy_add(order))

    # 3) randomized restarts: greedy-add in a random order, then local search
    #    (the seed anchor is the floor, so best never regresses)
    while not _deadline_hit():
        order = list(range(1, 25))
        rng.shuffle(order)
        cand = greedy_add(order)
        cand = local_search(cand, rng, iters=1500)
        consider(cand)
        # occasionally also try local search seeded from the current best
        if len(best) >= 12:
            cand2 = local_search(list(best), rng, iters=800)
            consider(cand2)

    # 4) greedy removal from full set (last resort)
    if not _deadline_hit():
        consider(greedy_remove(len(best)))

    # final safety: ensure nonempty and valid
    if not best:
        best = cand_seed()
    if not is_valid(best):
        best = cand_seed()
    return list(best)


if __name__ == "__main__":
    print(construct())
