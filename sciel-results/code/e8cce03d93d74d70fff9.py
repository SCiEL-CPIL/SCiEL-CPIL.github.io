import random
import time
import math

# ---------- fast incremental Sidon / coverage evaluator ----------
# A Sidon set (distinct positive differences) is exactly: no difference value
# occurs more than once among all unordered pairs.
# We maintain diff_count[d] = number of pairs with difference d.

def _eval(set_):
    """Return (diff_count, is_sidon, coverage_ok) for a sorted set_ (0 in it)."""
    n = len(set_)
    dc = {}
    for i in range(n):
        a = set_[i]
        for j in range(i + 1, n):
            d = set_[j] - a
            dc[d] = dc.get(d, 0) + 1
    sidon = True
    for c in dc.values():
        if c > 1:
            sidon = False
            break
    cov = all(dc.get(d, 0) >= 1 for d in range(1, 31))
    return dc, sidon, cov


def _valid(set_):
    dc, sidon, cov = _eval(set_)
    return sidon and cov


# ---------- strong initial feasible set ----------
# Quadratic Sidon set {k^2} is a Sidon set (all differences distinct).
# We search small Sidon sets (subsets of small integers) that cover 1..30,
# using a randomized constructive + local shrink, to get a good starting point.

def _initial_feasible(rng, deadline):
    # Try to build a small Sidon set covering 1..30 with modest diameter.
    # Use the known small perfect-difference / Golomb structure by search.
    best = None
    best_diam = 10 ** 9

    # Candidate universe: integers 0..D. We'll do a randomized greedy + local
    # search to find any valid set, then shrink.
    for D in range(40, 200):
        if time.time() > deadline:
            break
        # Randomized greedy: add elements to cover missing differences.
        for attempt in range(40):
            if time.time() > deadline:
                break
            s = [0]
            dc = {}
            # we need to cover 1..30
            missing = set(range(1, 31))
            # repeatedly add a value that covers the most missing diffs without conflict
            for _ in range(60):
                if not missing:
                    break
                # candidate values 1..D
                best_add = None
                best_gain = -1
                for v in range(1, D + 1):
                    if v in s:
                        continue
                    # check conflict: new diffs must not duplicate existing
                    ok = True
                    gain = 0
                    for a in s:
                        d = abs(v - a)
                        if dc.get(d, 0) >= 1:
                            ok = False
                            break
                        if d in missing:
                            gain += 1
                    if ok and gain > best_gain:
                        best_gain = gain
                        best_add = v
                if best_add is None or best_gain == 0:
                    break
                s.append(best_add)
                s.sort()
                for a in s:
                    for b in s:
                        if a < b:
                            d = b - a
                            dc[d] = dc.get(d, 0) + 1
                missing = set(range(1, 31)) - set(dc.keys())
            if not missing:
                # valid set found
                diam = max(s)
                if diam < best_diam:
                    best_diam = diam
                    best = s[:]

    if best is not None:
        return best
    # Fallback: deterministic quadratic-based set (always valid Sidon, may need
    # more than 32? squares up to 30^2=900, but we only need 1..30 covered).
    # Use {0} plus squares 1^2..k^2; differences distinct. Cover 1..30?
    # squares: 0,1,4,9,16,25,36,... differences include 1,3,4,5,7,8,9,11,12,15,16,19,20,24,25,27,28,31...
    # not all 1..30. So build a guaranteed feasible set via a known construction:
    # take a Sidon set that is a "difference basis". Use greedy to extend to cover.
    s = [0]
    dc = {}
    missing = set(range(1, 31))
    v = 1
    while missing:
        # add v if no conflict, else try v+1
        placed = False
        for cand in range(v, v + 50):
            ok = True
            for a in s:
                d = abs(cand - a)
                if dc.get(d, 0) >= 1:
                    ok = False
                    break
            if ok:
                s.append(cand)
                s.sort()
                for a in s:
                    for b in s:
                        if a < b:
                            dc[b - a] = dc.get(b - a, 0) + 1
                missing = set(range(1, 31)) - set(dc.keys())
                v = cand
                placed = True
                break
        if not placed:
            v += 1
    return s


# ---------- iterated local search with incremental moves ----------
def construct():
    start = time.time()
    deadline = start + 145.0  # safety margin under 150s
    rng = random.Random(12345)

    init = _initial_feasible(rng, min(deadline, start + 20.0))
    best_set = init[:]
    best_diam = max(init)

    # If initial is already valid, we only need to shrink.
    # ILS loop: greedy descent (best improving single move) + perturbation.
    def _move_dc(dc, s, idx, newv):
        """Apply move: replace s[idx] with newv. Returns new dc (in place copy)."""
        # remove old diffs
        old = s[idx]
        for j in range(len(s)):
            if j == idx:
                continue
            d = abs(newv - s[j]) if False else abs(old - s[j])
            dc[d] -= 1
            if dc[d] == 0:
                del dc[d]
        # add new diffs
        for j in range(len(s)):
            if j == idx:
                continue
            d = abs(newv - s[j])
            dc[d] = dc.get(d, 0) + 1
        return dc

    def _try_shrink(s, dc, diam, max_iters):
        """Greedy descent: repeatedly apply best valid single move that reduces max."""
        n = len(s)
        improved = True
        while improved and max_iters > 0 and time.time() < deadline:
            improved = False
            max_iters -= 1
            cur_max = max(s)
            best_score = None
            best_move = None
            for idx in range(1, n):  # never move 0
                old = s[idx]
                # candidate new values: smaller than old (to shrink), within [1, cur_max]
                # focus on values that could reduce max: try old-1.. and some randoms
                cands = set()
                # try to pull the largest element down
                for v in range(1, old):
                    cands.add(v)
                # also a few random values below cur_max
                for _ in range(30):
                    cands.add(rng.randint(1, cur_max))
                for v in cands:
                    if v in s:
                        continue
                    # simulate
                    dc2 = dict(dc)
                    # remove old
                    for j in range(n):
                        if j == idx:
                            continue
                        d = abs(old - s[j])
                        dc2[d] -= 1
                        if dc2[d] == 0:
                            del dc2[d]
                    # add new
                    ok = True
                    for j in range(n):
                        if j == idx:
                            continue
                        d = abs(v - s[j])
                        c = dc2.get(d, 0)
                        if c >= 1:
                            ok = False
                            break
                        dc2[d] = c + 1
                    if not ok:
                        continue
                    # check coverage
                    if not all(dc2.get(d, 0) >= 1 for d in range(1, 31)):
                        continue
                    new_max = max(s[:idx] + [v] + s[idx + 1:])
                    # score: prefer lower max; tie-break fewer elements not needed (fixed n)
                    if best_score is None or new_max < best_score:
                        best_score = new_max
                        best_move = (idx, v, dc2, new_max)
            if best_move is not None:
                idx, v, dc2, new_max = best_move
                s = s[:idx] + [v] + s[idx + 1:]
                s.sort()
                dc = dc2
                diam = new_max
                improved = True
        return s, dc, diam

    def _perturb(s, dc, strength):
        """Randomly relocate a few elements to escape local optima, keep validity."""
        s = s[:]
        n = len(s)
        for _ in range(strength):
            idx = rng.randint(1, n - 1)
            cur_max = max(s)
            for _ in range(50):
                v = rng.randint(1, cur_max + 5)
                if v in s:
                    continue
                dc2 = dict(dc)
                old = s[idx]
                for j in range(n):
                    if j == idx:
                        continue
                    d = abs(old - s[j])
                    dc2[d] -= 1
                    if dc2[d] == 0:
                        del dc2[d]
                ok = True
                for j in range(n):
                    if j == idx:
                        continue
                    d = abs(v - s[j])
                    c = dc2.get(d, 0)
                    if c >= 1:
                        ok = False
                        break
                    dc2[d] = c + 1
                if not ok:
                    continue
                if not all(dc2.get(d, 0) >= 1 for d in range(1, 31)):
                    continue
                s = s[:idx] + [v] + s[idx + 1:]
                s.sort()
                dc = dc2
                break
        return s, dc

    # Main ILS
    s = init[:]
    s.sort()
    dc, sidon, cov = _eval(s)
    if not (sidon and cov):
        # recompute a valid starting point
        s = init[:]
        dc, sidon, cov = _eval(s)

    diam = max(s)
    iters = 0
    while time.time() < deadline:
        iters += 1
        # Greedy shrink
        s, dc, diam = _try_shrink(s, dc, diam, max_iters=2000)
        if diam < best_diam:
            best_diam = diam
            best_set = s[:]
        # Perturb to explore
        s, dc = _perturb(s, dc, rng.choice([2, 3, 4]))
        # quick re-eval to keep consistent
        dc2, sidon2, cov2 = _eval(s)
        if not (sidon2 and cov2):
            # revert to best
            s = best_set[:]
            dc = dc2
        else:
            dc = dc2
            diam = max(s)

    return best_set


if __name__ == "__main__":
    res = construct()
    print(res)
    print("Max:", max(res))
