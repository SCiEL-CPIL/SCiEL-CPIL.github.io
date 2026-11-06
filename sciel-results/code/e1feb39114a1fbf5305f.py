"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

_DEADLINE = 150.0
_MARGIN = 4.0

# ---------------------------------------------------------------------------
# Fast incremental Sidon tracker: O(n) per add/remove instead of O(n^2).
# ---------------------------------------------------------------------------
class _Tracker:
    __slots__ = ("arr", "diff_count", "cov")

    def __init__(self, arr):
        self.arr = sorted(arr)
        self.diff_count = {}
        self.cov = 0
        for i in range(len(self.arr)):
            ai = self.arr[i]
            for j in range(i + 1, len(self.arr)):
                d = self.arr[j] - ai
                c = self.diff_count.get(d, 0) + 1
                self.diff_count[d] = c
                if c == 1 and 1 <= d <= 30:
                    self.cov += 1

    def is_valid(self):
        if self.cov < 30:
            return False
        return all(c == 1 for c in self.diff_count.values())

    def add(self, x):
        a = self.arr
        n = len(a)
        i = 0
        while i < n and a[i] < x:
            i += 1
        a.insert(i, x)
        dc = self.diff_count
        for j in range(i):
            d = x - a[j]
            c = dc.get(d, 0) + 1
            dc[d] = c
            if c == 1 and 1 <= d <= 30:
                self.cov += 1
        for j in range(i + 1, len(a)):
            d = a[j] - x
            c = dc.get(d, 0) + 1
            dc[d] = c
            if c == 1 and 1 <= d <= 30:
                self.cov += 1

    def remove(self, x):
        a = self.arr
        n = len(a)
        i = 0
        while i < n and a[i] != x:
            i += 1
        if i >= n:
            return
        a.pop(i)
        dc = self.diff_count
        for j in range(i):
            d = x - a[j]
            c = dc[d] - 1
            if c == 0:
                del dc[d]
                if 1 <= d <= 30:
                    self.cov -= 1
            else:
                dc[d] = c
        for j in range(i, len(a)):
            d = a[j] - x
            c = dc[d] - 1
            if c == 0:
                del dc[d]
                if 1 <= d <= 30:
                    self.cov -= 1
            else:
                dc[d] = c

    def max(self):
        return self.arr[-1]


def _valid_list(s):
    """Cheap standalone validity check on a plain list."""
    if 0 not in s or len(s) < 2 or len(s) > 32:
        return False
    if max(s) > 10**6:
        return False
    t = _Tracker(s)
    return t.is_valid()


# ---------------------------------------------------------------------------
# Deterministic constructive search with a diameter cap.
# At each step place a single element (or a pair) that covers the most
# still-missing differences while keeping the set Sidon.  This is the
# mechanism that actually shrinks the diameter; random restarts alone do not.
# ---------------------------------------------------------------------------
def _constructive(cap, rng, time_limit):
    arr = [0]
    dc = {}
    cov = 0
    need = set(range(1, 31))

    def add(x):
        nonlocal cov
        n = len(arr)
        i = 0
        while i < n and arr[i] < x:
            i += 1
        arr.insert(i, x)
        for j in range(i):
            d = x - arr[j]
            c = dc.get(d, 0) + 1
            dc[d] = c
            if c == 1 and d in need:
                cov += 1
                need.discard(d)
        for j in range(i + 1, len(arr)):
            d = arr[j] - x
            c = dc.get(d, 0) + 1
            dc[d] = c
            if c == 1 and d in need:
                cov += 1
                need.discard(d)

    def remove(x):
        nonlocal cov
        n = len(arr)
        i = 0
        while i < n and arr[i] != x:
            i += 1
        if i >= n:
            return
        arr.pop(i)
        for j in range(i):
            d = x - arr[j]
            c = dc[d] - 1
            if c == 0:
                del dc[d]
                if d in range(1, 31):
                    cov -= 1
                    need.add(d)
            else:
                dc[d] = c
        for j in range(i, len(arr)):
            d = arr[j] - x
            c = dc[d] - 1
            if c == 0:
                del dc[d]
                if d in range(1, 31):
                    cov -= 1
                    need.add(d)
            else:
                dc[d] = c

    def sidon_ok():
        return all(c == 1 for c in dc.values())

    for _ in range(1000):
        if not need:
            break
        if len(arr) > 31:
            break
        if time.time() > time_limit:
            return None
        best_x, best_gain = None, 0
        best_pair, best_pair_gain = None, 0
        # single-element candidates
        cand = []
        for x in range(1, cap + 1):
            if x in arr:
                continue
            # quick reject if any diff already collides
            ok = True
            gain = 0
            for v in arr:
                d = abs(x - v)
                if d in dc and dc[d] >= 1:
                    ok = False
                    break
                if d in need:
                    gain += 1
            if not ok:
                continue
            cand.append((gain, x))
        cand.sort(reverse=True)
        for gain, x in cand[:40]:
            if gain > best_gain:
                best_gain = gain
                best_x = x
        # pair candidates (x, x+d) for small d
        for d in range(1, 25):
            for base in range(1, cap + 1 - d):
                x = base
                y = base + d
                if x in arr or y in arr:
                    continue
                ok = True
                gain = 0
                for v in arr:
                    for u in (x, y):
                        dd = abs(u - v)
                        if dd in dc and dc[dd] >= 1:
                            ok = False
                            break
                        if dd in need:
                            gain += 1
                    if not ok:
                        break
                if not ok:
                    continue
                dd = y - x
                if dd in dc and dc[dd] >= 1:
                    continue
                if gain > best_pair_gain:
                    best_pair_gain = gain
                    best_pair = (x, y)
        if best_pair is not None and best_pair_gain >= best_gain:
            x, y = best_pair
            add(x)
            add(y)
        elif best_x is not None and best_gain > 0:
            add(best_x)
        else:
            break
    if need:
        return None
    return sorted(arr)


# ---------------------------------------------------------------------------
# Refinement machinery (kept from the seed): shrink diameter, remove elements.
# ---------------------------------------------------------------------------
def _try_remove_elements(arr, deadline):
    t = _Tracker(arr)
    changed = True
    while changed and time.time() < deadline - _MARGIN:
        changed = False
        for i in range(len(t.arr) - 1, 0, -1):
            if t.arr[i] == 0:
                continue
            x = t.arr[i]
            t.remove(x)
            if t.is_valid():
                changed = True
            else:
                t.add(x)
    return list(t.arr)


def _local_shrink(arr, rng, deadline):
    t = _Tracker(arr)
    improved = True
    while improved and time.time() < deadline - _MARGIN:
        improved = False
        D = t.max()
        # replace each non-zero element with a smaller one
        for i in range(len(t.arr) - 1, 0, -1):
            x = t.arr[i]
            for _ in range(60):
                v = rng.randint(1, D - 1)
                if v in t.arr:
                    continue
                t.remove(x)
                t.add(v)
                if t.is_valid():
                    improved = True
                else:
                    t.remove(v)
                    t.add(x)
                if improved:
                    break
        # replace largest with a smaller one
        if time.time() < deadline - _MARGIN:
            largest = t.max()
            for _ in range(120):
                v = rng.randint(1, largest - 1)
                if v in t.arr:
                    continue
                t.remove(largest)
                t.add(v)
                if t.is_valid():
                    improved = True
                else:
                    t.remove(v)
                    t.add(largest)
    return list(t.arr)


def _sa_refine(arr, rng, deadline, T0=40.0, cooling=0.997, max_iter=4000):
    t = _Tracker(arr)
    best = list(t.arr)
    best_max = t.max()
    cur_max = best_max
    T = T0
    for _ in range(max_iter):
        if time.time() > deadline - _MARGIN:
            break
        i = rng.randint(1, len(t.arr) - 1)
        old = t.arr[i]
        if rng.random() < 0.6:
            delta = rng.randint(-30, 30)
            new = old + delta
        else:
            new = rng.randint(1, t.max())
        if new <= 0 or new in t.arr:
            continue
        t.remove(old)
        t.add(new)
        if t.is_valid():
            nm = t.max()
            if nm < cur_max or (nm == cur_max and rng.random() < 0.3):
                cur_max = nm
                if nm < best_max:
                    best_max = nm
                    best = list(t.arr)
        else:
            # accept worsening with small prob (keep same size, allow exploration)
            if rng.random() < math.exp(-1.0 / max(T, 0.1)):
                cur_max = t.max()
            else:
                t.remove(new)
                t.add(old)
        T *= cooling
    return best


def _rand_initial(D, rng, size=12):
    s = {0}
    for _ in range(size - 1):
        v = int(rng.random() ** 0.7 * D) + 1
        s.add(v)
    return sorted(s)


def construct():
    start = time.time()
    deadline = start + _DEADLINE
    rng = random.Random(42)

    best = None
    best_max = 10**6

    def consider(s):
        nonlocal best, best_max
        if s is None or not _valid_list(s):
            return
        m = max(s)
        if m < best_max:
            best = list(s)
            best_max = m

    # Phase 1: deterministic constructive search with decreasing caps.
    for cap in [40, 45, 50, 55, 60, 70, 80, 100, 130, 170, 250]:
        if time.time() > deadline - _MARGIN:
            break
        for _ in range(3):
            if time.time() > deadline - _MARGIN:
                break
            s = _constructive(cap, rng, deadline - _MARGIN)
            consider(s)

    # Phase 2: random initial + SA to discover new basins.
    for D in [45, 60, 80, 110, 150]:
        if time.time() > deadline - _MARGIN:
            break
        for _ in range(30):
            if time.time() > deadline - _MARGIN:
                break
            s = _rand_initial(D, rng, size=12)
            r = _sa_refine(list(s), rng, deadline, T0=50.0, cooling=0.995, max_iter=800)
            consider(r)

    # Phase 3: aggressive refinement on the best found.
    if best is not None:
        if time.time() < deadline - _MARGIN:
            consider(_sa_refine(list(best), rng, deadline, T0=80.0, cooling=0.998, max_iter=5000))
        if time.time() < deadline - _MARGIN:
            consider(_try_remove_elements(list(best), deadline))
        if time.time() < deadline - _MARGIN:
            consider(_local_shrink(list(best), rng, deadline))
        if time.time() < deadline - _MARGIN:
            consider(_sa_refine(list(best), rng, deadline, T0=30.0, cooling=0.997, max_iter=3000))
        if time.time() < deadline - _MARGIN:
            consider(_try_remove_elements(list(best), deadline))

    if best is None:
        # Ultimate fallback: guaranteed-valid large construction.
        a = [0]
        def diffs(ss):
            return [y - x for i, x in enumerate(ss) for y in ss[i + 1:]]
        for d in range(1, 31):
            if d in diffs(sorted(a)):
                continue
            placed = False
            for x in sorted(set(v + dd for v in a for dd in (d, -d) if v + dd >= 0) - set(a)):
                trial = sorted(a + [x])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    placed = True
                    break
            if not placed:
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    if len(set(diffs(trial))) == len(diffs(trial)):
                        a = trial
                        break
                    y += 1
        return a

    return best
# EVOLVE-BLOCK-END
