import time
import numpy as np

_DEADLINE = 120.0
_TARGET = set(range(1, 31))

def _valid(a):
    s = sorted(a)
    if 0 not in s or len(s) > 32:
        return False
    diffs = [s[j] - s[i] for i in range(len(s)) for j in range(i+1, len(s))]
    if len(set(diffs)) != len(diffs):
        return False
    return all(d in diffs for d in _TARGET)

def _greedy():
    a = [0]
    def diffs_of(s):
        s = sorted(s)
        return [s[j]-s[i] for i in range(len(s)) for j in range(i+1, len(s))]
    for d in range(1, 31):
        if d in diffs_of(a):
            continue
        for cand in range(1, 10**6):
            trial = sorted(a + [cand])
            dd = diffs_of(trial)
            if len(set(dd)) == len(dd) and d in dd:
                a = trial
                break
    return a

def _search(t0, init_best):
    best = [init_best]  # (diameter, list)
    def dfs(a, used, covered, max_bound):
        if time.time() - t0 > _DEADLINE - 5.0:
            return
        if len(covered) == 30:
            d = a[-1]
            if d < best[0][0]:
                best[0] = (d, list(a))
            return
        if len(a) >= 32:
            return
        lo = a[-1] + 1
        ub = best[0][0] - 1
        if lo > ub:
            return
        arr = np.array(a, dtype=np.int64)
        x = lo
        while x <= ub:
            diffs = np.abs(x - arr)
            # distinct among themselves
            if len(set(diffs.tolist())) != len(diffs):
                x += 1
                continue
            # not in used
            if any(int(dv) in used for dv in diffs):
                x += 1
                continue
            newused = used | set(int(dv) for dv in diffs)
            newcovered = covered | set(int(dv) for dv in diffs if int(dv) <= 30)
            a.append(int(x))
            dfs(a, newused, newcovered, ub)
            a.pop()
            x += 1
            if time.time() - t0 > _DEADLINE - 5.0:
                return
            if best[0][0] < ub:
                ub = best[0][0] - 1
    dfs([0], set(), set(), init_best[0] - 1)
    return best[0][1]

_t0 = time.time()
_g = _greedy()
_gd = max(_g)
_best = _search(_t0, (_gd, _g))
if not _valid(_best):
    _best = _g

def construct():
    return list(_best)
