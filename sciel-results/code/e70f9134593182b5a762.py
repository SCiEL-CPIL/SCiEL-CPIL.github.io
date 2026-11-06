"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 18
FULL = 1 << N

# EVOLVE-BLOCK-START

def _build_full():
    bits = np.arange(FULL, dtype=np.uint16)[:, None]
    return ((bits >> np.arange(N)[None, :]) & 1).astype(np.uint8)

def _batcher(n):
    comps = []
    def oe_merge(idx):
        m = len(idx)
        if m <= 1:
            return
        if m == 2:
            a, b = idx[0], idx[1]
            comps.append((a, b) if a < b else (b, a))
            return
        half = m // 2
        oe_merge([idx[k] for k in range(0, half, 2)] + [idx[k] for k in range(half, m, 2)])
        oe_merge([idx[k] for k in range(1, half, 2)] + [idx[k] for k in range(half + 1, m, 2)])
        for k in range(half):
            a, b = idx[2 * k], idx[2 * k + 1]
            comps.append((a, b) if a < b else (b, a))
    def sort_range(lo, hi):
        m = hi - lo
        if m <= 1:
            return
        if m % 2 == 0:
            mid = lo + m // 2
            sort_range(lo, mid)
            sort_range(mid, hi)
            oe_merge(list(range(lo, hi)))
        else:
            sort_range(lo, hi - 1)
            for i in range(hi - 2, lo - 1, -1):
                comps.append((i, hi - 1))
    sort_range(0, n)
    return comps

def _odd_even():
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

def _apply(net, X, out):
    out[:] = X
    for i, j in net:
        t = np.minimum(out[:, i], out[:, j])
        out[:, j] = np.maximum(out[:, i], out[:, j])
        out[:, i] = t
    return out

def _bad_count(Y):
    return int(np.sum(np.any(Y[:, :-1] > Y[:, 1:], axis=1)))

def construct():
    start = time.time()
    DEADLINE = 80.0
    Xfull = _build_full()
    rng = np.random.default_rng(12345)
    sub_size = 16384
    idx = rng.choice(FULL, size=sub_size, replace=False)
    Xsub = Xfull[idx]
    scratch = np.empty_like(Xsub)
    scratch_full = np.empty_like(Xfull)

    def full_valid(net):
        _apply(net, Xfull, scratch_full)
        return _bad_count(scratch_full) == 0

    def sub_bad(net):
        _apply(net, Xsub, scratch)
        return _bad_count(scratch)

    # seed
    net = _batcher(N)
    if not full_valid(net):
        net = _odd_even()
    assert full_valid(net)
    best = list(net)
    best_cost = len(best)

    def greedy(net, time_cap):
        # single-direction passes
        for _ in range(50):
            if time.time() > time_cap:
                break
            changed = False
            pos = 0
            while pos < len(net):
                trial = net[:pos] + net[pos + 1:]
                if full_valid(trial):
                    net = trial
                    changed = True
                else:
                    pos += 1
            if not changed:
                break
        return net

    net = greedy(net, start + 22.0)
    best = list(net); best_cost = len(net)

    # SA
    current = list(net)
    cur_E = sub_bad(current)
    cur_cost = cur_E * 1000 + len(current)
    last_improve = 0
    it = 0
    T0 = 40.0
    Tend = 0.3
    while time.time() < DEADLINE - 4.0:
        it += 1
        frac = min(1.0, it / 18000.0)
        T = T0 * (Tend / T0) ** frac
        r = rng.random()
        new = list(current)
        if r < 0.35:
            if len(new) > 1:
                new.pop(int(rng.integers(len(new))))
            else:
                continue
        elif r < 0.55:
            if len(new) > 1:
                p = int(rng.integers(len(new))); q = int(rng.integers(len(new)))
                new[p], new[q] = new[q], new[p]
            else:
                continue
        elif r < 0.75:
            if len(new) > 1:
                p = int(rng.integers(len(new))); q = int(rng.integers(len(new)))
                if p != q:
                    new.insert(q, new.pop(p))
                else:
                    continue
            else:
                continue
        elif r < 0.85:
            i = int(rng.integers(N)); j = int(rng.integers(N))
            if i == j:
                continue
            if i > j: i, j = j, i
            new.insert(int(rng.integers(len(new) + 1)), (i, j))
        else:
            i = int(rng.integers(N)); j = int(rng.integers(N))
            if i == j:
                continue
            if i > j: i, j = j, i
            p = int(rng.integers(len(new)))
            new[p] = (i, j)
        new_E = sub_bad(new)
        new_cost = new_E * 1000 + len(new)
        delta = new_cost - cur_cost
        if delta <= 0 or rng.random() < np.exp(-delta / max(T, 1e-9)):
            current = new
            cur_E = new_E
            cur_cost = new_cost
            if cur_E == 0 and full_valid(current):
                if len(current) < best_cost:
                    best_cost = len(current)
                    best = list(current)
                    last_improve = it
        # periodic greedy
        if it % 2500 == 0 and cur_E == 0:
            pos = 0
            while pos < len(current):
                if time.time() > DEADLINE - 4.0:
                    break
                trial = current[:pos] + current[pos + 1:]
                if full_valid(trial):
                    current = trial
                    cur_E = 0
                    cur_cost = len(current)
                    if len(current) < best_cost:
                        best_cost = len(current)
                        best = list(current)
                        last_improve = it
                else:
                    pos += 1
        # restart if stuck
        if it - last_improve > 6000:
            current = list(best)
            cur_E = sub_bad(current)
            cur_cost = cur_E * 1000 + len(current)
            last_improve = it

    # final greedy on best
    if time.time() < DEADLINE - 1.0:
        b = greedy(best, DEADLINE - 1.0)
        if len(b) < best_cost:
            best = b; best_cost = len(best)

    return list(best)

# EVOLVE-BLOCK-END
