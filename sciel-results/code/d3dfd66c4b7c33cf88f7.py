"""Suite task: 12 distinct positive integers, all 4096 subset sums distinct, minimize max."""
import time
import random

_N = 12
_MAX_BOUND = 1048576


def _subset_sums(seq):
    sums = [0]
    for v in seq:
        sums += [s + v for s in sums]
    return sums


def _valid(seq):
    if len(seq) != _N:
        return False
    if any(v <= 0 for v in seq):
        return False
    if len(set(seq)) != _N:
        return False
    return len(set(_subset_sums(seq))) == (1 << _N)


def _complete(seq):
    s = 0
    for v in seq:
        if v > s + 1:
            return False
        s += v
    return True


def _polish(seq, deadline, rng):
    best = list(seq)
    best_max = max(best)
    while time.time() < deadline - 1.0 and best_max > 1:
        progress = False
        for i in range(_N):
            if time.time() >= deadline - 1.0:
                break
            hi = best[i] - 1
            prefix = sum(best[:i])
            hi = min(hi, prefix + 1)
            if hi < 1:
                continue
            others = set(best) - {best[i]}
            cands = list(range(1, hi + 1))
            rng.shuffle(cands)
            for v in cands:
                if v in others:
                    continue
                trial = best[:i] + [v] + best[i + 1:]
                if _complete(trial):
                    best = trial
                    best_max = max(best)
                    progress = True
                    break
            if progress:
                break
        if not progress:
            break
    return best


def _build_complete(caps, rng):
    """Build a complete sequence where element i is capped by caps[i].
    Greedily choose the smallest unused value <= s+1 and <= caps[i]."""
    s = 0
    used = set()
    seq = []
    for i in range(_N):
        cap = caps[i]
        v = 1
        while v in used:
            v += 1
        if v > s + 1 or v > cap:
            return None
        seq.append(v)
        used.add(v)
        s += v
    return seq


def construct():
    start = time.time()
    deadline = start + 85.0
    rng = random.Random(98765)

    best = [1 << i for i in range(_N)]
    best_max = max(best)

    # Greedy complete (max 2047).
    g = _build_complete([_MAX_BOUND] * _N, rng)
    if g and _valid(g):
        gp = _polish(g, deadline, rng)
        if _valid(gp) and max(gp) < best_max:
            best = gp
            best_max = max(best)

    # Randomized complete sequences with decreasing caps on the last element.
    cap = best_max
    while cap > 1 and time.time() < deadline - 6.0:
        # caps: first 11 free (large), last = cap
        caps = [_MAX_BOUND] * (_N - 1) + [cap]
        found = None
        for attempt in range(3000):
            if time.time() >= deadline - 6.0:
                break
            seq = _build_complete(caps, rng)
            if seq is None:
                continue
            if max(seq) > cap:
                continue
            if not _valid(seq):
                continue
            if found is None or max(seq) < max(found):
                found = seq
                if max(seq) == cap:
                    break
        if found is not None and max(found) < best_max:
            best = found
            best_max = max(best)
        # polish
        if time.time() < deadline - 3.0:
            bp = _polish(best, deadline, rng)
            if _valid(bp) and max(bp) < best_max:
                best = bp
                best_max = max(best)
        cap = min(cap, best_max)
        if cap <= 1:
            break

    # Final polish.
    if time.time() < deadline - 0.5:
        bp = _polish(best, deadline, rng)
        if _valid(bp) and max(bp) < best_max:
            best = bp
            best_max = max(best)

    if not _valid(best):
        best = [1 << i for i in range(_N)]
    return best
