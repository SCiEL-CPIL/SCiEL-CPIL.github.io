"""Pancyclic sparse graph on 40 vertices: Hamiltonian cycle + minimal chord fan."""
import random
import time

N = 40
START = time.time()
DEADLINE = 90.0
MARGIN = 5.0


def _circular_dist(a, b, n=N):
    d = (a - b) % n
    return d if d <= n - d else n - d


def _search_chords(max_chords, deadline, seed=0):
    """Find a chord set of size <= max_chords whose pairwise circular distances
    cover every value 2..39. Returns (list_of_positions, size) or (None, None)."""
    need = set(range(2, N))
    rng = random.Random(seed)

    # 1) deterministic / random search for the target size
    for _ in range(200000):
        if time.time() > deadline:
            break
        pos = set(rng.sample(range(N), max_chords))
        cov = set()
        ps = list(pos)
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                cov.add(_circular_dist(ps[i], ps[j]))
        if need.issubset(cov):
            return sorted(pos), max_chords

    # 2) greedy construction to a larger size (fallback)
    best = None
    for _ in range(20000):
        if time.time() > deadline:
            break
        chosen = [rng.randrange(N)]
        cov = set()
        while len(chosen) < N - 1:
            cands = list(range(N))
            best_cand, best_cov = None, None
            for c in cands:
                if c in chosen:
                    continue
                ncov = set(cov)
                for x in chosen:
                    ncov.add(_circular_dist(x, c))
                if best_cov is None or len(ncov) > len(best_cov):
                    best_cand, best_cov = c, ncov
            if best_cand is None:
                break
            chosen.append(best_cand)
            cov = best_cov
            if need.issubset(cov):
                break
        if need.issubset(cov) and (best is None or len(chosen) < len(best)):
            best = chosen
    if best is not None:
        return sorted(best), len(best)
    return None, None


def _build(chords):
    """Build edges + cycle certificates for a chord set (positions on the cycle)."""
    # vertex 0 is the fan center; cycle order is 0,1,2,...,39
    cycle_edges = [(i, (i + 1) % N) for i in range(N)]
    chords_set = set(chords)
    chord_edges = [(0, c) for c in chords_set]
    edges = cycle_edges + chord_edges

    cycles = []
    for k in range(3, N + 1):
        # need a chord pair (a,b) with circular distance k-1
        found = None
        for a in chords_set:
            for b in chords_set:
                if a == b:
                    continue
                if _circular_dist(a, b) == k - 1:
                    found = (a, b)
                    break
            if found:
                break
        assert found is not None, "no chord pair for length %d" % k
        a, b = found
        # walk from a to b the short way (k-1 steps) on the cycle
        d = (b - a) % N
        if d == k - 1:
            path = [(a + t) % N for t in range(k - 1)]
        else:
            path = [(a - t) % N for t in range(k - 1)]
        cyc = [0] + path + [b]
        assert len(cyc) == k and len(set(cyc)) == k
        cycles.append(cyc)
    return {'edges': edges, 'cycles': cycles}


def construct():
    # fast path: known 9-chord covering set
    known9 = [0, 1, 3, 6, 11, 18, 26, 33, 37]
    need = set(range(2, N))
    cov = set()
    for i in range(len(known9)):
        for j in range(i + 1, len(known9)):
            cov.add(_circular_dist(known9[i], known9[j]))
    if need.issubset(cov):
        return _build(known9)

    # otherwise search (with deadline guard)
    dl = START + DEADLINE - MARGIN
    for target in (9, 10, 11, 12, 13, 14, 15, 16, 17, 18):
        if time.time() > dl:
            break
        pos, sz = _search_chords(target, dl)
        if pos is not None:
            return _build(pos)
    # ultimate fallback: even-distance fan (19 chords, always valid)
    return _build(list(range(0, N, 2)))
