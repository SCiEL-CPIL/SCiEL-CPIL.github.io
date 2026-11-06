import time
import random
from itertools import combinations

N = 40
REQUIRED = list(range(3, N + 1))
REQUIRED_SET = set(REQUIRED)
CANDIDATES = list(range(2, N - 1))
NUM_CANDIDATES = len(CANDIDATES)

# Precompute single-chord coverage: chord k covers lengths k+1 and N-k+1
SINGLE = [set() for _ in range(NUM_CANDIDATES)]
for i, k in enumerate(CANDIDATES):
    l1 = k + 1
    l2 = N - k + 1
    if 3 <= l1 <= N:
        SINGLE[i].add(l1)
    if 3 <= l2 <= N:
        SINGLE[i].add(l2)

# Precompute pair coverage: pair (i,j) covers length |CANDIDATES[i]-CANDIDATES[j]|+2
PAIR = [[0] * NUM_CANDIDATES for _ in range(NUM_CANDIDATES)]
for i in range(NUM_CANDIDATES):
    for j in range(i + 1, NUM_CANDIDATES):
        l = abs(CANDIDATES[i] - CANDIDATES[j]) + 2
        PAIR[i][j] = l
        PAIR[j][i] = l


def _full_coverage(idx_set):
    """Return the set of all covered lengths for a set of candidate indices."""
    cov = {N}
    for i in idx_set:
        cov.update(SINGLE[i])
    idx_list = sorted(idx_set)
    for a, b in combinations(idx_list, 2):
        l = PAIR[a][b]
        if 3 <= l <= N:
            cov.add(l)
    return cov


def _build_solution(S):
    """Build edges and cycles from chord set S (list of actual vertex indices).
    Returns dict or None if not all lengths are covered."""
    S = sorted(S)
    edges = []
    seen = set()
    for i in range(N):
        u, v = i, (i + 1) % N
        if u > v:
            u, v = v, u
        if (u, v) not in seen:
            seen.add((u, v))
            edges.append((u, v))
    for k in S:
        u, v = 0, k
        if u > v:
            u, v = v, u
        if (u, v) not in seen:
            seen.add((u, v))
            edges.append((u, v))

    length_map = {N: ('rim',)}
    for k in S:
        l1 = k + 1
        l2 = N - k + 1
        if l1 not in length_map and 3 <= l1 <= N:
            length_map[l1] = ('chord_short', k)
        if l2 not in length_map and 3 <= l2 <= N:
            length_map[l2] = ('chord_long', k)
    for u, v in combinations(S, 2):
        l = abs(u - v) + 2
        if l not in length_map and 3 <= l <= N:
            length_map[l] = ('chord_pair', u, v)

    for L in range(3, N + 1):
        if L not in length_map:
            return None

    cycles = []
    for L in range(3, N + 1):
        e = length_map[L]
        t = e[0]
        if t == 'rim':
            cycles.append(list(range(N)))
        elif t == 'chord_short':
            k = e[1]
            cycles.append(list(range(0, k + 1)))
        elif t == 'chord_long':
            k = e[1]
            cycles.append([0] + list(range(N - 1, k - 1, -1)))
        else:  # chord_pair
            u, v = e[1], e[2]
            if u > v:
                u, v = v, u
            cycles.append([0, v] + list(range(v - 1, u - 1, -1)))
    return {'edges': edges, 'cycles': cycles}


def _search_min_chords(deadline):
    """Branch-and-bound exact-cover search for the minimum chord set."""
    all_cov = _full_coverage(set(range(NUM_CANDIDATES)))
    if not REQUIRED_SET.issubset(all_cov):
        return None

    best = {'size': 10, 'set': None}

    def dfs(covered, chosen, depth):
        if time.time() > deadline:
            return
        if len(chosen) + 1 >= best['size']:
            return
        if REQUIRED_SET.issubset(covered):
            if len(chosen) < best['size']:
                best['size'] = len(chosen)
                best['set'] = set(chosen)
            return
        # Candidates that add at least one needed length
        needed = REQUIRED_SET - covered
        useful = [i for i in range(NUM_CANDIDATES)
                  if (SINGLE[i] & needed) or any(
                      PAIR[i][j] in needed for j in chosen)]
        if not useful:
            return
        # Order by marginal gain (desc), tie-break deterministically
        scored = []
        for i in useful:
            gain = 0
            for L in SINGLE[i]:
                if L in needed:
                    gain += 1
            for j in chosen:
                l = PAIR[i][j]
                if l in needed:
                    gain += 1
            scored.append((gain, i))
        scored.sort(key=lambda x: (-x[0], x[1]))
        for gain, i in scored:
            if gain <= 0:
                break
            dfs(covered | SINGLE[i], chosen + [i], depth + 1)

    dfs({N}, [], 0)
    return best['set']


def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin before 90s limit

    # Phase 1: exact-cover branch-and-bound for the minimum chord set
    best_set = _search_min_chords(deadline - 5.0)

    if best_set is None:
        # Fallback: random search for a small chord set
        rng = random.Random(42)
        best_set = None
        for m in [7, 8, 9]:
            if time.time() > deadline - 5.0:
                break
            for _ in range(200000):
                if time.time() > deadline - 5.0:
                    break
                idx = set(rng.sample(range(NUM_CANDIDATES), m))
                if REQUIRED_SET.issubset(_full_coverage(idx)):
                    best_set = idx
                    break
            if best_set is not None:
                break

    if best_set is None:
        # Guaranteed fallback: fan graph
        edges = [(0, i) for i in range(1, N)] + [(i, i + 1) for i in range(1, N - 1)]
        cycles = [list(range(L)) for L in range(3, N + 1)]
        return {'edges': edges, 'cycles': cycles}

    S = [CANDIDATES[i] for i in best_set]
    result = _build_solution(S)
    if result is None:
        # Should not happen; safe fallback
        edges = [(0, i) for i in range(1, N)] + [(i, i + 1) for i in range(1, N - 1)]
        cycles = [list(range(L)) for L in range(3, N + 1)]
        return {'edges': edges, 'cycles': cycles}

    # Phase 2: polish pass - try to remove any redundant chords
    S_set = set(S)
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for k in sorted(S_set):
            trial = S_set - {k}
            if REQUIRED_SET.issubset(_full_coverage(trial)):
                S_set = trial
                improved = True
                break

    S_final = [CANDIDATES[i] for i in S_set]
    final = _build_solution(S_final)
    if final is not None:
        return final
    return result


if __name__ == '__main__':
    r = construct()
    print('edges:', len(r['edges']))
    print('cycles:', len(r['cycles']))
    # quick validity check
    es = set()
    for u, v in r['edges']:
        if u > v:
            u, v = v, u
        es.add((u, v))
    for cyc in r['cycles']:
        for i in range(len(cyc)):
            a, b = cyc[i], cyc[(i + 1) % len(cyc)]
            if a > b:
                a, b = b, a
            assert (a, b) in es, 'missing edge'
    print('valid')
