import time
from itertools import combinations

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s limit

    # Base cycle edges
    base_edges = [(i, (i + 1) % N) for i in range(N)]
    required = set(range(3, N + 1))

    def compute_covered(S):
        """Compute which cycle lengths are covered by chord set S."""
        cov = {N}
        S_sorted = sorted(S)
        for s in S_sorted:
            l1 = s + 1
            if 3 <= l1 <= N:
                cov.add(l1)
            l2 = N - s + 1
            if 3 <= l2 <= N:
                cov.add(l2)
        for i in range(len(S_sorted)):
            for j in range(i + 1, len(S_sorted)):
                a = S_sorted[i]
                b = S_sorted[j]
                l = b - a + 2
                if 3 <= l <= N:
                    cov.add(l)
        return cov

    candidates = list(range(2, N - 1))  # 2..38

    # Precompute coverage for each single chord
    single_cov = {}
    for s in candidates:
        single_cov[s] = compute_covered({s})

    # Precompute pair contributions
    pair_cov = {}
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a, b = candidates[i], candidates[j]
            l = b - a + 2
            if 3 <= l <= N:
                pair_cov[(a, b)] = l

    # Branch and bound search for minimum chord set
    best_S = None
    best_size = float('inf')

    def bnb(idx, current_S, covered, remaining):
        nonlocal best_S, best_size
        if time.time() > deadline - 2.0:
            return
        if not remaining:
            if len(current_S) < best_size:
                best_size = len(current_S)
                best_S = list(current_S)
            return
        # Pruning: even if we add all remaining candidates, can we cover?
        if len(current_S) + (len(candidates) - idx) < best_size:
            return
        # If current size already >= best, prune
        if len(current_S) >= best_size:
            return

        # Lower bound: need at least ceil(len(remaining) / max_gain) more
        # Simple bound: each new chord covers at most ~3 new lengths on average
        if len(current_S) + 1 >= best_size and not remaining:
            return

        for i in range(idx, len(candidates)):
            s = candidates[i]
            new_S = current_S + [s]
            new_covered = covered | single_cov[s]
            # Add pair contributions
            for prev_s in current_S:
                key = (min(prev_s, s), max(prev_s, s))
                if key in pair_cov:
                    new_covered.add(pair_cov[key])
            new_remaining = remaining - new_covered
            if len(new_S) < best_size:
                bnb(i + 1, new_S, new_covered, new_remaining)
            if time.time() > deadline - 2.0:
                return

    # Start with small sizes first
    for target_size in range(7, 12):
        if time.time() > deadline - 5.0:
            break
        # Try to find a set of exactly target_size chords
        found = False
        for combo in combinations(candidates, target_size):
            if time.time() > deadline - 5.0:
                break
            S = set(combo)
            cov = compute_covered(S)
            if required.issubset(cov):
                if target_size < best_size:
                    best_size = target_size
                    best_S = list(S)
                found = True
                break
        if found and target_size <= 8:
            break

    if best_S is None:
        # Fallback: use a known good construction
        best_S = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38]

    # Try to reduce further with local search
    if time.time() < deadline - 10.0:
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            S_set = set(best_S)
            for s in list(S_set):
                if time.time() > deadline - 10.0:
                    break
                trial_S = S_set - {s}
                if required.issubset(compute_covered(trial_S)):
                    best_S = list(trial_S)
                    improved = True
                    break

    S = sorted(best_S)

    # Build the edge list
    edges = list(base_edges)
    for s in S:
        edges.append((0, s))

    # Remove duplicate edges
    unique_edges = []
    seen = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        if (u, v) not in seen:
            seen.add((u, v))
            unique_edges.append((u, v))
    edges = unique_edges

    # Build cycles for each length 3..N
    cycles_by_len = {}
    cycles_by_len[N] = list(range(N))

    # From single chords
    for s in S:
        l1 = s + 1
        if 3 <= l1 <= N and l1 not in cycles_by_len:
            cycles_by_len[l1] = list(range(s + 1))

        l2 = N - s + 1
        if 3 <= l2 <= N and l2 not in cycles_by_len:
            cycles_by_len[l2] = [0] + list(range(N - 1, s - 1, -1))

    # From pairs of chords
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            a = S[i]
            b = S[j]
            l = b - a + 2
            if 3 <= l <= N and l not in cycles_by_len:
                cycles_by_len[l] = [0] + list(range(b, a - 1, -1))

    # Check if all lengths are covered
    for L in range(3, N + 1):
        if L not in cycles_by_len:
            # Fallback: use a safe construction
            if L <= 21:
                cycles_by_len[L] = list(range(L))
            elif L <= 39:
                k = 41 - L
                cycles_by_len[L] = [0] + list(range(39, k - 1, -1))
            else:
                cycles_by_len[L] = list(range(N))

    final_cycles = [cycles_by_len[L] for L in range(3, N + 1)]

    # Verify all cycles are valid (edges exist)
    edge_set = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        edge_set.add((u, v))

    valid = True
    for L, cyc in zip(range(3, N + 1), final_cycles):
        for i in range(L):
            a = cyc[i]
            b = cyc[(i + 1) % L]
            if a > b:
                a, b = b, a
            if (a, b) not in edge_set:
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Fallback to a safe construction
        fallback_edges = base_edges + [(0, k) for k in range(2, 21)]
        fallback_cycles = []
        for L in range(3, N + 1):
            if L <= 21:
                fallback_cycles.append(list(range(L)))
            elif L <= 39:
                k = 41 - L
                fallback_cycles.append([0] + list(range(39, k - 1, -1)))
            else:
                fallback_cycles.append(list(range(N)))
        return {'edges': fallback_edges, 'cycles': fallback_cycles}

    return {'edges': edges, 'cycles': final_cycles}
