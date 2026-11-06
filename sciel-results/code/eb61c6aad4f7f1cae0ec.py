import time
from itertools import combinations

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s limit

    # Precompute base cycle edges (the 40-cycle)
    base_edges = [(i, (i + 1) % N) for i in range(N)]

    required = set(range(3, N + 1))
    
    # Precompute for each chord s in 2..38:
    #   single_lengths: lengths covered by chord s alone
    #   pair_lengths[s1][s2]: length covered by pair (s1, s2)
    chord_singles = {}
    for s in range(2, N - 1):
        lengths = set()
        l1 = s + 1
        if 3 <= l1 <= N:
            lengths.add(l1)
        l2 = N - s + 1
        if 3 <= l2 <= N:
            lengths.add(l2)
        chord_singles[s] = lengths
    
    # Pair lengths: for a < b, length = b - a + 2
    chord_pairs = {}
    for a in range(2, N - 1):
        for b in range(a + 1, N - 1):
            l = b - a + 2
            if 3 <= l <= N:
                if a not in chord_pairs:
                    chord_pairs[a] = {}
                if b not in chord_pairs:
                    chord_pairs[b] = {}
                chord_pairs[a][b] = l
                chord_pairs[b][a] = l
    
    def compute_covered(S):
        """Compute covered lengths given chord set S."""
        cov = {N}
        S_list = sorted(S)
        for s in S_list:
            cov.update(chord_singles[s])
        for i in range(len(S_list)):
            a = S_list[i]
            if a not in chord_pairs:
                continue
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                if b in chord_pairs.get(a, {}):
                    cov.add(chord_pairs[a][b])
        return cov
    
    def incremental_cover(S, add_s):
        """Compute newly covered lengths when adding chord add_s to set S."""
        new_lengths = set()
        S_list = sorted(S)
        # Single contributions from add_s
        new_lengths.update(chord_singles[add_s])
        # Pair contributions with existing chords
        if add_s in chord_pairs:
            for s in S_list:
                if s in chord_pairs[add_s]:
                    new_lengths.add(chord_pairs[add_s][s])
        return new_lengths

    candidates = list(range(2, N - 1))  # 2..38
    
    best_S = None
    best_size = float('inf')
    
    import random
    rng = random.Random(42)
    
    # Multiple greedy restarts
    for trial in range(500):
        if time.time() > deadline - 10.0:
            break
        
        S = set()
        covered = {N}
        
        while not required.issubset(covered):
            best_gain = -1
            best_s = None
            for s in candidates:
                if s in S:
                    continue
                new_cov = incremental_cover(S, s)
                gain = len(new_cov & required - covered)
                if gain > best_gain:
                    best_gain = gain
                    best_s = s
            if best_s is None or best_gain <= 0:
                break
            S.add(best_s)
            covered |= incremental_cover(S - {best_s}, best_s)
        
        if required.issubset(covered):
            if len(S) < best_size:
                best_size = len(S)
                best_S = S.copy()
                if best_size <= 8:
                    break
    
    if best_S is None:
        # Fallback: use a known good construction
        best_S = set(range(2, 39, 3))
    
    # Local search: try removals and swaps
    if time.time() < deadline - 15.0:
        improved = True
        while improved and time.time() < deadline - 15.0:
            improved = False
            S_list = sorted(best_S)
            
            # Try removals
            for i in range(len(S_list)):
                if time.time() > deadline - 15.0:
                    break
                trial_S = best_S - {S_list[i]}
                if required.issubset(compute_covered(trial_S)):
                    best_S = trial_S
                    improved = True
                    break
            
            # Try swaps: remove one, add another
            if not improved and len(S_list) >= 2:
                for i in range(len(S_list)):
                    if time.time() > deadline - 15.0:
                        break
                    for s in candidates:
                        if s in best_S:
                            continue
                        trial_S = (best_S - {S_list[i]}) | {s}
                        if len(trial_S) == len(best_S) and required.issubset(compute_covered(trial_S)):
                            best_S = trial_S
                            improved = True
                            break
                    if improved:
                        break
    
    S = best_S
    S_sorted = sorted(S)
    
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
    for s in S_sorted:
        l1 = s + 1
        if 3 <= l1 <= N and l1 not in cycles_by_len:
            cycles_by_len[l1] = list(range(s + 1))
        
        l2 = N - s + 1
        if 3 <= l2 <= N and l2 not in cycles_by_len:
            cycles_by_len[l2] = [0] + list(range(N - 1, s - 1, -1))
    
    # From pairs of chords
    for i in range(len(S_sorted)):
        for j in range(i + 1, len(S_sorted)):
            a = S_sorted[i]
            b = S_sorted[j]
            l = b - a + 2
            if 3 <= l <= N and l not in cycles_by_len:
                cycles_by_len[l] = [0] + list(range(b, a - 1, -1))
    
    # Check if all lengths are covered
    for L in range(3, N + 1):
        if L not in cycles_by_len:
            # Fallback
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
    
    for L, cyc in zip(range(3, N + 1), final_cycles):
        for i in range(L):
            a = cyc[i]
            b = cyc[(i + 1) % L]
            if a > b:
                a, b = b, a
            if (a, b) not in edge_set:
                # Fallback to a safe construction
                return {
                    'edges': base_edges + [(0, k) for k in range(2, 21)],
                    'cycles': [
                        list(range(L)) if L <= 21 else ([0] + list(range(39, 41 - L - 1, -1)) if L <= 39 else list(range(N)))
                        for L in range(3, N + 1)
                    ]
                }
    
    return {'edges': edges, 'cycles': final_cycles}
