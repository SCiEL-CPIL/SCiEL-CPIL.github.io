import time
from itertools import combinations

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s limit

    # Precompute base cycle edges (the 40-cycle)
    base_edges = [(i, (i + 1) % N) for i in range(N)]

    # We add chords from vertex 0 to some subset S of {2, ..., N-2}
    # A chord (0, s) enables:
    #   - cycle of length s+1: [0, 1, ..., s]
    #   - cycle of length N-s+1: [0, N-1, N-2, ..., s]
    #   - for two chords a < b: cycle of length b-a+2: [0, b, b-1, ..., a]
    
    required = set(range(3, N + 1))
    
    def compute_covered(S):
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

    # Try to find a small set of chords that covers all required lengths
    # Minimum possible is 8 chords (since 2*8 + C(8,2) + 1 = 45 >= 38)
    # Let's search for optimal sets using a more systematic approach
    
    candidates = list(range(2, N - 1))  # 2..38
    
    # First try greedy with multiple restarts and different tie-breaking
    best_S = None
    best_size = float('inf')
    
    # Greedy with randomization
    import random
    rng = random.Random(42)
    
    for trial in range(200):
        if time.time() > deadline - 10.0:
            break
        
        S = set()
        covered = {N}
        
        while not required.issubset(covered):
            # Compute marginal gain for each candidate
            gains = []
            for s in candidates:
                if s in S:
                    continue
                new_cov = compute_covered(S | {s})
                gain = len(new_cov & required) - len(covered & required)
                gains.append((gain, s))
            
            if not gains:
                break
            
            # Sort by gain descending, break ties randomly
            gains.sort(key=lambda x: (-x[0], rng.random()))
            
            # Pick the best one
            best_gain, best_s = gains[0]
            if best_gain <= 0:
                break
            
            S.add(best_s)
            covered = compute_covered(S)
        
        if required.issubset(covered):
            if len(S) < best_size:
                best_size = len(S)
                best_S = S.copy()
                if best_size <= 8:
                    break  # Can't do better than 8
    
    if best_S is None:
        # Fallback: use a known good construction
        # Chords at 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38
        best_S = set(range(2, 39, 3))
    
    # Try to reduce further with local search
    if time.time() < deadline - 15.0:
        improved = True
        while improved and time.time() < deadline - 15.0:
            improved = False
            S_list = sorted(best_S)
            for i in range(len(S_list)):
                if time.time() > deadline - 15.0:
                    break
                # Try removing S_list[i]
                trial_S = best_S - {S_list[i]}
                if required.issubset(compute_covered(trial_S)):
                    best_S = trial_S
                    improved = True
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
