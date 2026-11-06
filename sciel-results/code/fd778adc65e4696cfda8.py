import time
import random
import itertools

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # The "fan" model from inspirations:
    # Vertices 0..N-1. Rim edges (i, i+1 mod N).
    # Chords are of the form (0, k) for k in S, where S is a subset of {2, ..., N-2}.
    # Total edges = N (rim) + |S| (chords).
    # We want to minimize |S| such that the graph is pancyclic (has cycles of length 3..N).
    
    # Cycle types in this fan graph:
    # 1. Rim: Length N. Always present.
    # 2. Chord Short: (0, 1, 2, ..., k, 0). Length k+1. Requires k in S.
    # 3. Chord Long: (0, N-1, N-2, ..., k, 0). Length N-k+1. Requires k in S.
    # 4. Chord Pair: (0, u, u+1, ..., v, 0). Length |u-v| + 2. Requires u, v in S.
    
    def get_covered_lengths(S):
        """
        Returns a set of cycle lengths covered by the fan graph with chords S.
        """
        lengths = {N}  # Rim
        for k in S:
            lengths.add(k + 1)       # Short
            lengths.add(N - k + 1)   # Long
        
        # Pair cycles
        S_list = sorted(S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                u = S_list[i]
                v = S_list[j]
                # The cycle is 0 -> u -> ... -> v -> 0
                # Length is (v - u) + 2
                lengths.add(v - u + 2)
        
        return lengths

    def is_pancyclic(S):
        """
        Check if the fan graph with chords S has cycles of all lengths 3..N.
        """
        lengths = get_covered_lengths(S)
        for L in range(3, N + 1):
            if L not in lengths:
                return False
        return True

    def find_S(m):
        """
        Search for a set S of size m such that the fan graph is pancyclic.
        Uses greedy initialization and local search (swaps).
        """
        candidates = list(range(2, N - 1))  # 2 to 38
        
        # 1. Greedy Initialization
        # Start with empty set, add chords that cover the most new lengths.
        S = set()
        covered = {N}
        
        # We need to cover 3..N-1.
        # Greedy step: pick k that maximizes new coverage.
        # Note: We don't strictly need to fill up to m if we reach pancyclic earlier,
        # but the problem asks to minimize edges, so we should try small m first.
        # However, the search for m=8 is the target.
        
        # Let's try a few different greedy strategies or random restarts.
        
        # Strategy A: Pure Greedy
        S_greedy = set()
        covered_greedy = {N}
        for _ in range(m):
            best_k = None
            best_gain = -1
            for k in candidates:
                if k in S_greedy:
                    continue
                new_S = S_greedy | {k}
                new_covered = get_covered_lengths(new_S)
                gain = len(new_covered - covered_greedy)
                if gain > best_gain:
                    best_gain = gain
                    best_k = k
            if best_k is not None:
                S_greedy.add(best_k)
                covered_greedy = get_covered_lengths(S_greedy)
        
        if is_pancyclic(S_greedy):
            return S_greedy
        
        # Strategy B: Local Search with Random Restarts
        # We will perform random swaps to improve coverage or fix missing lengths.
        
        # Pre-compute coverage for speed? No, m is small, N is small.
        
        # Heuristic: Start with a set that covers extremes.
        # Length 3 needs k=2 (k+1=3) or k=38 (N-k+1=39, wait, 3 is k+1=3 => k=2).
        # Length 39 needs k=2 (N-k+1=39) or pair diff 37.
        # So k=2 is very strong.
        
        # Let's bias the random restarts to include 2 and 38.
        
        for restart in range(5000):
            if time.time() > deadline:
                break
            
            # Initialize S
            S = set()
            # Force include 2 and 38?
            # If m >= 2:
            if m >= 2:
                S.add(2)
                S.add(38)
            # Fill rest randomly
            remaining = [k for k in candidates if k not in S]
            if len(remaining) >= m - len(S):
                S.update(random.sample(remaining, m - len(S)))
            
            # Local Search
            # We want to maximize the number of covered lengths in 3..N.
            # Score = number of covered lengths. Target = N-2 (lengths 3..N-1).
            
            best_S = set(S)
            best_score = len(get_covered_lengths(S) & set(range(3, N)))
            
            # If we already have full coverage, return
            if best_score == N - 2:
                return S
            
            for step in range(200):
                if time.time() > deadline:
                    break
                
                # Try swapping one element in S with one not in S
                # Or just random mutation?
                # Swap is better for maintaining size m.
                
                in_S = list(S)
                out_S = [k for k in candidates if k not in S]
                
                if not in_S or not out_S:
                    break
                
                u = random.choice(in_S)
                v = random.choice(out_S)
                
                new_S = set(S)
                new_S.remove(u)
                new_S.add(v)
                
                new_covered = get_covered_lengths(new_S)
                new_score = len(new_covered & set(range(3, N)))
                
                # Accept if better, or with some probability (simulated annealing lite)
                if new_score > best_score:
                    S = new_S
                    best_S = new_S
                    best_score = new_score
                    if best_score == N - 2:
                        return S
                elif new_score == best_score:
                    # Neutral move, accept with 50% chance to escape local minima
                    if random.random() < 0.5:
                        S = new_S
                # If worse, reject (strict hill climbing for speed)
        
        return None

    # Try to find minimal m
    # Theoretical lower bound:
    # Each chord covers 2 single lengths and ~m-1 pair lengths.
    # Total lengths to cover: 37 (3..39).
    # m=6: 6*2 + 15 = 27 < 37. Impossible?
    # m=7: 7*2 + 21 = 35 < 37. Likely impossible.
    # m=8: 8*2 + 28 = 44 > 37. Possible.
    
    # Let's try m=8 first.
    for m in [8, 9, 10, 11]:
        if time.time() > deadline:
            break
        
        S = find_S(m)
        if S:
            # Verify
            if is_pancyclic(S):
                # Build result
                edges = []
                # Rim
                for i in range(N):
                    edges.append((i, (i + 1) % N))
                # Chords
                for k in S:
                    edges.append((0, k))
                
                # Generate cycles
                cycles = []
                length_map = {}
                
                # Map lengths to cycle constructors
                for k in S:
                    l1 = k + 1
                    l2 = N - k + 1
                    if l1 not in length_map:
                        length_map[l1] = ('chord_short', k)
                    if l2 not in length_map:
                        length_map[l2] = ('chord_long', k)
                
                S_list = sorted(S)
                for i in range(len(S_list)):
                    for j in range(i + 1, len(S_list)):
                        u = S_list[i]
                        v = S_list[j]
                        l = v - u + 2
                        if l not in length_map:
                            length_map[l] = ('chord_pair', u, v)
                
                length_map[N] = ('rim',)
                
                # Check if all lengths 3..N are in length_map
                valid = True
                for L in range(3, N + 1):
                    if L not in length_map:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Construct cycles
                for L in range(3, N + 1):
                    typ = length_map[L]
                    if typ[0] == 'rim':
                        cycles.append(list(range(N)))
                    elif typ[0] == 'chord_short':
                        k = typ[1]
                        # 0, 1, ..., k
                        cycles.append([0] + list(range(1, k + 1)))
                    elif typ[0] == 'chord_long':
                        k = typ[1]
                        # 0, N-1, ..., k
                        cycles.append([0] + list(range(N - 1, k - 1, -1)))
                    elif typ[0] == 'chord_pair':
                        u, v = typ[1], typ[2]
                        # 0, u, u+1, ..., v
                        cycles.append([0, u] + list(range(u + 1, v + 1)))
                
                # Final verification
                edge_set = set((min(e), max(e)) for e in edges)
                for cycle in cycles:
                    for i in range(len(cycle)):
                        a = cycle[i]
                        b = cycle[(i + 1) % len(cycle)]
                        if (min(a, b), max(a, b)) not in edge_set:
                            valid = False
                            break
                        if len(set(cycle)) != len(cycle):
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    return {'edges': edges, 'cycles': cycles}
    
    # Fallback: Use the current program's approach if fan fails?
    # Or just return a valid (though suboptimal) solution.
    # The current program's greedy cover usually finds a solution.
    # Let's implement a quick fallback using the original logic but simplified.
    
    base_edges = [(i, (i + 1) % N) for i in range(N)]
    base_edge_set = set((min(e), max(e)) for e in base_edges)
    
    # Greedy cover on chords
    target_lengths = set(range(3, N))
    chord_coverages = []
    for u in range(N):
        for v in range(u + 1, N):
            if (u, v) in base_edge_set:
                continue
            len1 = v - u + 1
            len2 = N - (v - u) + 1
            covered = set()
            if 3 <= len1 <= 39:
                covered.add(len1)
            if 3 <= len2 <= 39:
                covered.add(len2)
            if covered:
                chord_coverages.append(((u, v), covered))
    
    uncovered = set(target_lengths)
    selected = set()
    
    while uncovered:
        best_idx = None
        best_count = 0
        for idx, (chord, covered) in enumerate(chord_coverages):
            if idx in selected:
                continue
            new_cov = len(covered & uncovered)
            if new_cov > best_count:
                best_count = new_cov
                best_idx = idx
        
        if best_idx is None or best_count == 0:
            break
        selected.add(best_idx)
        uncovered -= chord_coverages[best_idx][1]
    
    all_edges = list(base_edges)
    for idx in selected:
        u, v = chord_coverages[idx][0]
        edge = (min(u, v), max(u, v))
        if edge not in base_edge_set:
            all_edges.append(edge)
    
    edge_set = set((min(e), max(e)) for e in all_edges)
    
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            cycles.append(list(range(N)))
            continue
        
        found = False
        for u in range(N):
            for v in range(u + 1, N):
                len1 = v - u + 1
                len2 = N - (v - u) + 1
                if len1 == L or len2 == L:
                    edge = (min(u, v), max(u, v))
                    if edge in edge_set:
                        if len1 == L:
                            cycle = list(range(u, v + 1))
                        else:
                            cycle = [u]
                            cur = (u - 1) % N
                            while cur != v:
                                cycle.append(cur)
                                cur = (cur - 1) % N
                            cycle.append(v)
                        
                        valid = True
                        for i in range(len(cycle)):
                            a = cycle[i]
                            b = cycle[(i + 1) % len(cycle)]
                            if (min(a, b), max(a, b)) not in edge_set:
                                valid = False
                                break
                        if valid and len(set(cycle)) == L:
                            cycles.append(cycle)
                            found = True
                            break
            if found:
                break
        
        if not found:
            # Fallback: try all possible cycles of length L
            for u in range(N):
                cycle = [(u + i) % N for i in range(L)]
                valid = True
                for i in range(L):
                    a = cycle[i]
                    b = cycle[(i + 1) % L]
                    if (min(a, b), max(a, b)) not in edge_set:
                        valid = False
                        break
                if valid and len(set(cycle)) == L:
                    cycles.append(cycle)
                    found = True
                    break
            if not found:
                cycles.append(list(range(L)))
    
    # Final verification
    for cycle in cycles:
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            assert (min(u, v), max(u, v)) in edge_set, f"Edge ({u},{v}) not in graph"
        assert len(set(cycle)) == len(cycle), f"Cycle has duplicate vertices: {cycle}"
    
    return {'edges': all_edges, 'cycles': cycles}
