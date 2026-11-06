import time
import random
import itertools

N = 40

def construct():
    start = time.time()
    deadline = start + 75.0  # Safety margin before 90s limit

    # Base cycle: 0-1-2-...-39-0 (the rim)
    # Rim edges
    rim_edges = [(i, (i + 1) % N) for i in range(N)]
    rim_edge_set = set()
    for (u, v) in rim_edges:
        rim_edge_set.add((min(u, v), max(u, v)))

    # All possible chords (non-rim edges)
    all_chords = []
    for u in range(N):
        for v in range(u + 1, N):
            if (u, v) in rim_edge_set:
                continue
            all_chords.append((u, v))

    # Precompute: for each chord, the cycle lengths it creates with the rim
    # Chord (u,v): creates cycle of length (v-u+1) going one way and (N-(v-u)+1) going the other
    # But also, chords can create cycles that don't use the full rim arc.
    # Actually, a chord (u,v) creates exactly two cycles with the rim:
    #   Cycle 1: u, u+1, ..., v, u (length v-u+1)
    #   Cycle 2: u, u-1, ..., v, u (length N-(v-u)+1)
    # But chords can also combine with each other to form cycles.
    
    # Key insight: if we have chords (a,b) and (c,d) where the arcs overlap appropriately,
    # we can form cycles using portions of both chords.
    
    # For a simpler approach: a chord (u,v) with distance d = v-u creates:
    # - A cycle of length d+1 (the short arc + chord)
    # - A cycle of length N-d+1 (the long arc + chord)
    
    # To get all lengths 3..N, we need to cover lengths 3..N-1 with chords (N is the rim).
    
    chord_covers = {}
    for (u, v) in all_chords:
        d = v - u
        l1 = d + 1
        l2 = N - d + 1
        covers = set()
        if 3 <= l1 <= N:
            covers.add(l1)
        if 3 <= l2 <= N:
            covers.add(l2)
        chord_covers[(u, v)] = covers

    # We need to cover lengths 3..N-1
    target_lengths = set(range(3, N))

    def build_cycles(chords):
        """Given a set of chords, try to construct cycles of every length 3..N."""
        chords = list(set(chords))
        
        # Edge set
        edge_set = set(rim_edge_set)
        for (u, v) in chords:
            edge_set.add((min(u, v), max(u, v)))
        
        # For each length L, find a cycle
        length_to_cycle = {N: list(range(N))}  # Rim cycle
        
        # For each chord, it gives us two candidate cycles
        chord_cycles = {}
        for (u, v) in chords:
            d = v - u
            # Short arc cycle: u, u+1, ..., v
            short_cyc = list(range(u, v + 1))
            # Verify it's a valid cycle (all edges exist, distinct vertices)
            valid = len(short_cyc) == len(set(short_cyc))
            if valid:
                for i in range(len(short_cyc)):
                    a = short_cyc[i]
                    b = short_cyc[(i + 1) % len(short_cyc)]
                    if (min(a, b), max(a, b)) not in edge_set:
                        valid = False
                        break
            if valid:
                L = len(short_cyc)
                if L not in length_to_cycle and 3 <= L <= N:
                    length_to_cycle[L] = short_cyc
            
            # Long arc cycle: u, u-1, ..., v (wrapping around)
            long_cyc = [u]
            cur = (u - 1) % N
            while cur != v:
                long_cyc.append(cur)
                cur = (cur - 1) % N
            long_cyc.append(v)
            
            valid = len(long_cyc) == len(set(long_cyc))
            if valid:
                for i in range(len(long_cyc)):
                    a = long_cyc[i]
                    b = long_cyc[(i + 1) % len(long_cyc)]
                    if (min(a, b), max(a, b)) not in edge_set:
                        valid = False
                        break
            if valid:
                L = len(long_cyc)
                if L not in length_to_cycle and 3 <= L <= N:
                    length_to_cycle[L] = long_cyc
        
        # Check if all lengths are covered
        for L in range(3, N + 1):
            if L not in length_to_cycle:
                return None
        
        # Generate cycles in order 3..N
        cycles = []
        for L in range(3, N + 1):
            cycles.append(length_to_cycle[L])
        
        # Final verification
        for cyc in cycles:
            if len(cyc) != len(set(cyc)):
                return None
            for i in range(len(cyc)):
                a = cyc[i]
                b = cyc[(i + 1) % len(cyc)]
                if (min(a, b), max(a, b)) not in edge_set:
                    return None
        
        edges = rim_edges + [(u, v) for (u, v) in chords]
        return {'edges': edges, 'cycles': cycles}

    # Greedy set cover
    def greedy_cover():
        uncovered = set(target_lengths)
        chosen = []
        remaining = list(all_chords)
        while uncovered and remaining:
            best_chord = None
            best_count = 0
            for chord in remaining:
                cnt = len(chord_covers[chord] & uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best_chord = chord
            if best_chord is None:
                break
            chosen.append(best_chord)
            uncovered -= chord_covers[best_chord]
            remaining.remove(best_chord)
        return chosen, uncovered

    # Local search: try to remove chords or swap for fewer
    def local_search(chords, max_iters=500):
        chords = list(set(chords))
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try removing a chord
            improved = False
            for i in range(len(chords)):
                candidate = chords[:i] + chords[i+1:]
                # Check if still covers all target lengths
                covered = set()
                for c in candidate:
                    covered |= chord_covers[c]
                if covered >= target_lengths:
                    chords = candidate
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swapping one chord for another that covers more
            best_swap = None
            best_gain = 0
            for i in range(len(chords)):
                for new_chord in all_chords:
                    if new_chord in chords:
                        continue
                    candidate = chords[:i] + [new_chord] + chords[i+1:]
                    covered = set()
                    for c in candidate:
                        covered |= chord_covers[c]
                    if covered >= target_lengths:
                        # Check if this allows removing another chord
                        for j in range(len(candidate)):
                            if j == i:
                                continue
                            candidate2 = candidate[:j] + candidate[j+1:]
                            covered2 = set()
                            for c in candidate2:
                                covered2 |= chord_covers[c]
                            if covered2 >= target_lengths:
                                best_swap = candidate2
                                best_gain = 1
                                break
                        if best_swap:
                            break
                if best_swap:
                    break
            
            if best_swap:
                chords = best_swap
            else:
                break
        
        return chords

    # Random restart with greedy
    def random_greedy():
        random.shuffle(all_chords)
        uncovered = set(target_lengths)
        chosen = []
        remaining = list(all_chords)
        while uncovered and remaining:
            best_chord = None
            best_count = 0
            for chord in remaining:
                cnt = len(chord_covers[chord] & uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best_chord = chord
            if best_chord is None:
                break
            chosen.append(best_chord)
            uncovered -= chord_covers[best_chord]
            remaining.remove(best_chord)
        return chosen, uncovered

    # Try greedy first
    chords, uncovered = greedy_cover()
    if not uncovered:
        chords = local_search(chords)
        result = build_cycles(chords)
        if result:
            return result

    # Try random restarts
    best_chords = None
    best_len = float('inf')
    
    for _ in range(100):
        if time.time() > deadline:
            break
        chords_r, uncovered_r = random_greedy()
        if not uncovered_r:
            chords_r = local_search(chords_r)
            if len(chords_r) < best_len:
                best_len = len(chords_r)
                best_chords = chords_r
    
    if best_chords is not None:
        result = build_cycles(best_chords)
        if result:
            return result

    # Fallback: use the greedy result even if not all covered (shouldn't happen)
    if not uncovered:
        result = build_cycles(chords)
        if result:
            return result

    # Final fallback: fan graph (always valid but many edges)
    edges = [(0, i) for i in range(1, N)] + [(i, i + 1) for i in range(1, N - 1)]
    cycles = []
    for L in range(3, N + 1):
        cycles.append(list(range(L)))
    return {'edges': edges, 'cycles': cycles}
