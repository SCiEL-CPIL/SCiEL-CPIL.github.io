import time
import random
import sys
sys.setrecursionlimit(100000)

N = 40
TARGETS = list(range(3, N))  # 3..39, length 40 is the rim
UNIVERSE = len(TARGETS)  # 37

def construct():
    start = time.time()
    deadline = start + 85.0
    
    # Rim edges
    rim = set()
    for i in range(N):
        a, b = i, (i+1) % N
        if a > b: a, b = b, a
        rim.add((a,b))
    
    # All possible chords (non-rim, non-self)
    chords = []
    chord_set = set()
    for u in range(N):
        for v in range(u+1, N):
            if v == (u+1) % N:
                continue
            if u == 0 and v == N-1:
                continue
            chords.append((u,v))
            chord_set.add((u,v))
    
    # For each chord, the two rim-cycle lengths it directly creates
    # chord (u,v): d = (v-u) % N, but since u<v, d = v-u if v>u else v-u+N
    # Actually for u<v: d = v-u. Cycle1: u->u+1->...->v->u, length d+1
    # Cycle2: v->v+1->...->u (wrap) ->v, length N-d+1
    chord_lengths = {}
    for (u,v) in chords:
        d = v - u  # since u<v, d in 2..N-2 (excluding rim)
        l1 = d + 1
        l2 = N - d + 1
        lengths = set()
        if l1 in TARGETS:
            lengths.add(l1)
        if l2 in TARGETS:
            lengths.add(l2)
        chord_lengths[(u,v)] = frozenset(lengths)
    
    # Map length to list of chords that cover it
    length_to_chords = {L: [] for L in TARGETS}
    for c, ls in chord_lengths.items():
        for L in ls:
            length_to_chords[L].append(c)
    
    # Precompute for speed: for each chord, bitmask of covered lengths
    # Map length to bit index
    len_to_bit = {L: i for i, L in enumerate(TARGETS)}
    chord_mask = {}
    for c, ls in chord_lengths.items():
        m = 0
        for L in ls:
            m |= (1 << len_to_bit[L])
        chord_mask[c] = m
    
    full_mask = (1 << UNIVERSE) - 1
    
    # Strategy 1: Iterative deepening to find minimum chord count
    # Lower bound: each chord covers at most 2 lengths, need ceil(37/2) = 19
    # But some chords cover only 1 useful length if l1 or l2 is 2 or 40 (excluded)
    # Actually l1 = d+1, d>=2 so l1>=3; l2 = N-d+1, d<=N-2 so l2>=3.
    # For d=2: l1=3, l2=39. Both in TARGETS.
    # For d=18: l1=19, l2=23. Both in.
    # For d=19: l1=20, l2=21. Both in.
    # For d=20: l1=21, l2=20. Same.
    # So all chords cover exactly 2 lengths (since d in 2..18 gives distinct pairs, d=19 is self-pair)
    # Actually d and N-d give same pair. d in 2..18: 17 values, each covers 2 distinct lengths.
    # d=19: covers 20,21 (same pair).
    # So 18 distinct pairs cover 36 lengths, and we need 37. So need at least 19 chords.
    
    # Try to find solution with k chords for k = 19, 20, 21, ...
    # Use DFS with pruning
    
    def dfs_deepen(k, chord_list, covered_mask, depth, selected, best_found):
        """Find k chords covering all lengths. Returns list of chords or None."""
        if time.time() > deadline:
            return None
        if depth == k:
            if covered_mask == full_mask:
                return selected[:]
            return None
        if k - depth == 0:
            return None if covered_mask != full_mask else selected[:]
        
        # Pruning: can remaining chords cover the missing lengths?
        missing = full_mask & ~covered_mask
        if missing == 0:
            return selected[:] if depth <= k else None
        
        # Lower bound: need at least ceil(popcount(missing)/2) more chords
        remaining_needed = (missing.bit_count() + 1) // 2
        if k - depth < remaining_needed:
            return None
        
        # Choose the length with fewest covering chords among missing
        # Find the missing length with minimum options
        best_len = -1
        best_count = 999
        for i in range(UNIVERSE):
            if (missing >> i) & 1:
                L = TARGETS[i]
                cnt = len(length_to_chords[L])
                # Count how many of those are still available (not in selected)
                avail = 0
                for c in length_to_chords[L]:
                    if c not in selected_set:
                        avail += 1
                if avail < best_count:
                    best_count = avail
                    best_len = L
                    if best_count == 0:
                        return None
        
        if best_len == -1:
            return selected[:] if covered_mask == full_mask else None
        
        # Try chords that cover best_len
        for c in length_to_chords[best_len]:
            if c in selected_set:
                continue
            new_mask = covered_mask | chord_mask[c]
            selected.append(c)
            selected_set.add(c)
            result = dfs_deepen(k, chord_list, new_mask, depth+1, selected, best_found)
            if result is not None:
                return result
            selected.pop()
            selected_set.remove(c)
        
        return None
    
    selected_set = set()
    best_solution = None
    best_k = 999
    
    # Try k from 19 upward
    for k in range(19, 30):
        if time.time() > deadline:
            break
        selected = []
        selected_set.clear()
        result = dfs_deepen(k, chords, 0, 0, selected, None)
        if result is not None:
            best_solution = result
            best_k = k
            break
    
    # If DFS didn't find, use greedy as fallback
    if best_solution is None:
        # Greedy set cover
        remaining = set(TARGETS)
        selected = []
        while remaining:
            best_c = None
            best_cnt = 0
            for c in chords:
                if c in selected_set:
                    continue
                cnt = len(chord_lengths[c] & remaining)
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_c = c
            if best_c is None or best_cnt == 0:
                break
            selected.append(best_c)
            selected_set.add(best_c)
            remaining -= chord_lengths[best_c]
        if remaining:
            # Add more chords to cover remaining
            for c in chords:
                if c in selected_set:
                    continue
                if chord_lengths[c] & remaining:
                    selected.append(c)
                    selected_set.add(c)
                    remaining -= chord_lengths[c]
        best_solution = selected
        best_k = len(best_solution)
    
    # Local search: try to remove chords and re-verify
    # For now, keep the solution as-is
    
    # Build edge list
    edges = list(rim)
    for (u,v) in best_solution:
        if (u,v) not in edges:
            edges.append((u,v))
    
    # Deduplicate
    edge_set = set()
    for e in edges:
        a, b = e
        if a > b: a, b = b, a
        edge_set.add((a,b))
    edges = list(edge_set)
    
    # Generate cycles for each length 3..40
    # Length 40: rim
    cycles = [None] * (N - 2)  # index L-3 for length L
    
    # Build adjacency for cycle verification
    adj = {i: set() for i in range(N)}
    for (a,b) in edge_set:
        adj[a].add(b)
        adj[b].add(a)
    
    # For each chord, create the two rim-based cycles
    for (u,v) in best_solution:
        d = v - u
        l1 = d + 1
        l2 = N - d + 1
        
        # Cycle 1: u, u+1, ..., v
        if 3 <= l1 <= N:
            if v >= u:
                cyc = list(range(u, v+1))
            else:
                cyc = list(range(u, N)) + list(range(0, v+1))
            if len(cyc) == l1:
                cycles[l1-3] = cyc
        
        # Cycle 2: v, v+1, ..., u (wrap)
        if 3 <= l2 <= N:
            if u >= v:
                cyc = list(range(v, u+1))
            else:
                cyc = list(range(v, N)) + list(range(0, u+1))
            if len(cyc) == l2:
                if cycles[l2-3] is None:
                    cycles[l2-3] = cyc
    
    # Length 40
    cycles[37] = list(range(N))
    
    # Verify and fill any missing cycles by search
    # For each missing length, try to find a cycle in the graph
    for L in range(3, N+1):
        if cycles[L-3] is None:
            # Find a cycle of length L in the graph
            # Use DFS
            def find_cycle(length):
                if length == N:
                    return list(range(N))
                # Try all starting vertices
                for start in range(N):
                    # DFS
                    stack = [(start, [start], {start})]
                    while stack:
                        node, path, visited = stack.pop()
                        if len(path) == length:
                            if start in adj[node]:
                                return path
                        else:
                            for nb in adj[node]:
                                if nb not in visited:
                                    if nb == start and len(path) < length:
                                        continue
                                    stack.append((nb, path + [nb], visited | {nb}))
                    return None
                return None
            
            cyc = find_cycle(L)
            if cyc is not None:
                cycles[L-3] = cyc
            else:
                # Fallback: use rim-based cycle if possible
                # This should not happen
                cycles[L-3] = list(range(L))  # Invalid, but placeholder
    
    # Final verification
    for L in range(3, N+1):
        cyc = cycles[L-3]
        if cyc is None or len(cyc) != L:
            raise ValueError(f"Missing cycle for length {L}")
        if len(set(cyc)) != L:
            raise ValueError(f"Cycle for length {L} has duplicate vertices")
        for i in range(L):
            u = cyc[i]
            v = cyc[(i+1) % L]
            if (u,v) not in edge_set and (v,u) not in edge_set:
                raise ValueError(f"Cycle for length {L} has missing edge ({u},{v})")
    
    return {'edges': edges, 'cycles': cycles}

if __name__ == '__main__':
    res = construct()
    print(f"Edges: {len(res['edges'])}, Score: {len(res['edges']) - 40}")
    # Verify
    edge_set = set()
    for e in res['edges']:
        a, b = e
        if a > b: a, b = b, a
        edge_set.add((a,b))
    
    ok = True
    for L in range(3, N+1):
        c = res['cycles'][L-3]
        if len(c) != L:
            print(f"Length {L}: wrong cycle length {len(c)}")
            ok = False
            break
        if len(set(c)) != L:
            print(f"Length {L}: duplicate vertices")
            ok = False
            break
        for i in range(L):
            u = c[i]
            v = c[(i+1)%L]
            if (u,v) not in edge_set and (v,u) not in edge_set:
                print(f"Length {L}: missing edge ({u},{v})")
                ok = False
                break
        if not ok:
            break
    if ok:
        print("All cycles valid")
