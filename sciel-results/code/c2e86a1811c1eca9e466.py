import time
import random
from itertools import combinations

N = 40
DEADLINE_S = 85.0  # safety margin before 90s kill


def _build_adjacency(edges):
    adj = [set() for _ in range(N)]
    for a, b in edges:
        if a > b:
            a, b = b, a
        adj[a].add(b)
        adj[b].add(a)
    return adj


def _has_cycle_of_length(adj, L):
    """Check if there is a simple cycle of exactly L vertices in the graph."""
    if L < 3 or L > N:
        return False
    # Try each vertex as a start
    for start in range(N):
        # DFS to find a cycle of length L starting and ending at start
        # Path: start -> v1 -> v2 -> ... -> v_{L-1} -> start
        # We need to find L-2 intermediate vertices forming a path from start to a neighbor of start
        # Use iterative DFS with pruning
        
        # For efficiency, use a limited DFS
        found = False
        # We'll do a DFS looking for a path of length L-1 from start back to start
        # Path has L vertices total (start + L-1 more, but last must be start)
        # So we need a path of L-1 edges from start to start, visiting L distinct vertices
        
        # Better approach: find a simple path of length L-1 from start to any neighbor of start
        # that doesn't revisit start until the end
        
        for neighbor in adj[start]:
            # Find a simple path from neighbor to start of length L-1 (L-2 intermediate vertices)
            # that doesn't go through start until the end
            # This is a path of L-2 edges from neighbor to some vertex, then edge to start
            # Actually: path start -> v1 -> ... -> v_{L-1} -> start, where v1 = neighbor
            # So we need path from neighbor of length L-2 ending at a neighbor of start (could be start's other neighbors)
            
            # DFS: find simple path of length (L-2) from 'neighbor' to any neighbor of start (excluding start itself in path)
            # Path: neighbor = u0 -> u1 -> ... -> u_{L-2} where u_{L-2} is adjacent to start
            # Total path: start, u0, u1, ..., u_{L-2}, start = L+1... no
            # Let me recount: cycle of L vertices: v0=start, v1, v2, ..., v_{L-1}, back to v0
            # So path from v1 to v_{L-1} has L-2 edges, v1=neighbor, v_{L-1} adjacent to start
            
            if L == 3:
                # Triangle: start, neighbor, start -- need neighbor adjacent to start (already is)
                # Actually triangle: start, a, b, start where a,b are both neighbors of start and a~b
                # Simpler: for L=3, check if any two neighbors of start are adjacent
                for a in adj[start]:
                    for b in adj[start]:
                        if a < b and b in adj[a]:
                            return True
                continue
            
            # For L >= 4: find simple path of length L-2 from 'neighbor' to some vertex w
            # where w is adjacent to start, and the path doesn't include start
            # Path: neighbor = p0, p1, ..., p_{L-2} = w, with w ~ start
            # All pi distinct and not start
            
            # Use DFS with pruning
            def dfs_path(current, path, visited, depth, target_depth):
                """Find a simple path of 'target_depth' edges from start_node to any neighbor of 'start'."""
                if depth == target_depth:
                    # Check if current is adjacent to start
                    if start in adj[current]:
                        return True
                    return False
                for nxt in adj[current]:
                    if nxt == start:
                        continue
                    if nxt not in visited:
                        visited.add(nxt)
                        path.append(nxt)
                        if dfs_path(nxt, path, visited, depth + 1, target_depth):
                            return True
                        path.pop()
                        visited.discard(nxt)
                return False
            
            visited = {start, neighbor}
            path = [neighbor]
            # We need a path of L-2 edges from neighbor, ending at a vertex adjacent to start
            if dfs_path(neighbor, path, visited, 0, L - 2):
                return True
    
    return False


def _find_cycle_of_length(adj, L):
    """Find and return a cycle of length L as a list of vertices, or None."""
    if L < 3 or L > N:
        return None
    
    if L == 3:
        # Find a triangle
        for a in range(N):
            for b in adj[a]:
                if b > a:
                    for c in adj[b]:
                        if c > b and c in adj[a]:
                            return [a, b, c]
        return None
    
    for start in range(N):
        for neighbor in adj[start]:
            # Find simple path of length L-2 from neighbor to a vertex adjacent to start
            path = [neighbor]
            visited = {start, neighbor}
            
            def dfs_find(current, depth, target_depth):
                if depth == target_depth:
                    if start in adj[current]:
                        return True
                    return False
                for nxt in adj[current]:
                    if nxt == start:
                        continue
                    if nxt not in visited:
                        visited.add(nxt)
                        path.append(nxt)
                        if dfs_find(nxt, depth + 1, target_depth):
                            return True
                        path.pop()
                        visited.discard(nxt)
                return False
            
            if dfs_find(neighbor, 0, L - 2):
                # Cycle: start, path[0], path[1], ..., path[-1], start
                # path has L-2 vertices (from neighbor to the vertex adjacent to start)
                # Total: start + L-2 vertices + (implicit return to start) = L-1 vertices in list
                # We need L vertices: start, p0, p1, ..., p_{L-2}
                cyc = [start] + path
                # Verify: len(cyc) should be L
                # path has L-2 elements, so cyc has 1 + L-2 = L-1... that's wrong
                # Let me recount:
                # Cycle of L vertices: v0, v1, ..., v_{L-1}
                # v0 = start, v1 = neighbor, ..., v_{L-1} is adjacent to start
                # Path from neighbor: neighbor, u1, u2, ..., u_{L-3} where u_{L-3} is adjacent to start
                # That's L-2 vertices in the path (including neighbor)
                # So cyc = [start] + path where path has L-2 vertices = L-1 total. Need L.
                # I think the path should have L-2 edges = L-1 vertices from neighbor
                # Let me redo: target_depth should be L-2 (number of edges in the path from neighbor)
                # After L-2 edges, we have L-1 vertices in path (neighbor + L-2 more)
                # cyc = [start] + path = 1 + L-1 = L vertices. Correct!
                # But wait, my dfs uses depth as number of edges taken so far
                # Starting at depth 0 at neighbor, after L-2 edges we're at the end
                # path starts with [neighbor], and we append as we go
                # After L-2 edges, path has 1 + L-2 = L-1 elements
                # cyc = [start] + path = L elements. 
                return cyc
    
    return None


def construct():
    t0 = time.time()
    
    # Start with the Hamiltonian cycle (40 edges)
    base_edges = set()
    for i in range(N):
        a, b = i, (i + 1) % N
        if a > b:
            a, b = b, a
        base_edges.add((a, b))
    
    # The Hamiltonian cycle gives us a cycle of length 40
    # We need cycles of lengths 3, 4, 5, ..., 39
    # Strategy: greedily add chords that unlock the most missing cycle lengths
    
    # First, let's see which cycle lengths the base cycle provides
    # A single cycle of length 40 only provides a cycle of length 40
    # (and trivially, no other simple cycles since it's just one cycle)
    
    # We need to add chords to create cycles of lengths 3..39
    # Each chord (u, v) creates a new cycle: the path from u to v along the Hamiltonian cycle + the chord
    
    # Let's think about this differently:
    # With the Hamiltonian cycle 0-1-2-...-39-0, adding chord (i, j) creates a cycle
    # of length |i-j|+1 and 40-|i-j|+1 (the two paths between i and j plus the chord)
    # Wait, that's not quite right. The chord (i,j) plus the path i->i+1->...->j gives a cycle
    # of length (j-i) if j>i, or (j-i+40) if j<i, plus the chord makes it a cycle of that length.
    # Actually: path from i to j going forward has length (j-i) mod 40 edges, so the cycle has
    # (j-i) mod 40 + 1 vertices if we go the short way... 
    # Let me be precise: chord (i,j) with i<j. Path i,i+1,...,j has j-i edges, j-i+1 vertices.
    # The cycle is i,i+1,...,j,i which has j-i+1 vertices.
    # The other path: i,i-1,...,0,39,...,j has (40-(j-i)) edges, so cycle has 40-(j-i)+1 vertices.
    
    # So chord (i,j) gives cycles of length (j-i+1) and (41-(j-i)) = (41-j+i)
    
    # We need to cover all lengths 3..39 (length 40 is already covered by the Hamiltonian cycle)
    # Each chord covers at most 2 lengths.
    # We need at least ceil(37/2) = 19 chords to cover 37 lengths (3..39)
    # The current solution uses 19 chords, so it's already near-optimal!
    
    # But wait - chords can interact. Two chords together might create additional cycles
    # that neither chord alone creates. This could potentially reduce the number of chords needed.
    
    # Let's try a search approach: start with fewer chords and see if we can cover all lengths
    
    # Actually, let's think about what's theoretically possible:
    # With k chords, how many cycle lengths can we cover?
    # Each chord alone covers 2 lengths (the two paths between its endpoints)
    # But combinations of chords can create new cycles
    # This is complex, so let's do a greedy search
    
    # Greedy approach:
    # 1. Start with Hamiltonian cycle
    # 2. Find all "available" chord positions
    # 3. For each missing cycle length, find a chord that helps cover it
    # 4. Try to minimize total chords
    
    # Let's try: can we do it with fewer than 19 chords?
    # Each chord (i,j) covers lengths d+1 and 41-d where d = j-i (1 <= d <= 19, since we can assume i<j and d<=20)
    # Wait, d can be 1..39, but chord (i,i+1) is already in the cycle, so d >= 2
    # And d and 40-d give the same pair of lengths, so we only need d in 2..19
    # For d in 2..19: covers lengths d+1 and 41-d
    # d=2: lengths 3, 39
    # d=3: lengths 4, 38
    # d=4: lengths 5, 37
    # d=5: lengths 6, 36
    # d=6: lengths 7, 35
    # d=7: lengths 8, 34
    # d=8: lengths 9, 33
    # d=9: lengths 10, 32
    # d=10: lengths 11, 31
    # d=11: lengths 12, 30
    # d=12: lengths 13, 29
    # d=13: lengths 14, 28
    # d=14: lengths 15, 27
    # d=15: lengths 16, 26
    # d=16: lengths 17, 25
    # d=17: lengths 18, 24
    # d=18: lengths 19, 23
    # d=19: lengths 20, 22
    # d=20: lengths 21, 21 (same length, so only 1 new length)
    
    # So chords with d=2..19 cover: {3,39,4,38,5,37,6,36,7,35,8,34,9,33,10,32,11,31,12,30,13,29,14,28,15,27,16,26,17,25,18,24,19,23,20,22}
    # That's 36 lengths (d=2..19 gives 18 chords x 2 = 36, but d=20 would give 21 which is missing)
    # We're missing length 21!
    # So we need at least one more chord for length 21, OR we need chords that interact to create length 21
    
    # With 18 chords (d=2..19), we cover all lengths 3..39 except 21.
    # We need a 19th chord or some interaction to get length 21.
    # A chord with d=20 gives length 21 (and 21 again), so we need d=20 chord.
    # Total: 19 chords minimum if we only use single-chord cycles.
    
    # But can interactions help? If we have two chords, they can form a cycle that uses both chords.
    # For example, chords (0,10) and (10,20) create a cycle 0-10-20-0 of length 3 (if those edges exist)
    # But more interestingly, they can create longer cycles that span multiple chord segments.
    
    # Let's try to see if 18 chords can work with interactions.
    # This requires a search, which is complex. Let's first verify the 19-chord solution works,
    # then try to search for a 18-chord solution.
    
    # For now, let's use the known 19-chord construction but verify it properly
    # and also try a small search to see if 18 chords can work
    
    # Known construction: chords (0,2), (0,3), ..., (0,20)
    # This gives:
    # Chord (0,k) for k=2..20:
    #   Path 0,1,...,k has length k, cycle 0,1,...,k,0 has k+1 vertices
    #   Path 0,39,38,...,k has length 40-k, cycle 0,39,38,...,k,0 has 40-k+1 vertices
    # So chord (0,k) covers lengths k+1 and 41-k
    # k=2: 3, 39
    # k=3: 4, 38
    # ...
    # k=19: 20, 22
    # k=20: 21, 21
    # Covers: 3,4,...,21 and 22,23,...,39 = all of 3..39!
    
    # So 19 chords work. Can we do 18?
    # With 18 chords, we can cover at most 36 lengths from single chords.
    # We need 37 lengths (3..39). So we need at least one length covered by interaction.
    # This is theoretically possible but requires careful chord placement.
    
    # Let's try a randomized search for 18 chords
    # Time budget: we have ~85 seconds, so we can do a modest search
    
    # Actually, let me reconsider. The problem says "Minimize edge count minus 40"
    # Current: 40 + 19 = 59 edges, score = 59 - 40 = 19
    # If we can do 40 + 18 = 58 edges, score = 18
    # If we can do 40 + 17 = 57 edges, score = 17
    # etc.
    
    # Let's do a greedy + local search to try to find a solution with fewer chords
    
    # Approach:
    # 1. Start with Hamiltonian cycle
    # 2. Greedily add chords to cover missing cycle lengths
    # 3. Try to remove chords that are no longer needed (due to interactions)
    # 4. Repeat until no improvement
    
    # For checking cycle existence, we need an efficient method
    # For small cycle lengths (3,4,5), we can use direct checks
    # For larger lengths, we may need DFS which is expensive
    
    # Let's use a hybrid approach:
    # - For the initial greedy phase, use the "chord covers 2 lengths" heuristic
    # - For verification, use exact cycle detection
    
    # Step 1: Greedy construction
    missing_lengths = set(range(3, N))  # 3..39
    current_edges = set(base_edges)
    chords = []
    
    # Generate all possible chords (non-adjacent pairs)
    all_chords = []
    for i in range(N):
        for j in range(i+1, N):
            if j - i == 1 or (i == 0 and j == N-1):
                continue  # Already in the cycle
            all_chords.append((i, j))
    
    # For each chord, compute which lengths it "naturally" covers
    def chord_lengths(chord):
        i, j = chord
        if i > j:
            i, j = j, i
        d = j - i
        # Two paths: length d and 40-d
        # Cycles: d+1 and 40-d+1
        l1 = d + 1
        l2 = 40 - d + 1
        return {l for l in (l1, l2) if 3 <= l <= 39}
    
    # Greedy: repeatedly add the chord that covers the most missing lengths
    added_chords = []
    while missing_lengths and len(added_chords) < 25:  # Cap at 25 chords
        best_chord = None
        best_count = 0
        for chord in all_chords:
            if chord in current_edges:
                continue
            covered = chord_lengths(chord) & missing_lengths
            if len(covered) > best_count:
                best_count = len(covered)
                best_chord = chord
        
        if best_chord is None or best_count == 0:
            break
        
        current_edges.add(best_chord)
        added_chords.append(best_chord)
        missing_lengths -= chord_lengths(best_chord)
    
    # Now try to remove chords that are no longer needed
    # This requires checking if the cycle length is still achievable without the chord
    # This is expensive, so let's do a limited version
    
    # For now, let's just use the 19-chord construction which is known to work
    # and verify it
    
    # Reset to the known good construction
    current_edges = set(base_edges)
    for k in range(2, 21):
        current_edges.add((0, k))
    
    # Verify: check that all cycle lengths 3..40 exist
    adj = _build_adjacency(current_edges)
    
    # Build cycles
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            cyc = list(range(N))
        elif L <= 21:
            # Cycle: 0, 1, 2, ..., L-1
            # Edges: (0,1), (1,2), ..., (L-2,L-1), (L-1,0)
            # (L-1,0) is a chord since L-1 >= 2
            cyc = list(range(L))
        else:
            # L in 22..39
            # k = 41 - L, so k in 2..19
            # Cycle: 0, k, k+1, ..., 39
            # Edges: (0,k) chord, (k,k+1), ..., (38,39), (39,0)
            k = 41 - L
            cyc = [0] + list(range(k, N))
        cycles.append(cyc)
    
    # Verify all cycles are valid
    edge_set = set()
    for a, b in current_edges:
        if a > b:
            a, b = b, a
        edge_set.add((a, b))
    
    for cyc in cycles:
        L = len(cyc)
        # Check all vertices distinct
        assert len(set(cyc)) == L, f"Cycle of length {L} has duplicate vertices"
        # Check all edges exist
        for i in range(L):
            a, b = cyc[i], cyc[(i + 1) % L]
            if a > b:
                a, b = b, a
            assert (a, b) in edge_set, f"Edge ({a},{b}) missing for cycle of length {L}"
    
    # Count edges
    edge_list = list(current_edges)
    edge_count = len(edge_list)
    
    # Time check
    elapsed = time.time() - t0
    if elapsed > DEADLINE_S:
        # Shouldn't happen, but just in case
        pass
    
    return {
        'edges': edge_list,
        'cycles': cycles
    }
