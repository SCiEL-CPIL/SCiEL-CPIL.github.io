import time

N = 40

def construct():
    start_time = time.time()
    
    # Structural change: Use a sparse chord set based on circular difference coverage.
    # Each chord (0, d) with distance d covers cycle lengths d+1 and 41-d.
    # We need to cover lengths 3-39 (length 40 is the Hamiltonian cycle).
    # 
    # Key insight: Instead of using all chords from vertex 0, use chords from
    # multiple vertices to potentially cover more lengths per edge through
    # composite cycle constructions.
    #
    # New approach: Use a "star" of chords but verify if we can reduce count.
    # The minimum number of chords needed: each chord covers at most 2 lengths.
    # We need 37 lengths (3-39), so at least 19 chords. Current uses 19.
    #
    # Alternative: Can we use a different base graph? What if we use a complete
    # bipartite K_{20,20} structure? That has 400 edges - too many.
    #
    # Let me try: Use a 2-regular base (two disjoint cycles) plus chords.
    # Two disjoint 20-cycles: 40 edges. Then add chords to make it pancyclic.
    
    # Actually, let's try a completely different approach:
    # Use a "ladder" graph (2 x 20 grid) which has 2*20 + 20 = 60 edges.
    # Then remove some edges and add others to reduce count.
    
    # Ladder: vertices (i,0) and (i,1) for i=0..19
    # Map: vertex i -> (i//2, i%2) where i=0..39
    # Edges: horizontal (i,0)-(i+1,0), (i,1)-(i+1,1) for i=0..18
    #         vertical (i,0)-(i,1) for i=0..19
    
    # Total ladder edges: 2*19 + 20 = 58 edges. Better than 59!
    
    # Now I need to verify this is pancyclic and find cycles of all lengths 3-40.
    
    edges = []
    
    # Horizontal edges on top row: (0,2,4,...,38) -> vertices 0,2,4,...,38
    for i in range(19):
        edges.append((2*i, 2*i + 2))
    
    # Horizontal edges on bottom row: (1,3,5,...,39) -> vertices 1,3,5,...,39
    for i in range(19):
        edges.append((2*i + 1, 2*i + 3))
    
    # Vertical rungs
    for i in range(20):
        edges.append((2*i, 2*i + 1))
    
    # That's 58 edges. Now find cycles of all lengths 3-40.
    
    # Build adjacency for cycle finding
    adj = {i: set() for i in range(N)}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    
    def find_cycle(length):
        """Find a cycle of given length in the ladder graph."""
        if length == 40:
            # Hamiltonian cycle in ladder: snake through
            # 0,1,3,5,...,39,38,36,...,2,0? No, need to verify.
            # Ladder Hamiltonian: 0,2,4,...,38,39,37,35,...,1,0
            return list(range(0, 40, 2)) + list(range(39, -1, -2))
        
        if length <= 4:
            # Small cycles: use 2x2 squares or triangles
            # In ladder, smallest cycle is 4 (a square)
            if length == 4:
                return [0, 1, 3, 2]  # 0-1-3-2-0
            elif length == 3:
                # No triangles in bipartite ladder! Problem.
                # Need to add an edge to create a triangle.
                return None
        
        # For larger cycles, use the ladder structure
        # A cycle of length L in ladder: go along top for k steps, cross down,
        # go back along bottom for k steps, cross up. Length = 2k + 2.
        # Or go along top for k, cross, bottom for k-1, cross, top for 1. Etc.
        
        # General: use a "rectangle" of width w and height 2
        # Cycle: (0,0)-(1,0)-...-(w,0)-(w,1)-(w-1,1)-...-(0,1)-(0,0)
        # Length = 2w + 2
        # For odd lengths, need different construction.
        
        # For even length L = 2w + 2: w = (L-2)/2
        # For odd length L: need to include a "zigzag"
        
        if length % 2 == 0:
            w = (length - 2) // 2
            if w <= 19:
                cycle = []
                # Top row: 0, 2, 4, ..., 2w
                for i in range(w + 1):
                    cycle.append(2 * i)
                # Right side down
                cycle.append(2 * w + 1)
                # Bottom row back: 2w+1, 2w-1, ..., 1
                for i in range(w, -1, -1):
                    cycle.append(2 * i + 1)
                return cycle
        
        # Odd length: use a different pattern
        # For length 5: 0,2,4,5,3,1,0? Check edges: 0-2,2-4,4-5,5-3,3-1,1-0
        # 4-5 is vertical (ok), 5-3 is horizontal on bottom (ok), 3-1 is horizontal (ok), 1-0 is vertical (ok)
        # That works! Length 6.
        
        # For odd L = 2w + 1: 
        # Pattern: 0,2,...,2w, 2w+1, 2w-1, ..., 1, 0
        # That's w+1 + w + 1 = 2w+2 vertices. Even.
        
        # Try: 0,2,...,2w, 2w+1, 2w-1, ..., 3, 1, 0
        # Count: (w+1) + (w) + 1 = 2w+2. Still even.
        
        # For odd length, I need a different approach. Maybe add a chord.
        
        return None
    
    # The ladder is bipartite, so no odd cycles. I need to add at least one
    # non-bipartite edge. Let me add chord (0,2) to create a triangle 0-1-2-0.
    # Wait, 0-2 is already an edge in the ladder (horizontal top). 
    # 0-1-2-0: edges 0-1 (vertical), 1-2 (not an edge!), 2-0 (horizontal).
    # 1 and 2 are not adjacent. So no triangle yet.
    
    # Add edge (1,2) to create triangle 0-1-2-0? 0-1,1-2,2-0. Yes!
    edges.append((1, 2))
    
    # Now rebuild adjacency
    adj = {i: set() for i in range(N)}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    
    def find_cycle_bfs(length):
        """Find a simple cycle of given length using backtracking."""
        if length > N:
            return None
        if length < 3:
            return None
        
        # Try to find cycle starting from vertex 0
        # Use DFS with pruning
        visited = [False] * N
        path = [0]
        visited[0] = True
        
        def dfs(current, target_len):
            if len(path) == target_len:
                if current in adj[0]:
                    return list(path)
                return None
            
            if time.time() - start_time > 55:
                return None
            
            for neighbor in sorted(adj[current]):
                if neighbor == 0 and len(path) < target_len:
                    continue
                if not visited[neighbor]:
                    visited[neighbor] = True
                    path.append(neighbor)
                    result = dfs(neighbor, target_len)
                    if result is not None:
                        return result
                    path.pop()
                    visited[neighbor] = False
            return None
        
        result = dfs(0, length)
        return result
    
    # Precompute cycles for all lengths
    cycles = []
    for L in range(3, N + 1):
        if time.time() - start_time > 50:
            # Fallback: use Hamiltonian cycle for remaining
            cycles.append(list(range(N)))
            continue
        
        if L == 40:
            cycles.append(list(range(N)))
        else:
            cyc = find_cycle_bfs(L)
            if cyc is None:
                # Fallback
                cycles.append(list(range(N)))
            else:
                cycles.append(cyc)
    
    # Validate
    edge_set = set()
    for e in edges:
        a, b = min(e), max(e)
        edge_set.add((a, b))
    
    valid = True
    for i, cyc in enumerate(cycles):
        L = i + 3
        if len(cyc) != L:
            valid = False
            break
        if len(set(cyc)) != L:
            valid = False
            break
        for j in range(L):
            u = cyc[j]
            v = cyc[(j + 1) % L]
            a, b = min(u, v), max(u, v)
            if (a, b) not in edge_set:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to original solution
        ham_edges = [(i, (i + 1) % N) for i in range(N)]
        chord_edges = [(0, d) for d in range(2, 21)]
        edges = ham_edges + chord_edges
        cycles = []
        for L in range(3, N + 1):
            if L == 40:
                cycles.append(list(range(N)))
            else:
                if L <= 21:
                    cycle_verts = list(range(L))
                else:
                    end_vertex = 41 - L
                    cycle_verts = [0] + list(range(N - 1, end_vertex - 1, -1))
                cycles.append(cycle_verts)
    
    return {'edges': edges, 'cycles': cycles}
