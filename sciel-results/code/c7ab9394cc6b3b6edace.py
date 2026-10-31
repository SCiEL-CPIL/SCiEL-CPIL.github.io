import time
import random
import sys
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Strategy: 
    1. If k=3 and g=16, use a specialized randomized lift + local search.
    2. Otherwise, fallback to LUW construction (which is robust for other parameters).
    """
    
    # Fallback: LUW construction for general cases
    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    if k != 3 or g != 16:
        return luw_construct(k, g)

    # --- Specialized Strategy for k=3, g=16 ---
    # Target: Find a 3-regular graph with girth >= 16.
    # Method: Random lift of Petersen graph (10 vertices) to size N, then local search.
    # The Petersen graph has girth 5. Lifting it increases girth.
    # We use a "random lift" where each vertex is split into M copies, and edges are
    # permuted randomly. This preserves 3-regularity.
    # Then we perform 2-switches to break cycles of length < 16.
    
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Parameters
    # Start with a reasonable size. Moore bound for n(3,16) is ~512.
    # Known best is 936. Let's aim for ~600-1000.
    # If we start too small, we might not be able to break all cycles.
    # If we start too large, local search is slow.
    # Let's try N=1000 initially.
    
    def generate_random_lift(base_n, lift_factor):
        """
        Generate a random lift of a base graph (Petersen).
        Base graph: Petersen (10 vertices).
        Lift factor: M. Total vertices N = 10 * M.
        """
        N = base_n * lift_factor
        # Petersen edges (0-indexed)
        # Outer cycle: 0-1-2-3-4-0
        # Inner star: 5-7-9-6-8-5
        # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
        petersen_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
        ]
        
        # Map base vertex to list of lifted vertices
        # v in base -> [v*M, v*M+1, ..., v*M+M-1]
        adj = [[] for _ in range(N)]
        
        for u, v in petersen_edges:
            # For each pair of base vertices (u,v), we need to connect the lifted sets
            # in a 1-regular bipartite manner (perfect matching) to maintain degree 3.
            # Random permutation of the M vertices in set v to connect to set u.
            perm = list(range(M))
            random.shuffle(perm)
            
            for i in range(M):
                u_lifted = u * M + i
                v_lifted = v * M + perm[i]
                adj[u_lifted].append(v_lifted)
                adj[v_lifted].append(u_lifted)
                
        # Verify 3-regular
        for i in range(N):
            if len(adj[i]) != 3:
                raise ValueError(f"Vertex {i} has degree {len(adj[i])}")
        
        return N, adj

    def find_short_cycles(adj, N, girth):
        """
        Find all cycles of length < girth.
        Returns a list of cycles (lists of vertex indices).
        Uses BFS from each vertex, but limited to depth girth//2.
        To avoid duplicates and inefficiency, we only report cycles where the start vertex is the smallest.
        """
        cycles = []
        max_depth = girth // 2
        
        for start in range(N):
            # BFS
            dist = [-1] * N
            parent = [-1] * N
            dist[start] = 0
            q = deque([start])
            
            while q:
                u = q.popleft()
                if dist[u] >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        if v != parent[u] and v != start: # Avoid going back to start immediately in BFS tree logic
                            # Found a cycle
                            # Cycle length = dist[u] + dist[v] + 1
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < girth:
                                # Reconstruct cycle
                                # Path from u to start
                                path_u = []
                                cur = u
                                while cur != start:
                                    path_u.append(cur)
                                    cur = parent[cur]
                                path_u.append(start)
                                
                                # Path from v to start
                                path_v = []
                                cur = v
                                while cur != start:
                                    path_v.append(cur)
                                    cur = parent[cur]
                                path_v.append(start)
                                
                                # Cycle is path_u reversed + path_v (excluding start duplicate)
                                # Actually, cycle is u -> ... -> start -> ... -> v -> u
                                # Let's just store the set of edges or vertices for simplicity
                                # For local search, we need to identify edges to switch.
                                # We store the cycle as a tuple of sorted vertices to deduplicate
                                cycle_verts = set(path_u) | set(path_v)
                                # This is not a unique representation of the cycle structure, but good enough for "there is a cycle"
                                # Better: store the actual path
                                # Cycle: u -> parent[u] -> ... -> start -> parent[v] -> ... -> v -> u
                                # Let's reconstruct the list
                                c = []
                                cur = u
                                while cur != start:
                                    c.append(cur)
                                    cur = parent[cur]
                                c.append(start)
                                cur = v
                                while cur != start:
                                    c.append(cur)
                                    cur = parent[cur]
                                # c is now u...start...v
                                # Add u to close
                                c.append(u)
                                
                                # Deduplicate: only keep if start is min vertex in cycle
                                if start == min(c):
                                    cycles.append(c)
            # Optimization: If we found many cycles, we can stop early? No, we need to fix them all.
        
        return cycles

    def try_switch(adj, N, cycles, girth):
        """
        Attempt to break a cycle using a 2-switch.
        Returns True if a successful switch was made, False otherwise.
        """
        if not cycles:
            return False
        
        # Pick a random cycle
        # To be effective, pick a short cycle
        cycles.sort(key=lambda x: len(x))
        cycle = cycles[0]
        
        # A 2-switch involves two non-adjacent edges (u1, v1) and (u2, v2) in the graph
        # such that swapping them to (u1, v2) and (u2, v1) doesn't create multi-edges
        # and ideally breaks the cycle.
        # Simple heuristic: Pick two edges in the cycle that are far apart, and swap one endpoint
        # with a random neighbor from outside the cycle.
        
        # Let's pick a random edge in the cycle (u, v)
        # And a random edge (w, z) not in the cycle (or in the cycle but far)
        # Swap (u, v) and (w, z) -> (u, z) and (w, v)
        
        # Edges in cycle
        cycle_edges = set()
        for i in range(len(cycle) - 1):
            u, v = cycle[i], cycle[i+1]
            cycle_edges.add((min(u, v), max(u, v)))
        
        # Pick an edge in the cycle to break
        # Let's pick the first edge (u, v)
        u, v = cycle[0], cycle[1]
        
        # Find a neighbor w of u (w != v)
        # And a neighbor z of w (z != u) such that (u, z) and (v, w) are valid new edges
        # This is getting complex. Let's use a simpler "edge switch" heuristic:
        # Pick two disjoint edges (a, b) and (c, d). Replace with (a, d) and (c, b).
        # This preserves degrees.
        # We want to break the cycle. So pick (a, b) from the cycle.
        # Pick (c, d) from outside the cycle (or far in the cycle).
        
        # Let's pick (a, b) = (u, v) from cycle.
        # Find a random vertex c (not in cycle, or in cycle but far)
        # and one of its neighbors d.
        
        # To ensure we don't create multi-edges:
        # (a, d) must not exist.
        # (c, b) must not exist.
        
        # Let's try 100 random switches
        for _ in range(100):
            # Pick a random vertex c
            c = random.randint(0, N - 1)
            if c in cycle:
                continue # Avoid picking from the cycle itself for the second edge to maximize breaking chance? 
                         # Actually, picking from the cycle might just move the cycle.
                         # Picking from outside is better for breaking.
            
            # Pick a random neighbor d of c
            if not adj[c]:
                continue
            d = random.choice(adj[c])
            
            # Check if (u, d) and (c, v) are valid
            # (u, d) not in adj[u]
            if d in adj[u]:
                continue
            # (c, v) not in adj[c]
            if v in adj[c]:
                continue
            
            # Also check (u, d) != (c, v) and (u, d) != (v, c) etc. (Simple graph)
            # Since we check existence, we just need to ensure we don't create a loop (u==d, c==v)
            if u == d or c == v:
                continue
            
            # Perform switch
            # Remove (u, v) and (c, d)
            adj[u].remove(v)
            adj[v].remove(u)
            adj[c].remove(d)
            adj[d].remove(c)
            
            # Add (u, d) and (c, v)
            adj[u].append(d)
            adj[d].append(u)
            adj[c].append(v)
            adj[v].append(c)
            
            return True
        
        return False

    # Main Loop
    # Try different sizes
    # Start with a size that is likely to have girth >= 16 after lifting.
    # Petersen girth 5. Lift M. Girth roughly 5 * log(M) ? No.
    # In a random lift, the girth grows logarithmically with N?
    # Actually, for random regular graphs, girth ~ 2 log_{k-1}(N).
    # For k=3, log_2(N).
    # If N=1000, log_2(1000) ~ 10. Girth ~ 20?
    # So a random 3-regular graph of size 1000 likely has girth >= 16.
    # The "lift" of Petersen is a specific structured random graph.
    # Let's just generate random 3-regular graphs and check.
    # But generating random 3-regular graphs is harder.
    # The lift of Petersen is easy to generate.
    
    best_n = float('inf')
    best_edges = []
    
    # Try lift factors M = 50, 100, 150, 200
    # N = 10 * M
    for M in [50, 100, 150, 200, 300]:
        if time.time() > deadline:
            break
            
        N = 10 * M
        try:
            N_cur, adj = generate_random_lift(10, M)
        except Exception:
            continue
        
        # Check initial girth
        # If girth is already >= 16, we are done.
        # If not, try to fix.
        
        # Fast girth check: BFS from all vertices up to depth 7 (g/2 = 8, so depth 7 finds cycles up to 14? 
        # Cycle length L. BFS depth D. If we find a back edge at depth D, cycle len <= 2D+1?
        # To guarantee girth >= 16, we need to ensure no cycles of len 14, 12, ..., 3.
        # Max cycle len to check: 14.
        # BFS depth 7: dist[u]+dist[v]+1 <= 7+7+1 = 15. So depth 7 finds cycles up to 15.
        # So if no cycle found in depth 7 BFS, girth >= 16.
        
        max_depth = 7
        valid = True
        cycles_found = []
        
        for start in range(N_cur):
            dist = [-1] * N_cur
            parent = [-1] * N_cur
            dist[start] = 0
            q = deque([start])
            
            while q:
                u = q.popleft()
                if dist[u] >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        if v != parent[u] and v != start:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < 16:
                                # Found a short cycle
                                valid = False
                                # We don't need to store all cycles for the check, 
                                # but for local search we do.
                                # For now, just break and try to fix.
                                # To save time, break out of loops
                                q.clear()
                                break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Extract edges
            edges = []
            for u in range(N_cur):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            
            if N_cur < best_n:
                best_n = N_cur
                best_edges = edges
            
            # If we found a valid graph, we can try to find a smaller one?
            # Or just return this one.
            # Since random lifts of size 500 might work, let's try smaller first?
            # I put 50 first. If it works, we return 500.
            # If not, we try 1000.
            # We should break if we find a valid one?
            # The task is to minimize N.
            # So we should try sizes in increasing order and return the first valid one.
            # But local search might make a smaller one valid.
            # Let's implement local search for the first few sizes.
            
            # For now, if valid, save and continue to see if smaller M (already tried) or 
            # if we should break.
            # Since M increases, N increases.
            # If M=50 (N=500) is valid, we should return it.
            # If M=50 is invalid, we try M=100.
            # If M=100 is valid, we return 1000.
            # So we should break on first valid.
            break
        
        # If not valid, try local search
        if time.time() > deadline:
            continue
            
        # Local search
        # We need to find the cycles to break them.
        # Re-find cycles
        cycles_found = find_short_cycles(adj, N_cur, 16)
        
        for step in range(500): # Max 500 steps
            if time.time() > deadline:
                break
                
            if not cycles_found:
                # Check if truly valid
                # Re-verify
                valid = True
                for start in range(N_cur):
                    dist = [-1] * N_cur
                    parent = [-1] * N_cur
                    dist[start] = 0
                    q = deque([start])
                    while q:
                        u = q.popleft()
                        if dist[u] >= 7:
                            continue
                        for v in adj[u]:
                            if dist[v] == -1:
                                dist[v] = dist[u] + 1
                                parent[v] = u
                                q.append(v)
                            else:
                                if v != parent[u] and v != start:
                                    if dist[u] + dist[v] + 1 < 16:
                                        valid = False
                                        q.clear()
                                        break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    edges = []
                    for u in range(N_cur):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    if N_cur < best_n:
                        best_n = N_cur
                        best_edges = edges
                    break # Found valid, stop searching this size
                else:
                    # Still invalid, re-find cycles
                    cycles_found = find_short_cycles(adj, N_cur, 16)
                    if not cycles_found:
                        break # Should not happen
            
            # Try to switch
            if not try_switch(adj, N_cur, cycles_found, 16):
                break # No more switches possible?
            
            # Re-find cycles
            cycles_found = find_short_cycles(adj, N_cur, 16)
        
        # End local search for this M

    if best_n != float('inf'):
        return best_n, best_edges
    
    # Fallback to LUW if random lift failed
    return luw_construct(k, g)
