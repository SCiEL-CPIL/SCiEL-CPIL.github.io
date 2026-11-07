"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _count_distinct_cycle_lengths(n, edges):
    """Count distinct simple cycle lengths in the graph."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    lengths = set()
    
    def dfs(start, curr, visited, length):
        for nxt in adj[curr]:
            if nxt == start and length >= 4:
                lengths.add(length)
            elif nxt > start and not visited[nxt]:
                visited[nxt] = True
                dfs(start, nxt, visited, length + 1)
                visited[nxt] = False
    
    for start in range(n):
        visited = [False] * n
        visited[start] = True
        dfs(start, start, visited, 1)
    
    return len(lengths)

def construct():
    """Find a 3-regular bipartite graph on 12 vertices minimizing distinct cycle lengths."""
    deadline = time.time() + 85
    
    part_a = [0, 1, 2, 3, 4, 5]
    part_b = [6, 7, 8, 9, 10, 11]
    
    # Try specific constructions that might have all 4-cycles
    
    # Construction 1: 3 disjoint 4-cycles won't work (degree 2)
    # Construction 2: A 4-regular graph? No, need degree >= 3, let's try 3-regular
    
    # Try: the "cube" Q3 has 8 vertices, 3-regular, cycles 4 and 6
    # Need 12 vertices
    
    # Let me try a systematic small search over 3-regular bipartite graphs
    # This is expensive, so try specific candidates first
    
    # Candidate: two disjoint "prism" graphs? 
    # A triangular prism has 6 vertices, 3-regular, cycles: 3 (triangle!) - no good
    # Square prism (cube) has 8 vertices - too many for 2 components
    
    # Let me try: one connected 3-regular bipartite graph on 12 vertices
    # with the property that all cycles are 4-cycles
    
    # For all cycles to be 4-cycles in a bipartite graph, we need:
    # No 6-cycles, no 8-cycles, etc.
    # A 6-cycle: a1-b1-a2-b2-a3-b3-a1
    # This requires 3 A-vertices and 3 B-vertices with specific connections
    
    # Let me try the following 3-regular bipartite graph:
    # A = {0,1,2,3,4,5}, B = {6,7,8,9,10,11}
    # 0: 6,7,8
    # 1: 6,9,10
    # 2: 7,9,11
    # 3: 8,10,11
    # 4: 6,8,11  -- wait, let me make this systematic
    
    # Actually, let me try a known small graph: the Petersen graph is 3-regular on 10 vertices
    # with girth 5, so no 4-cycles. Not useful.
    
    # Let me just try a few hand-crafted candidates and check
    
    candidates = []
    
    # Candidate A: 
    # 0: 6,7,8
    # 1: 6,9,10
    # 2: 7,9,11
    # 3: 8,10,11
    # 4: 6,10,11
    # 5: 7,8,9
    edges_a = [
        [0,6],[0,7],[0,8],
        [1,6],[1,9],[1,10],
        [2,7],[2,9],[2,11],
        [3,8],[3,10],[3,11],
        [4,6],[4,10],[4,11],
        [5,7],[5,8],[5,9],
    ]
    candidates.append(edges_a)
    
    # Candidate B:
    # 0: 6,7,8
    # 1: 6,9,10
    # 2: 7,9,11
    # 3: 8,10,11
    # 4: 6,8,9
    # 5: 7,10,11
    edges_b = [
        [0,6],[0,7],[0,8],
        [1,6],[1,9],[1,10],
        [2,7],[2,9],[2,11],
        [3,8],[3,10],[3,11],
        [4,6],[4,8],[4,9],
        [5,7],[5,10],[5,11],
    ]
    candidates.append(edges_b)
    
    # Candidate C: two K_{3,3} (known to give {4,6})
    edges_c = []
    for offset in (0, 6):
        pa = [offset, offset+1, offset+2]
        pb = [offset+3, offset+4, offset+5]
        for u in pa:
            for v in pb:
                edges_c.append([u, v])
    candidates.append(edges_c)
    
    best_edges = None
    best_count = float('inf')
    
    for edges in candidates:
        if time.time() > deadline:
            break
        
        # Verify 3-regular
        deg = [0] * 12
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if any(d != 3 for d in deg):
            continue
        
        # Verify no triangles (bipartite, so guaranteed)
        # Count distinct cycle lengths
        cnt = _count_distinct_cycle_lengths(12, edges)
        
        if cnt < best_count:
            best_count = cnt
            best_edges = edges
            
            if cnt == 1:
                break
    
    if best_edges is None:
        best_edges = edges_c
    
    return best_edges

# EVOLVE-BLOCK-END
