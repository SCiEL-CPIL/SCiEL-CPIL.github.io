"""Construct a C5-free graph with no independent set of size 9, maximizing n."""
# EVOLVE-BLOCK-START
import random
import time

def _has_c5(adj, n):
    """Check if graph has a 5-cycle using bitsets."""
    # For each pair of non-adjacent vertices, check if they share two common neighbors
    # that are connected to each other (forming a 4-path), then check for the closing edge
    # Actually, let's use a simpler approach: check all 5-cycles
    # A 5-cycle is v0-v1-v2-v3-v4-v0 where all edges exist
    # We can check: for each triple (a,b,c) with edges a-b, b-c,
    # find d adjacent to c but not a, then check if d is adjacent to some e adjacent to a
    # This is getting complex. Let's use a direct but optimized check.
    
    # Use the fact that C5-free means no induced C5 either (since C5 is its own complement's cycle)
    # Actually C5-free as subgraph: check all 5-cycles
    
    # Efficient method: For each edge (u,v), the neighbors of u and v (excluding each other)
    # should not contain a path of length 3 connecting back.
    # Simpler: just check all 5-tuples with early termination
    
    # Better: use adjacency matrix and check
    # For a 5-cycle v0-v1-v2-v3-v4-v0:
    # v0 and v2 are non-adjacent (in a simple 5-cycle)
    # v0 and v3 are non-adjacent
    # v1 and v3 are adjacent
    # v1 and v4 are non-adjacent
    # v2 and v4 are adjacent
    
    # Let's just do a bounded search
    adj_set = [set(adj[i]) for i in range(n)]
    
    for v0 in range(n):
        n0 = adj_set[v0]
        for v1 in n0:
            n1 = adj_set[v1] - {v0}
            for v2 in n1:
                if v2 == v0:
                    continue
                n2 = adj_set[v2] - {v0, v1}
                for v3 in n2:
                    if v3 == v0 or v3 == v1:
                        continue
                    n3 = adj_set[v3] - {v0, v1, v2}
                    for v4 in n3:
                        if v4 == v0 or v4 == v1 or v4 == v2:
                            continue
                        # Check if v4 is adjacent to v0
                        if v0 in adj_set[v4]:
                            return True
    return False

def _independence_number(adj, n):
    """Compute independence number using Bron-Kerbosch on complement."""
    # Independence number = clique number of complement
    # Use greedy + exact for small graphs
    comp_adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if j not in adj[i]:
                comp_adj[i].add(j)
                comp_adj[j].add(i)
    
    # Use a simple branch and bound for clique number
    best = [0]
    def bron_kerbosch(R, P, X):
        if not P and not X:
            if len(R) > best[0]:
                best[0] = len(R)
            return
        # Choose pivot from P ∪ X
        if P or X:
            pivot = list(P | X)[0]
            candidates = [v for v in P if v not in comp_adj[pivot]]
        else:
            candidates = list(P)
        
        for v in candidates:
            bron_kerbosch(R | {v}, P & comp_adj[v], X & comp_adj[v])
            P = P - {v}
            X = X | {v}
    
    bron_kerbosch(set(), set(range(n)), set())
    return best[0]

def construct():
    """
    Returns a simple graph on n vertices (0..n-1) that is:
    - C5-free (no 5-cycle as a subgraph)
    - has no independent set of size 9
    
    Construction: Start with 8 disjoint K4s (n=32, alpha=8),
    then try to add edges between components to reduce alpha while
    maintaining C5-freeness.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n = 32
    # Build adjacency as sets
    adj = [set() for _ in range(n)]
    edges = []
    
    # Create 8 disjoint K4s
    for b in range(8):
        base = b * 4
        for i in range(4):
            for j in range(i + 1, 4):
                u = base + i
                v = base + j
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
    
    # Try to add edges between blocks to reduce independence number
    # Current alpha = 8 (pick one from each K4)
    # We need alpha <= 8, so we're already good.
    # But we want to maximize n, so let's try n=33,34,35
    
    # For n > 32, we need to add vertices and edges
    # Strategy: try random greedy construction for n=33,34,35
    
    best_result = {'n': 32, 'edges': edges}
    
    for target_n in [35, 34, 33]:
        if time.time() > deadline - 5:
            break
        
        # Try multiple random attempts
        for attempt in range(50):
            if time.time() > deadline - 2:
                break
            
            random.seed(42 + attempt * 1000 + target_n)
            
            # Start with empty graph, add edges greedily
            adj_try = [set() for _ in range(target_n)]
            edges_try = []
            
            # Pre-generate all possible edges in random order
            all_edges = []
            for i in range(target_n):
                for j in range(i+1, target_n):
                    all_edges.append((i, j))
            random.shuffle(all_edges)
            
            # Greedy: add edge if it doesn't create C5 and alpha stays <= 8
            # But checking alpha each time is expensive.
            # Alternative: start from the 8-K4 construction and add vertices/edges
            
            # Better approach: start with 8 K4s + extra vertices, then add edges
            adj_try = [set() for _ in range(target_n)]
            edges_try = []
            
            # Place 8 K4s in first 32 vertices
            for b in range(8):
                base = b * 4
                for i in range(4):
                    for j in range(i + 1, 4):
                        u = base + i
                        v = base + j
                        adj_try[u].add(v)
                        adj_try[v].add(u)
                        edges_try.append([u, v])
            
            # Now add edges involving vertices 32..target_n-1
            # and between existing vertices to reduce alpha
            
            # First, connect new vertices to existing ones
            for nv in range(32, target_n):
                # Connect to random vertices in each block
                for b in range(8):
                    base = b * 4
                    # Connect to 1-2 vertices in this block
                    num_conn = random.randint(1, 2)
                    targets = random.sample(range(base, base+4), num_conn)
                    for t in targets:
                        if t not in adj_try[nv]:
                            adj_try[nv].add(t)
                            adj_try[t].add(nv)
                            edges_try.append([nv, t])
                
                # Check C5 after adding these edges
                if _has_c5(adj_try, target_n):
                    # Remove some edges to fix
                    # This is complex; just skip this attempt
                    break
            else:
                # No break, so no C5 detected yet
                # Now try to add more edges between blocks to reduce alpha
                # Current alpha might be > 8 if new vertices form independent sets
                
                # Try adding edges between random pairs of vertices from different blocks
                cross_edges = []
                for i in range(target_n):
                    for j in range(i+1, target_n):
                        if j not in adj_try[i]:
                            # Check if they're in different "blocks" (roughly)
                            block_i = i // 4 if i < 32 else 8
                            block_j = j // 4 if j < 32 else 8
                            if block_i != block_j:
                                cross_edges.append((i, j))
                
                random.shuffle(cross_edges)
                
                added = 0
                for u, v in cross_edges:
                    if time.time() > deadline - 1:
                        break
                    if v in adj_try[u]:
                        continue
                    
                    # Try adding edge
                    adj_try[u].add(v)
                    adj_try[v].add(u)
                    edges_try.append([u, v])
                    
                    # Check C5
                    if _has_c5(adj_try, target_n):
                        # Remove this edge
                        adj_try[u].remove(v)
                        adj_try[v].remove(u)
                        edges_try.pop()
                    else:
                        added += 1
                        # Optionally check alpha periodically
                        if added % 5 == 0:
                            alpha = _independence_number(adj_try, target_n)
                            if alpha >= 9:
                                # Too much independence, remove last few edges
                                for _ in range(min(added, 5)):
                                    if edges_try:
                                        eu, ev = edges_try.pop()
                                        adj_try[eu].discard(ev)
                                        adj_try[ev].discard(eu)
                                break
                
                # Final check
                if not _has_c5(adj_try, target_n):
                    alpha = _independence_number(adj_try, target_n)
                    if alpha < 9:
                        best_result = {'n': target_n, 'edges': edges_try}
                        break  # Found valid graph for this target_n
    
    return best_result
# EVOLVE-BLOCK-END
