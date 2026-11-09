"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 15}

def _is_valid(edges, n):
    """Check if edges form a simple graph with no equal-degree path endpoints."""
    # Check for duplicates/loops
    edge_set = set()
    for e in edges:
        u, v = e
        if not isinstance(u, int) or not isinstance(v, int):
            return False
        if isinstance(u, bool) or isinstance(v, bool):
            return False
        if u == v:
            return False
        if u < 0 or u >= n or v < 0 or v >= n:
            return False
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
    
    # Build adjacency and degrees
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    deg = [len(adj[i]) for i in range(n)]
    
    # For every edge (u, v), check all paths s-u-v-t where s != t
    # s is neighbor of u (s != v), t is neighbor of v (t != u)
    for u, v in edges:
        for s in adj[u]:
            if s == v:
                continue
            for t in adj[v]:
                if t == u:
                    continue
            if s == t:
                continue
            if deg[s] == deg[t]:
                return False
    return True

def construct():
    n = P['n']
    deadline = time.time() + 75.0  # Safety margin before 90s kill
    
    # Greedy: start with empty graph, add edges in order of (u,v) pairs
    # Try to add edge (u,v) if it doesn't create a violation
    # We need to be smart about ordering to get dense graphs
    
    # Strategy: build a graph where degrees are all distinct or carefully paired
    # Key insight: if all vertices have distinct degrees, then NO two endpoints can have equal degree
    # But in a graph with n vertices, max degree is n-1, min is 0
    # For n=15, degrees range 0..14, so we can have at most 15 distinct degrees
    # A graph with all distinct degrees is impossible (handshaking: sum of degrees must be even,
    # and there's a known result that no simple graph has all distinct degrees)
    # But we can get close: use degrees like 1,2,3,...,14 (skip 0)
    # Sum = 1+2+...+14 = 105, which is odd -> impossible
    # Try 0,2,3,4,...,14: sum = 0+2+3+...+14 = 105, still odd
    # Try 1,3,4,5,...,14: sum = 1+3+4+...+14 = 105, still odd
    # We need even sum. Let's try to construct a near-regular or carefully designed graph.
    
    # Alternative: Use a graph where for every edge (u,v), all neighbors of u have distinct degrees 
    # from all neighbors of v. This is hard to guarantee greedily.
    
    # Better approach: Start with a known good structure.
    # Consider a path graph: 0-1-2-3-...-14
    # Degrees: 1,2,2,2,...,2,1 (endpoints degree 1, internal degree 2)
    # For edge (i, i+1): neighbors of i (excluding i+1) and neighbors of i+1 (excluding i)
    # For edge (0,1): s must be neighbor of 0 except 1 -> none (0 only connects to 1)
    # So no paths of length 3 through edge (0,1). Valid!
    # For edge (1,2): s in adj[1]\{2} = {0}, t in adj[2]\{1} = {3}
    #   deg[0]=1, deg[3]=2 -> not equal. OK.
    # For edge (2,3): s in adj[2]\{3} = {1}, t in adj[3]\{2} = {4}
    #   deg[1]=2, deg[4]=2 -> EQUAL! Violation!
    # So pure path fails.
    
    # Let's try a star: center 0, leaves 1-14
    # Degrees: deg[0]=14, deg[i]=1 for i=1..14
    # For edge (0, i): s in adj[0]\{i} = all other leaves, t in adj[i]\{0} = empty
    # No paths of length 3. Valid! Edge count = 14. Same as seed.
    
    # To get more edges, we need a more complex structure.
    # Let's try: make two hubs with distinct degree neighborhoods.
    
    # New strategy: Use a greedy algorithm with backtracking-free addition.
    # Try all edges in a specific order, add if valid.
    
    best_edges = []
    best_count = 0
    
    # Try multiple random orderings
    import random
    random.seed(42)
    
    all_possible = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    for attempt in range(500):
        if time.time() > deadline:
            break
        
        # Shuffle edge order
        edge_order = all_possible[:]
        random.shuffle(edge_order)
        
        current_edges = []
        edge_set = set()
        adj = [set() for _ in range(n)]
        deg = [0] * n
        
        for u, v in edge_order:
            if time.time() > deadline:
                break
            
            # Check if adding (u,v) would violate the constraint
            # We need to check: for the new edge (u,v), all paths s-u-v-t must have deg[s] != deg[t]
            # s in adj[u] (current), t in adj[v] (current)
            # Also, after adding, adj[u] gets v, adj[v] gets u
            # But the path s-u-v-t uses s from adj[u] BEFORE adding v, and t from adj[v] BEFORE adding u
            # Wait, no: the path is s-u-v-t, so s is adjacent to u, v is adjacent to u, 
            # t is adjacent to v, v is adjacent to t. 
            # s and t are in the FULL graph after adding the edge.
            # So s can be v? No, s != v because it's a simple path s-u-v-t, so s != v.
            # Similarly t != u.
            # s is any neighbor of u in the final graph except v.
            # t is any neighbor of v in the final graph except u.
            
            # After adding (u,v):
            # adj[u] will include v, adj[v] will include u
            # s ranges over (adj[u] ∪ {v}) \ {v} = adj[u]  (since v not yet in adj[u])
            # Wait: adj[u] currently doesn't have v. After adding, it does.
            # s must be in adj[u]_final \ {v} = adj[u]_current
            # t must be in adj[v]_final \ {u} = adj[v]_current
            # So we only need to check s in adj[u]_current, t in adj[v]_current
            
            valid = True
            for s in adj[u]:
                for t in adj[v]:
                    if s == t:
                        continue
                    # After adding edge, degrees change:
                    # deg[u] += 1, deg[v] += 1
                    # s and t degrees unchanged (unless s==v or t==u, but s != v, t != u)
                    # Wait, s is in adj[u], so s != u. s could be v? No, v not in adj[u] yet.
                    # So s != v. Similarly t != u.
                    # So deg[s] and deg[t] don't change when we add (u,v), unless s==v or t==u
                    # which we've excluded. But what if s==v? No, v not in adj[u].
                    # What if s is some other vertex? Its degree doesn't change.
                    # So deg[s] and deg[t] remain as they are.
                    if deg[s] == deg[t]:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                current_edges.append([u, v])
                edge_set.add((min(u,v), max(u,v)))
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges[:]
        
        if best_count >= 30:  # Good enough, stop early
            break
    
    # Convert to proper format
    result = []
    for e in best_edges:
        u, v = e
        result.append([int(u), int(v)])
    
    return result
# EVOLVE-BLOCK-END
