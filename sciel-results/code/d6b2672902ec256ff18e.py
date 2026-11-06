import time
import random

def construct():
    """
    Constructs a C5-free graph with no independent set of size 9, maximizing n.
    Uses proven 8xK4 (n=32) as base, then attempts local search for n=33+.
    """
    deadline = time.time() + 115  # Safety margin before 150s limit
    
    # Proven n=32 solution: 8 disjoint K4's
    def make_k4_solution():
        n = 32
        edges = []
        for b in range(8):
            base = b * 4
            for i in range(4):
                for j in range(i + 1, 4):
                    edges.append([base + i, base + j])
        return {'n': n, 'edges': edges}
    
    fallback = make_k4_solution()
    
    # Try to find n=33, 34, or 35 solutions using structured search
    for target_n in [33, 34, 35]:
        if time.time() > deadline - 15:
            break
        result = _search_graph(target_n, deadline)
        if result is not None:
            return result
    
    return fallback

def _has_c5_fast(n, adj):
    """Check if graph has a C5 using adjacency matrix. Returns True if C5 exists."""
    # For each pair of adjacent vertices (v0,v1), look for paths of length 3 from v1 back to v0
    # through vertices not in {v0, v1}
    for v0 in range(n):
        for v1 in range(n):
            if v1 <= v0 or not adj[v0][v1]:
                continue
            # Find common neighbors of v0 and v1 (excluding each other)
            # Actually, we need: v0-v1-v2-v3-v4-v0 where all distinct
            # v2 is neighbor of v1 (not v0), v3 neighbor of v2 (not v0,v1), v4 neighbor of v3 (not v0,v1,v2) and neighbor of v0
            for v2 in range(n):
                if v2 == v0 or v2 == v1 or not adj[v1][v2]:
                    continue
                for v3 in range(n):
                    if v3 == v0 or v3 == v1 or v3 == v2 or not adj[v2][v3]:
                        continue
                    for v4 in range(n):
                        if v4 == v0 or v4 == v1 or v4 == v2 or v4 == v3:
                            continue
                        if adj[v3][v4] and adj[v4][v0]:
                            return True
    return False

def _alpha_bound(n, adj, vertex_order=None):
    """Upper bound on independence number using greedy coloring bound."""
    if vertex_order is None:
        vertex_order = list(range(n))
    # Simple bound: alpha <= n - min_vertex_cover
    # Or use: alpha <= sum over maximal cliques... 
    # Simpler: greedy maximal independent set gives lower bound, 
    # for upper bound use: alpha <= n - (min degree) * something
    # Use a quick bound: alpha <= n - max_matching_in_complement... 
    # Just return n as a loose upper bound for now, use exact check separately
    return n

def _exact_alpha(n, adj, timeout=5.0):
    """Exact maximum independent set size using branch and bound."""
    best = [0]
    start_time = time.time()
    
    # Order vertices by degree (higher degree first for pruning)
    deg = [sum(adj[v]) for v in range(n)]
    order = sorted(range(n), key=lambda v: -deg[v])
    
    def backtrack(candidates, current_size):
        if time.time() - start_time > timeout:
            return
        if not candidates:
            if current_size > best[0]:
                best[0] = current_size
            return
        if current_size + len(candidates) <= best[0]:
            return
        
        # Pick vertex with max degree in candidates for better pruning
        v = max(candidates, key=lambda u: sum(adj[u][w] for w in candidates if w != u))
        rest = [u for u in candidates if u != v]
        
        # Include v
        new_candidates = [u for u in rest if not adj[v][u]]
        backtrack(new_candidates, current_size + 1)
        
        # Exclude v
        backtrack(rest, current_size)
    
    backtrack(order, 0)
    return best[0]

def _is_valid(n, adj, alpha_timeout=5.0):
    """Check if graph is C5-free and has alpha < 9."""
    if _has_c5_fast(n, adj):
        return False
    alpha = _exact_alpha(n, adj, alpha_timeout)
    if alpha >= 9:
        return False
    return True

def _search_graph(target_n, deadline):
    """Structured search for valid graph with target_n vertices."""
    random.seed(12345)
    
    # Strategy 1: Start from 8xK4 and add vertices
    # Strategy 2: Try structured constructions
    
    # Try: 7xK4 + 1xK5? No, K5 contains C5.
    # Try: 7xK4 + one vertex connected to some vertices
    # With 7xK4 we have 28 vertices, alpha=7. Add 5 more vertices.
    # The 5 new vertices must not form an independent set of size 9 with existing ones.
    # Each new vertex can be connected to at most... 
    
    # Let's try a different structure: use a blow-up of a small C5-free graph
    # C5-free graphs include: forests, complete bipartite, etc.
    
    # Try: complete bipartite K(a,b) with a+b = target_n
    # K(a,b) is C5-free (bipartite). Alpha = max(a,b).
    # Need max(a,b) < 9, so a,b <= 8. Max n = 16. Too small.
    
    # Try: union of complete bipartite graphs
    # K(4,4) has 8 vertices, alpha=4, C5-free.
    # 4xK(4,4) = 32 vertices, alpha=16. Too high.
    # 2xK(4,4) = 16 vertices, alpha=8. Valid! But n=16.
    
    # Try: K(4,4) with internal edges added carefully
    # In K(4,4) with parts A={a1..a4}, B={b1..b4}, add edges within A and B
    # Adding edge a1-a2: creates C5? a1-a2-b1-c1... no, only 2 parts.
    # In bipartite graph, adding edge within one part: 
    # a1-a2 (new), a2-b1, b1-a3, a3-b2, b2-a1 - that's a 5-cycle! 
    # So adding any edge within a part of K(4,4) can create C5.
    
    # Let's try a more systematic approach: local search from random starts
    best_n_found = 0
    
    for attempt in range(500):
        if time.time() > deadline - 10:
            break
        
        # Generate a candidate graph
        adj = [[False] * target_n for _ in range(target_n)]
        
        # Start with a base structure: random graph with moderate density
        # Then iteratively remove edges to eliminate C5's
        p = 0.3
        for i in range(target_n):
            for j in range(i + 1, target_n):
                if random.random() < p:
                    adj[i][j] = True
                    adj[j][i] = True
        
        # Iteratively find and break C5's
        for iteration in range(200):
            if time.time() > deadline - 10:
                break
            
            # Find a C5
            c5 = _find_one_c5(target_n, adj)
            if c5 is None:
                break
            # Remove a random edge from the C5
            v0, v1, v2, v3, v4 = c5
            edge_idx = random.randint(0, 4)
            if edge_idx == 0:
                u, v = v0, v1
            elif edge_idx == 1:
                u, v = v1, v2
            elif edge_idx == 2:
                u, v = v2, v3
            elif edge_idx == 3:
                u, v = v3, v4
            else:
                u, v = v4, v0
            adj[u][v] = False
            adj[v][u] = False
        
        # Check if now C5-free
        if not _has_c5_fast(target_n, adj):
            # Check alpha
            alpha = _exact_alpha(target_n, adj, timeout=3.0)
            if alpha < 9:
                edges = []
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if adj[i][j]:
                            edges.append([i, j])
                return {'n': target_n, 'edges': edges}
    
    return None

def _find_one_c5(n, adj):
    """Find one C5 in the graph, return (v0,v1,v2,v3,v4) or None."""
    for v0 in range(n):
        for v1 in range(n):
            if v1 <= v0 or not adj[v0][v1]:
                continue
            for v2 in range(n):
                if v2 == v0 or v2 == v1 or not adj[v1][v2]:
                    continue
                for v3 in range(n):
                    if v3 == v0 or v3 == v1 or v3 == v2 or not adj[v2][v3]:
                        continue
                    for v4 in range(n):
                        if v4 == v0 or v4 == v1 or v4 == v2 or v4 == v3:
                            continue
                        if adj[v3][v4] and adj[v4][v0]:
                            return (v0, v1, v2, v3, v4)
    return None

def _local_search_improve(n, adj, deadline):
    """Try to improve a C5-free graph by adding edges while maintaining C5-free and alpha<9."""
    # This is complex, skip for now
    return adj

def construct():
    """
    Constructs a C5-free graph with no independent set of size 9, maximizing n.
    Uses proven 8xK4 (n=32) as base.
    """
    deadline = time.time() + 115  # Safety margin before 150s limit
    
    # Proven n=32 solution: 8 disjoint K4's
    def make_k4_solution():
        n = 32
        edges = []
        for b in range(8):
            base = b * 4
            for i in range(4):
                for j in range(i + 1, 4):
                    edges.append([base + i, base + j])
        return {'n': n, 'edges': edges}
    
    fallback = make_k4_solution()
    
    # Try to find n=33, 34, or 35 solutions
    for target_n in [33, 34, 35]:
        if time.time() > deadline - 20:
            break
        result = _search_graph(target_n, deadline)
        if result is not None:
            return result
    
    return fallback

def _has_c5_fast(n, adj):
    """Check if graph has a C5. Returns True if C5 exists."""
    for v0 in range(n):
        for v1 in range(n):
            if v1 <= v0 or not adj[v0][v1]:
                continue
            for v2 in range(n):
                if v2 == v0 or v2 == v1 or not adj[v1][v2]:
                    continue
                for v3 in range(n):
                    if v3 == v0 or v3 == v1 or v3 == v2 or not adj[v2][v3]:
                        continue
                    for v4 in range(n):
                        if v4 == v0 or v4 == v1 or v4 == v2 or v4 == v3:
                            continue
                        if adj[v3][v4] and adj[v4][v0]:
                            return True
    return False

def _exact_alpha(n, adj, timeout=5.0):
    """Exact maximum independent set size using branch and bound."""
    best = [0]
    start_time = time.time()
    
    deg = [sum(adj[v]) for v in range(n)]
    order = sorted(range(n), key=lambda v: -deg[v])
    
    def backtrack(candidates, current_size):
        if time.time() - start_time > timeout:
            return
        if not candidates:
            if current_size > best[0]:
                best[0] = current_size
            return
        if current_size + len(candidates) <= best[0]:
            return
        
        v = max(candidates, key=lambda u: sum(adj[u][w] for w in candidates if w != u))
        rest = [u for u in candidates if u != v]
        
        new_candidates = [u for u in rest if not adj[v][u]]
        backtrack(new_candidates, current_size + 1)
        backtrack(rest, current_size)
    
    backtrack(order, 0)
    return best[0]

def _find_one_c5(n, adj):
    """Find one C5, return tuple or None."""
    for v0 in range(n):
        for v1 in range(n):
            if v1 <= v0 or not adj[v0][v1]:
                continue
            for v2 in range(n):
                if v2 == v0 or v2 == v1 or not adj[v1][v2]:
                    continue
                for v3 in range(n):
                    if v3 == v0 or v3 == v1 or v3 == v2 or not adj[v2][v3]:
                        continue
                    for v4 in range(n):
                        if v4 == v0 or v4 == v1 or v4 == v2 or v4 == v3:
                            continue
                        if adj[v3][v4] and adj[v4][v0]:
                            return (v0, v1, v2, v3, v4)
    return None

def _search_graph(target_n, deadline):
    """Structured search for valid graph with target_n vertices."""
    random.seed(12345)
    
    for attempt in range(300):
        if time.time() > deadline - 10:
            break
        
        adj = [[False] * target_n for _ in range(target_n)]
        
        p = 0.35
        for i in range(target_n):
            for j in range(i + 1, target_n):
                if random.random() < p:
                    adj[i][j] = True
                    adj[j][i] = True
        
        # Break C5's
        for iteration in range(100):
            if time.time() > deadline - 10:
                break
            c5 = _find_one_c5(target_n, adj)
            if c5 is None:
                break
            v0, v1, v2, v3, v4 = c5
            edge_idx = random.randint(0, 4)
            edges_c5 = [(v0,v1), (v1,v2), (v2,v3), (v3,v4), (v4,v0)]
            u, v = edges_c5[edge_idx]
            adj[u][v] = False
            adj[v][u] = False
        
        if not _has_c5_fast(target_n, adj):
            alpha = _exact_alpha(target_n, adj, timeout=3.0)
            if alpha < 9:
                edges = []
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if adj[i][j]:
                            edges.append([i, j])
                return {'n': target_n, 'edges': edges}
    
    return None
