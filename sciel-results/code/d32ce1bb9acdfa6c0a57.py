import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Candidate Paley graphs: q must be prime, q % 4 == 1
    candidates = [37, 29, 17]
    
    best_graph = None
    best_n = 0
    
    for q in candidates:
        if time.time() > deadline:
            break
            
        # Generate Paley graph P_q
        # Vertices 0..q-1
        # Edge (u,v) if (v-u) % q is a quadratic residue
        squares = set()
        for x in range(1, q):
            squares.add((x * x) % q)
        
        adj = [set() for _ in range(q)]
        edges = []
        for i in range(q):
            for j in range(i + 1, q):
                if (j - i) % q in squares:
                    adj[i].add(j)
                    adj[j].add(i)
                    edges.append([i, j])
        
        # Check K4
        if not has_K4(adj, q):
            continue
            
        # Check Independent Set of size 6
        if has_independent_set_6(adj, q):
            continue
            
        # Valid
        if q > best_n:
            best_n = q
            best_graph = {'n': q, 'edges': edges}
            
    # Fallback to P_17 if others failed (shouldn't happen for 17)
    if best_graph is None:
        # Manually construct P_17
        q = 17
        squares = set()
        for x in range(1, q):
            squares.add((x * x) % q)
        edges = []
        for i in range(q):
            for j in range(i + 1, q):
                if (j - i) % q in squares:
                    edges.append([i, j])
        best_graph = {'n': q, 'edges': edges}
        
    return best_graph

def has_K4(adj, n):
    # Check if any 4 vertices form a clique
    # For each pair (u, v), find common neighbors w.
    # If any w has a common neighbor x with u, v, w, then K4.
    # Simpler: Iterate all triples (u, v, w) that form a triangle.
    # Check if any vertex x is connected to u, v, w.
    
    # Precompute common neighbors for speed?
    # n <= 37, so O(n^4) is ~ 1.8M ops. Fast enough.
    
    for u in range(n):
        for v in range(u + 1, n):
            if v not in adj[u]:
                continue
            # u-v is an edge
            # Find common neighbors w
            common_uv = adj[u] & adj[v]
            for w in common_uv:
                if w <= v:
                    continue
                # u, v, w form a triangle
                # Check if any x is connected to u, v, w
                # x must be in adj[u] & adj[v] & adj[w]
                # We already know w is in adj[u] & adj[v].
                # So we need x in adj[w] & adj[u] & adj[v]
                # Basically, is there a 4th vertex?
                # The set of common neighbors of u, v, w is adj[u] & adj[v] & adj[w]
                # If this set is non-empty, we have a K4.
                # Note: u, v, w are in the set? No, adj doesn't contain self.
                # So if (adj[u] & adj[v] & adj[w]) is not empty, K4 exists.
                if adj[u] & adj[v] & adj[w]:
                    return True
    return False

def has_independent_set_6(adj, n):
    # Check if there is an independent set of size 6.
    # Use backtracking with pruning.
    # An independent set is a set of vertices with no edges between them.
    # This is equivalent to a clique in the complement graph.
    
    # Build complement adjacency
    comp_adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if j not in adj[i]:
                comp_adj[i].add(j)
                comp_adj[j].add(i)
                
    # Find clique of size 6 in comp_adj
    # Use Bron-Kerbosch or simple DFS
    # Since we only need to know if size 6 exists, we can stop early.
    
    def dfs(current_clique, available):
        if len(current_clique) + len(available) < 6:
            return False
        if len(current_clique) == 6:
            return True
        
        # Pruning: if available is empty, return False
        if not available:
            return False
            
        # Pick a vertex from available
        # Heuristic: pick vertex with highest degree in available?
        # Or just first one.
        # To speed up, we can order vertices by degree in the subgraph induced by available.
        
        # Simple DFS
        for v in list(available):
            new_clique = current_clique | {v}
            # New available: neighbors of v in available
            new_available = available & comp_adj[v]
            if dfs(new_clique, new_available):
                return True
            # Remove v from available for next iterations? 
            # In standard clique finding, we pass (available - {v} - neighbors(v)) to the "exclude" set.
            # Here we are looking for ANY clique.
            # We should remove v from available to avoid re-exploring subsets not containing v.
            available.discard(v)
            
        return False
        
    # Start with all vertices
    # To make it faster, sort vertices by degree in comp_adj?
    # Let's just run it. n=37 is small.
    all_vertices = set(range(n))
    return dfs(set(), all_vertices)
