import time
import random

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110.0  # safety margin before 120s limit
    
    n = 16
    # All possible edges in K_16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Precompute for each edge, the set of edges that would form a triangle with it
    # For edge (u,v), any edge (u,w) or (v,w) where w != u,v would form a triangle
    edge_to_forbidden = {}
    for e_idx, (u, v) in enumerate(all_edges):
        forbidden = set()
        for f_idx, (a, b) in enumerate(all_edges):
            if f_idx == e_idx:
                continue
            # Check if adding (a,b) to a graph containing (u,v) creates a triangle
            # Triangle if {a,b} shares two vertices with {u,v} and the third is shared
            # Actually: (u,v) and (a,b) form a triangle if there exists w such that
            # (u,v), (u,w), (v,w) are all present. But we're checking pairwise.
            # Two edges (u,v) and (a,b) are incompatible (form triangle) if:
            # They share exactly one vertex, say u=a, and then we need edge (v,b)
            # OR they share no vertices but... no, two edges alone can't form a triangle.
            # Wait: the constraint is that the FINAL graph must be triangle-free.
            # So if we include edges (u,v), (u,w), (v,w), that's a triangle.
            # So when building incrementally, if we have (u,v) and (u,w), we cannot add (v,w).
            # For pairwise incompatibility: edge e1=(u,v) and e2=(a,b) are "conflicting" 
            # if adding both would force a triangle? No, two edges don't form a triangle.
            # The triangle constraint is a 3-edge constraint.
            pass
        
        # For the greedy approach, we track adjacency and check triangle-freeness dynamically
        edge_to_forbidden[e_idx] = set()
    
    # Better approach: use bitsets for adjacency and do greedy with backtracking
    
    # Build the Groetzsch graph as a starting core (vertices 0-10)
    # Then use greedy fill with bitsets to maximize edges on all 16 vertices
    
    def groetzsch_edges():
        """Return edges of the Grötzsch graph on vertices 0..10."""
        cycle = [(i, (i + 1) % 5) for i in range(5)]
        edges = [list(e) for e in cycle]
        for u, w in cycle:
            edges += [[u, 5 + w], [w, 5 + u]]
        edges += [[5 + u, 10] for u in range(5)]
        return edges
    
    def is_triangle_free(adj, u, v, n):
        """Check if adding edge (u,v) to graph with adjacency adj keeps it triangle-free."""
        # A triangle exists if there's a common neighbor of u and v
        common = adj[u] & adj[v]
        return common == 0
    
    def is_not_3colorable(adj, n):
        """Check if graph with adjacency adj is NOT 3-colorable using backtracking."""
        # Try to 3-color the graph
        color = [-1] * n
        # Order vertices by degree descending for better pruning
        order = sorted(range(n), key=lambda v: -adj[v].bit_count())
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            # Determine which colors are forbidden
            forbidden = 0
            for w in range(n):
                if (adj[v] >> w) & 1:
                    if color[w] >= 0:
                        forbidden |= (1 << color[w])
            for c in range(3):
                if not (forbidden >> c) & 1:
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            return False
        
        return not backtrack(0)
    
    def greedy_fill(start_edges, adj, n, deadline, seed):
        """Greedy fill: try to add edges in random order, keep if triangle-free."""
        random.seed(seed)
        
        # Current adjacency
        cur_adj = [adj[i] for i in range(n)]
        cur_edges = [tuple(e) for e in start_edges]
        
        # All possible edges not yet in graph
        existing = set()
        for u, v in cur_edges:
            existing.add((u, v))
            existing.add((v, u))
        
        candidates = [(i, j) for i in range(n) for j in range(i+1, n) if (i,j) not in existing]
        random.shuffle(candidates)
        
        # Greedy: add edge if it doesn't create a triangle
        for u, v in candidates:
            if time.time() > deadline:
                break
            if is_triangle_free(cur_adj, u, v, n):
                # Add the edge
                cur_adj[u] |= (1 << v)
                cur_adj[v] |= (1 << u)
                cur_edges.append((u, v))
        
        return cur_edges, cur_adj
    
    def local_search(edges, adj, n, deadline, seed):
        """Try to improve by removing and re-adding edges."""
        random.seed(seed)
        cur_adj = [adj[i] for i in range(n)]
        cur_edges = list(edges)
        
        existing = set()
        for u, v in cur_edges:
            existing.add((u, v))
            existing.add((v, u))
        
        candidates = [(i, j) for i in range(n) for j in range(i+1, n) if (i,j) not in existing]
        random.shuffle(candidates)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for u, v in candidates:
                if time.time() > deadline:
                    break
                if is_triangle_free(cur_adj, u, v, n):
                    cur_adj[u] |= (1 << v)
                    cur_adj[v] |= (1 << u)
                    cur_edges.append((u, v))
                    existing.add((u, v))
                    existing.add((v, u))
                    candidates.remove((u, v))
                    improved = True
        
        return cur_edges, cur_adj
    
    best_edges = []
    best_count = 0
    
    # Start with Groetzsch core
    core_edges = groetzsch_edges()
    core_adj = [0] * n
    for u, v in core_edges:
        core_adj[u] |= (1 << v)
        core_adj[v] |= (1 << u)
    
    # Verify core is not 3-colorable
    if not is_not_3colorable(core_adj, n):
        # Fallback: just return core
        return core_edges
    
    # Do multiple random restarts of greedy fill
    for seed in range(200):
        if time.time() > deadline:
            break
        
        edges, adj = greedy_fill(core_edges, core_adj, n, deadline, seed)
        
        # Verify the result is still not 3-colorable
        if is_not_3colorable(adj, n):
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = [list(e) for e in edges]
        
        # Try local search from this solution
        if time.time() < deadline:
            edges2, adj2 = local_search(edges, adj, n, deadline, seed + 10000)
            if is_not_3colorable(adj2, n):
                if len(edges2) > best_count:
                    best_count = len(edges2)
                    best_edges = [list(e) for e in edges2]
    
    # Also try starting from empty graph with different cores
    # Try Mycielski of C5 is Grötzsch, so that's our core
    # Also try other 4-chromatic triangle-free graphs on 11 vertices
    
    if not best_edges:
        return core_edges
    
    return best_edges
# EVOLVE-BLOCK-END
