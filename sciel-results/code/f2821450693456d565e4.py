"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Keep under 120s wall-clock with safety margin
    n = 20
    vertices = list(range(n))
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    best_edges = None
    best_count = 0
    
    def count_edges(edges):
        return len(edges)
    
    # Fast validity check: can we add edge (u, v) to current edge set?
    # Constraint: no three vertices have three common neighbors (no K3,3)
    # For a graph G with edge set E, adding (u,v) is valid iff:
    # There is no triple {a,b,c} such that |N(a) ∩ N(b) ∩ N(c)| >= 3
    # After adding (u,v), check if any triple now has 3 common neighbors.
    # Efficient check: for each pair (b,c), if v is neighbor of b and c,
    # then check if |N(u) ∩ N(b) ∩ N(c)| == 2 (would become 3).
    # Similarly for pairs where u is neighbor of b and c.
    
    def is_valid_addition(u, v, adj):
        # adj: set of sets, adj[x] = set of neighbors of x
        # Check triples containing u (but not v as a vertex in triple, since N(u)∩N(v) unchanged)
        # Actually, adding (u,v):
        # - For any triple T not containing u or v: C(T) unchanged.
        # - For triple {u, b, c}: C' = (N(u)∪{v}) ∩ N(b) ∩ N(c) = C ∪ ({v} ∩ N(b) ∩ N(c))
        #   So |C'| = |C| + 1 if v ∈ N(b) ∩ N(c)
        #   Invalid if v ∈ N(b) ∩ N(c) and |N(u) ∩ N(b) ∩ N(c)| == 2
        # - For triple {v, b, c}: similarly, invalid if u ∈ N(b) ∩ N(c) and |N(v) ∩ N(b) ∩ N(c)| == 2
        # - For triple {u, v, w}: C = N(u)∩N(v)∩N(w), unchanged by adding (u,v)
        
        # Check all pairs (b, c) where b,c != u,v
        for b in vertices:
            if b == u or b == v:
                continue
            for c in vertices:
                if c == u or c == v or c == b:
                    continue
                # Check triple {u, b, c}
                if v in adj[b] and v in adj[c]:
                    # Count common neighbors of u, b, c (excluding potential v since not yet added)
                    common = adj[u] & adj[b] & adj[c]
                    if len(common) >= 2:
                        return False
                # Check triple {v, b, c}
                if u in adj[b] and u in adj[c]:
                    common = adj[v] & adj[b] & adj[c]
                    if len(common) >= 2:
                        return False
        return True
    
    def build_greedy(order, adj, edges_set):
        """Try to add edges in given order, returning new edge set if valid."""
        for (u, v) in order:
            if (u, v) in edges_set or (v, u) in edges_set:
                continue
            if is_valid_addition(u, v, adj):
                edges_set.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
        return edges_set, adj
    
    def local_search(edges_set, adj, max_steps=2000):
        """Local search: remove random edge, try to re-add rejected edges."""
        if not edges_set:
            return edges_set, adj
        
        current_edges = set(edges_set)
        current_adj = {v: set(adj[v]) for v in vertices}
        current_count = len(current_edges)
        
        for step in range(max_steps):
            # Pick a random edge to remove
            edge_to_remove = random.choice(list(current_edges))
            u, v = edge_to_remove
            current_edges.discard((u, v))
            current_edges.discard((v, u))
            current_adj[u].discard(v)
            current_adj[v].discard(u)
            
            # Try to add back all edges in some order
            # Use a greedy pass over all non-current edges
            candidates = [e for e in all_edges if e not in current_edges]
            random.shuffle(candidates)
            
            for (a, b) in candidates:
                if (a, b) in current_edges or (b, a) in current_edges:
                    continue
                if is_valid_addition(a, b, current_adj):
                    current_edges.add((a, b))
                    current_adj[a].add(b)
                    current_adj[b].add(a)
            
            if len(current_edges) > current_count:
                current_count = len(current_edges)
                edges_set = set(current_edges)
                adj = {v: set(current_adj[v]) for v in vertices}
        
        return edges_set, adj
    
    # Multiple restarts with random edge orderings
    restarts = 50
    for r in range(restarts):
        if time.time() > deadline - 10:
            break
        
        adj = {v: set() for v in vertices}
        edges_set = set()
        
        # Random ordering of edges
        shuffled_edges = all_edges[:]
        random.shuffle(shuffled_edges)
        
        # Greedy construction
        edges_set, adj = build_greedy(shuffled_edges, adj, edges_set)
        current_count = len(edges_set)
        
        # Local search improvement
        if current_count > best_count and time.time() < deadline - 15:
            edges_set, adj = local_search(edges_set, adj, max_steps=500)
            current_count = len(edges_set)
        
        if current_count > best_count:
            best_count = current_count
            best_edges = [(u, v) for (u, v) in edges_set]
    
    # Also try a structured approach: start with complement of a random graph
    if best_edges is None:
        best_edges = []
    
    # Convert to list of [u, v] format
    result = []
    seen = set()
    for (u, v) in best_edges:
        if u < v:
            key = (u, v)
        else:
            key = (v, u)
        if key not in seen:
            seen.add(key)
            result.append([u, v])
    
    return result
# EVOLVE-BLOCK-END
