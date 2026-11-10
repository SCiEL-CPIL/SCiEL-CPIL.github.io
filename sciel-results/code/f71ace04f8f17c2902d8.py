import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Build PG(2,5) polarity graph: 31 vertices, 90 edges
    # Vertices: (x,y) in Z5^2 plus vertex 30 (infinity)
    # Edges: (x,y)~(x',y') iff x*x' + y*y' = x'+y' (polarity)
    # (x,y)~30 iff x+y = 0

    def get_neighbors(v):
        """Get neighbors of vertex v in PG(2,5)."""
        if v == 30:
            # Vertex 30 connects to (x,y) where x+y=0
            return [(x, (-x) % 5) for x in range(5)]
        x, y = v // 5, v % 5
        nbrs = []
        for x2 in range(5):
            for y2 in range(5):
                if (x * x2 + y * y2) % 5 == (x2 + y2) % 5:
                    nbrs.append(x2 * 5 + y2)
        if (x + y) % 5 == 0:
            nbrs.append(30)
        return nbrs

    # Build base edge list (90 edges)
    base_edges = set()
    for v in range(31):
        for u in get_neighbors(v):
            if v < u:
                base_edges.add((v, u))

    assert len(base_edges) == 90, f"Expected 90 edges, got {len(base_edges)}"

    # Precompute neighbor sets for fast C4 counting
    def build_adjacency(edges):
        adj = [set() for _ in range(31)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def count_c4(adj):
        """Count 4-cycles: for each pair of vertices, count common neighbors,
        C4 count = sum over all pairs of C(common_neighbors, 2) - (triangles contribution)
        Actually: number of C4 = (1/4) * sum over all ordered pairs (u,v) of |N(u) ∩ N(v)| choose 2
        But we need to be careful: this counts each C4 exactly 4 times (once per starting vertex)
        and each pair of opposite vertices in the C4 contributes 1 to the common neighbor count.
        
        Better: For each pair of vertices (u,v) that are NOT adjacent, count common neighbors.
        Each C4 has 2 pairs of opposite vertices. So C4 count = (1/2) * sum over non-adjacent pairs of C(common, 2).
        
        Wait, let me think again. A 4-cycle u-w-v-x-u has opposite pairs (u,v) and (w,x).
        For pair (u,v): common neighbors include w and x, so C(2,2)=1.
        For pair (w,x): common neighbors include u and v, so C(2,2)=1.
        So each C4 is counted twice (once for each opposite pair).
        But we also need to make sure u,v are not adjacent (otherwise it's not a C4 but contains a triangle).
        
        Actually the standard formula: number of C4 = (1/4) * sum_{u,v} C(|N(u) ∩ N(v)|, 2)
        where the sum is over all unordered pairs {u,v}.
        This works because:
        - If u,v are adjacent: common neighbors form triangles, and C(k,2) counts pairs of common neighbors
          which form 4-cycles u-a-v-b-u where a,b are common neighbors. But if u-v is an edge,
          u-a-v-b-u is still a valid 4-cycle (it's a cycle of length 4, not a triangle).
          Wait, u-a-v-b-u: edges are u-a, a-v, v-b, b-u. That's a 4-cycle. Yes.
        - Each 4-cycle is counted 4 times (once for each of its 4 vertices as "u" in the pair... no).
        
        Let me use the cleaner formula:
        For each unordered pair {u,v}, let c(u,v) = |N(u) ∩ N(v)|.
        The number of 4-cycles containing u and v as opposite vertices is C(c(u,v), 2) if u,v non-adjacent,
        or C(c(u,v), 2) if u,v adjacent (the 4-cycles go through two common neighbors).
        
        Actually, for ANY pair {u,v}, the number of 4-cycles where u and v are opposite is C(c(u,v), 2).
        Each 4-cycle has exactly 2 pairs of opposite vertices. So:
        Total C4 = (1/2) * sum over all unordered pairs {u,v} of C(c(u,v), 2).
        
        Let me verify with a simple 4-cycle: vertices 0,1,2,3 with edges 0-1,1-2,2-3,3-0.
        Pairs: {0,1}: c=0 (no common neighbors... wait N(0)={1,3}, N(1)={0,2}, intersection={}. c=0.
        {0,2}: N(0)={1,3}, N(2)={1,3}, intersection={1,3}, c=2. C(2,2)=1.
        {0,3}: N(0)={1,3}, N(3)={0,2}, intersection={}. c=0.
        {1,2}: c=0.
        {1,3}: N(1)={0,2}, N(3)={0,2}, intersection={0,2}, c=2. C(2,2)=1.
        {2,3}: c=0.
        Sum = 2. C4 = 2/2 = 1. Correct!
        """
        total = 0
        for u in range(31):
            for v in range(u + 1, 31):
                common = len(adj[u] & adj[v])
                if common >= 2:
                    total += common * (common - 1) // 2
        return total // 2

    # Convert base edges to list
    base_edge_list = list(base_edges)
    
    # The 26 candidate edges to add (all pairs not in base, total pairs = 465, base = 90, so 375 candidates)
    all_pairs = set()
    for i in range(31):
        for j in range(i + 1, 31):
            all_pairs.add((i, j))
    
    candidate_edges = list(all_pairs - base_edges)
    assert len(candidate_edges) == 375, f"Expected 375 candidates, got {len(candidate_edges)}"

    # For efficiency, precompute which candidates share vertices with which base edges
    # We'll use a fast incremental C4 count when adding/removing one edge.
    
    def count_c4_with_one_extra(adj_base, extra_edge):
        """Count C4 in graph = base + one extra edge, using precomputed adj_base."""
        u, v = extra_edge
        # New C4s must use the extra edge (u,v).
        # A C4 using edge (u,v) has the form u-v-a-b-u or u-v-b-a-u where a,b are common neighbors of u and v
        # in the base graph, AND a-b must be an edge in the base graph.
        # Wait, that's not right. A C4 containing edge (u,v) is u-v-x-y-u where x,y are vertices
        # such that v-x, x-y, y-u are edges. So x is a neighbor of v, y is a neighbor of u, and x-y is an edge.
        # Also x != u, y != v, x != y.
        
        # Actually simpler: the number of new C4s added by edge (u,v) equals the number of paths
        # of length 2 from u to v in the base graph, i.e., |N_base(u) ∩ N_base(v)|.
        # Wait no. A 4-cycle containing edge (u,v) is u-v-a-b-u. This requires:
        # v-a edge, a-b edge, b-u edge. So a ∈ N(v), b ∈ N(u), and a-b is an edge.
        # The number of such 4-cycles = number of edges (a,b) where a ∈ N(v), b ∈ N(u), a≠b.
        # This is NOT simply |N(u)∩N(v)|.
        
        # Let me reconsider. In the full graph (base + edge (u,v)), a C4 is a set of 4 vertices
        # forming a cycle. If the C4 uses edge (u,v), it has form u-v-x-y-u.
        # Conditions: (v,x) ∈ E, (x,y) ∈ E, (y,u) ∈ E, where E includes the extra edge but
        # x,y ≠ the other endpoint of extra edge in a way that creates shorter cycles.
        # Since (u,v) is the only new edge, all other edges are base edges.
        # So we need: v-x is base edge, x-y is base edge, y-u is base edge.
        # x ∈ N_base(v), y ∈ N_base(u), (x,y) ∈ base_edges.
        # Also x ≠ u (since u-v-x-u would be a triangle if x=u, but x=u means v-u which is the extra edge, not base)
        # y ≠ v similarly.
        # x ≠ y (otherwise u-v-x-u is a triangle).
        
        # Count: for each x in N_base(v), for each y in N_base(u), check if (x,y) is a base edge and x≠y.
        # This is O(deg(v) * deg(u)) which is fine.
        
        count = 0
        for x in adj_base[v]:
            if x == u:
                continue
            for y in adj_base[u]:
                if y == v or y == x:
                    continue
                if (min(x,y), max(x,y)) in base_edges:
                    count += 1
        return count

    # Better approach: use the full C4 count formula but incrementally.
    # When we add edge (u,v), the change in C4 count is:
    # For each pair {a,b} where a ∈ N(u) and b ∈ N(v) (in base), and a≠b, and (a,b) is an edge:
    #   This creates a new C4 u-v-b-a-u (or u-v-a-b-u).
    # Plus: the edge (u,v) itself creates new "common neighbor" contributions.
    #   For any vertex w that is a common neighbor of u and v in base, adding (u,v) means
    #   w is now also connected to both u and v via the direct edge... no, w's adjacency doesn't change.
    #   
    # Actually the cleanest way: delta_C4 = number of length-2 paths from u to v in base graph
    # that don't go through the edge (u,v) itself (which doesn't exist in base).
    # A length-2 path u-x-v means x is a common neighbor. But that gives a triangle u-x-v-u, not a C4.
    # 
    # I'm overcomplicating this. Let me just use the full count formula with the modified adjacency.
    # For 31 vertices, the full count is O(31^2 * 31) ≈ 30k operations, very fast.

    def count_c4_full(adj):
        total = 0
        for u in range(31):
            for v in range(u + 1, 31):
                common = len(adj[u] & adj[v])
                if common >= 2:
                    total += common * (common - 1) // 2
        return total // 2

    # Initial state: base + one random candidate edge
    random.seed(42)
    
    # Try all single-edge additions first to find the best starting point
    base_adj = build_adjacency(base_edge_list)
    
    best_initial = None
    best_initial_c4 = float('inf')
    
    for extra in candidate_edges:
        adj = [set(s) for s in base_adj]
        u, v = extra
        adj[u].add(v)
        adj[v].add(u)
        c4 = count_c4_full(adj)
        if c4 < best_initial_c4:
            best_initial_c4 = c4
            best_initial = extra
    
    current_extra = best_initial
    current_adj = [set(s) for s in base_adj]
    u, v = current_extra
    current_adj[u].add(v)
    current_adj[v].add(u)
    current_c4 = best_initial_c4
    
    best_extra = current_extra
    best_c4 = current_c4
    
    # Simulated annealing: try swapping the extra edge
    T = 10.0
    cooling = 0.9995
    
    while time.time() < deadline:
        # Pick a random candidate to swap to
        new_extra = random.choice(candidate_edges)
        if new_extra == current_extra:
            continue
        
        # Compute C4 with new extra
        adj_new = [set(s) for s in base_adj]
        nu, nv = new_extra
        adj_new[nu].add(nv)
        adj_new[nv].add(nu)
        new_c4 = count_c4_full(adj_new)
        
        delta = new_c4 - current_c4
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_extra = new_extra
            current_adj = adj_new
            current_c4 = new_c4
            if new_c4 < best_c4:
                best_c4 = new_c4
                best_extra = new_extra
        
        T *= cooling
        if T < 0.01:
            T = 0.01
    
    # Return base edges + best extra edge
    result = list(base_edges) + [list(best_extra)]
    return result
