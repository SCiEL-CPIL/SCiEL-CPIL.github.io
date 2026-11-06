"""Construct a graph on 31 vertices with 91 edges minimizing the number of 4-cycles."""
import time
import random

N = 31
M = 91
DEADLINE = 115.0  # safety margin before 150s kill

def construct():
    start_time = time.time()
    
    # --- Build a strong C4-sparse initial graph using GF(5) polarity structure ---
    # Vertices: 0..24 are (x, y) in GF(5)^2, 25..30 are 6 extra points
    # Use a polarity of the projective plane PG(2,5) projected/adjusted, 
    # or more simply: use a known C4-free construction for 25 vertices (affine plane AG(2,5) based)
    # and then add 6 vertices + edges to reach 31 vertices, 91 edges.
    
    # Actually, let's use a simpler strong start: 
    # The polarity graph of PG(2,5) has 31 vertices and 31*? edges... 
    # Let me use a different approach: start with a random graph and use local search.
    # But a structured start is better. Let me build from a known C4-free 31-vertex graph with 90 edges.
    
    # Construction via finite field GF(5):
    # Label vertices 0..24 as (i,j) in Z5 x Z5, and 25..30 as special points.
    # For the 25 affine points, use the construction where (i,j) ~ (i',j') iff j-j' = a*(i-i')^2 for some a in {1,2,3,4}
    # This gives each vertex degree 4*2=8 within the 25, total edges = 25*8/2 = 100. Too many.
    
    # Let me use a different approach. Start with a random graph and do heavy local search.
    # Or use a structured start from a polarity.
    
    # Polarity graph of PG(2,q) for q=5: vertices are points+lines of PG(2,5), 31+31=62 vertices. Too many.
    
    # Let me just use a good random start with local search. The key insight from the cell findings:
    # bitmask + O(1) delta C4 allows 200k+ iterations. Let's do that.
    
    # For a better start, use a greedy construction that avoids C4s as much as possible.
    
    def count_c4s(adj_matrix):
        """Count 4-cycles in graph given as adjacency matrix (list of int bitmasks)."""
        c4_count = 0
        n = len(adj_matrix)
        for i in range(n):
            for j in range(i+1, n):
                common = adj_matrix[i] & adj_matrix[j]
                c4_count += common.bit_count()
        return c4_count // 2  # each C4 counted twice (once per pair of opposite vertices)
    
    def build_greedy_c4_sparse():
        """Build a graph with M edges, greedily adding edges that create fewest new C4s."""
        adj = [0] * N
        edges = []
        
        # Precompute all possible edges
        all_edges = [(i, j) for i in range(N) for j in range(i+1, N)]
        random.shuffle(all_edges)
        
        # For each candidate edge, compute how many C4s it would create
        # An edge (u,v) creates C4s = number of common neighbors of u and v
        # (each common neighbor w gives a C4 u-w-v-? ... wait, no)
        # Actually, adding edge (u,v) creates C4s where u and v are adjacent in the C4.
        # A C4 containing edge (u,v) has form u-v-x-y-u where x is common neighbor of v and y, 
        # and y is common neighbor of u and x. This is complex.
        # Simpler: adding edge (u,v) creates new C4s = number of length-2 paths from u to v 
        # that don't use the edge (u,v) itself. That is, for each common neighbor w of u and v,
        # the path u-w-v is a length-2 path. But a C4 needs a 4-cycle, so we need u-w-v-x-u 
        # where x is adjacent to both u and v. So new C4s from adding (u,v) = |N(u) ∩ N(v)|.
        
        # Wait, that's not right either. If we add edge (u,v), a new C4 is formed for each 
        # pair (w,x) where w ~ u, w ~ v, x ~ u, x ~ v, and w != x. That's |N(u)∩N(v)| choose 2?
        # No. A 4-cycle u-v-w-x-u requires: u~v (new edge), v~w, w~x, x~u.
        # So for each w in N(v) and x in N(u) with w~x and w!=x, we get a C4.
        # This is complex. Let's just use: new C4s = number of common neighbors of u and v,
        # because for each common neighbor w, if there exists x such that x~u, x~v, x~w... 
        # Actually the simplest correct formula: adding edge (u,v) creates exactly 
        # |{w : w~u and w~v}| new triangles, not C4s.
        
        # For C4s: adding edge (u,v) creates a C4 for each pair of vertices (a,b) 
        # such that a~u, a~v, b~u, b~v, a~b, a!=b. That's |N(u)∩N(v)| choose 2? No,
        # we need a~b too. This is getting complicated.
        
        # Let's just use a simpler heuristic: prefer edges between vertices with few common neighbors.
        
        for (u, v) in all_edges:
            if len(edges) >= M:
                break
            common = (adj[u] & adj[v]).bit_count()
            if common < 3 or len(edges) < M - 20:  # allow some C4s if needed
                # Check it's not already an edge
                if not (adj[u] >> v) & 1:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append([u, v])
        
        # If we don't have enough edges, add random ones
        if len(edges) < M:
            for (u, v) in all_edges:
                if len(edges) >= M:
                    break
                if not (adj[u] >> v) & 1:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append([u, v])
        
        return adj, edges
    
    # Build initial graph
    adj, edges = build_greedy_c4_sparse()
    
    # Compute initial C4 count
    current_c4s = count_c4s(adj)
    best_c4s = current_c4s
    best_adj = adj[:]
    
    # Local search: swap edges to reduce C4 count
    def delta_c4_remove_add(adj, u1, v1, u2, v2):
        """Compute change in C4 count when removing edge (u1,v1) and adding edge (u2,v2)."""
        # This is complex, so we'll just recompute locally for candidate swaps
        pass
    
    # For efficiency, let's use a different approach: 
    # For each edge in the graph, try removing it and adding a non-edge, compute delta.
    # Delta in C4s from removing edge (u,v): 
    #   - C4s destroyed: those using edge (u,v). Count = for each pair (a,b) with a~u,a~v,b~u,b~v,a~b,a!=b: 1
    #     Simplified: the number of C4s containing edge (u,v) equals the number of common neighbors 
    #     of u and v that are also connected in a way that forms a C4. 
    #     Actually, a C4 containing edge (u,v) has the form u-v-a-b-u where a~v, a~b, b~u.
    #     So count = number of pairs (a,b) such that a~v, b~u, a~b, a!=u, a!=v, b!=u, b!=v.
    #     This is hard to compute quickly.
    
    # Let's use a simpler but effective approach: 
    # Randomly try swaps and keep if C4 count decreases.
    
    import numpy as np
    
    # Convert adj to numpy for faster operations
    adj_np = np.array(adj, dtype=np.uint64)
    
    def count_c4s_np(adj_np):
        """Count C4s using numpy."""
        c4 = 0
        for i in range(N):
            for j in range(i+1, N):
                common = adj_np[i] & adj_np[j]
                c4 += common.bit_count()
        return c4 // 2
    
    # Verify initial count
    current_c4s = count_c4s_np(adj_np)
    best_c4s = current_c4s
    best_adj_np = adj_np.copy()
    
    # Build edge set for quick lookup
    edge_set = set()
    for i in range(N):
        for j in range(i+1, N):
            if (adj_np[i] >> j) & 1:
                edge_set.add((i, j))
    
    non_edge_set = set()
    for i in range(N):
        for j in range(i+1, N):
            if (i, j) not in edge_set:
                non_edge_set.add((i, j))
    
    # Precompute adjacency lists
    def get_adj_list(adj_np, v):
        return [u for u in range(N) if u != v and (adj_np[v] >> u) & 1]
    
    # Local search with swaps
    random.seed(42)
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() - start_time < DEADLINE:
        if no_improve > max_no_improve:
            # Restart with perturbation
            adj_np = best_adj_np.copy()
            edge_set = set()
            for i in range(N):
                for j in range(i+1, N):
                    if (adj_np[i] >> j) & 1:
                        edge_set.add((i, j))
            # Perturb: random swap
            if edge_set and non_edge_set:
                e1 = random.choice(list(edge_set))
                e2 = random.choice(list(non_edge_set))
                # Remove e1, add e2
                u1, v1 = e1
                adj_np[u1] &= ~(1 << v1)
                adj_np[v1] &= ~(1 << u1)
                edge_set.remove(e1)
                non_edge_set.add(e1)
                u2, v2 = e2
                adj_np[u2] |= (1 << v2)
                adj_np[v2] |= (1 << u2)
                edge_set.add(e2)
                non_edge_set.remove(e2)
            no_improve = 0
            continue
        
        # Pick a random edge to remove and a random non-edge to add
        if not edge_set or not non_edge_set:
            break
        
        e1 = random.choice(list(edge_set))
        e2 = random.choice(list(non_edge_set))
        
        u1, v1 = e1
        u2, v2 = e2
        
        # Try the swap
        # Remove (u1,v1)
        adj_np[u1] &= ~(1 << v1)
        adj_np[v1] &= ~(1 << u1)
        # Add (u2,v2)
        adj_np[u2] |= (1 << v2)
        adj_np[v2] |= (1 << u2)
        
        new_c4s = count_c4s_np(adj_np)
        
        if new_c4s < current_c4s:
            # Accept
            edge_set.remove(e1)
            non_edge_set.add(e1)
            edge_set.add(e2)
            non_edge_set.remove(e2)
            current_c4s = new_c4s
            no_improve = 0
            if current_c4s < best_c4s:
                best_c4s = current_c4s
                best_adj_np = adj_np.copy()
        else:
            # Revert
            adj_np[u1] |= (1 << v1)
            adj_np[v1] |= (1 << u1)
            adj_np[u2] &= ~(1 << v2)
            adj_np[v2] &= ~(1 << u2)
            no_improve += 1
    
    # Extract edges from best_adj_np
    best_adj_np = best_adj_np.copy()
    result_edges = []
    for i in range(N):
        for j in range(i+1, N):
            if (best_adj_np[i] >> j) & 1:
                result_edges.append([i, j])
    
    # Verify we have exactly M edges
    if len(result_edges) != M:
        # Fallback: use a simple construction
        result_edges = []
        for i in range(N):
            for j in range(i+1, N):
                if len(result_edges) >= M:
                    break
                result_edges.append([i, j])
            if len(result_edges) >= M:
                break
        result_edges = result_edges[:M]
    
    return result_edges
