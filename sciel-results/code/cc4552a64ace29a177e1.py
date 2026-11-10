"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n = 20
    
    # Enumerate all possible edges
    all_edges = []
    edge_idx = {}
    for u in range(n):
        for v in range(u + 1, n):
            idx = len(all_edges)
            all_edges.append((u, v))
            edge_idx[(u, v)] = idx
    
    m = len(all_edges)  # 190 possible edges
    
    # Build conflict graph: edge i conflicts with edge j if having both violates K2,3
    # Two edges (u,v) and (a,b) conflict if:
    # 1. They share a vertex and together create a K2,3
    # 2. More generally, adding both edges creates a pair of vertices with 3+ common neighbors
    
    # For each pair of edges, check if they can coexist in a K2,3-free graph
    # This is complex, so we build the conflict graph incrementally
    
    # Conflict[i] is a bitmask of edges that conflict with edge i
    conflict = [0] * m
    
    # Two edges conflict if there exists a pair of vertices (x,y) such that
    # both edges are "responsible" for x and y having 3 common neighbors
    # 
    # A simpler approach: two edges e1=(u,v) and e2=(a,b) conflict if:
    # - They share exactly one vertex, say u=a, and v!=b
    #   Then {u,v} and {u,b} are two pairs. If there are 2 other common neighbors of v and b
    #   (besides u), then adding both edges might create K2,3. But this depends on other edges.
    #
    # Actually, the conflict depends on the full set of edges. So we can't precompute
    # a static conflict graph easily.
    #
    # Alternative approach: Use branch-and-bound with dynamic feasibility checking.
    # At each step, we maintain the current set of selected edges and check if adding
    # a new edge keeps the graph K2,3-free.
    
    # Let's use a different formulation:
    # State: current set of edges (as a bitmask)
    # We want to maximize the number of edges while maintaining K2,3-free property.
    
    # Precompute for each edge (u,v), the set of vertices that would be "affected"
    # when we add this edge.
    
    # For branch-and-bound, we need:
    # 1. A way to quickly check if a set of edges is K2,3-free
    # 2. A good upper bound on how many more edges we can add
    
    # Let's represent the graph as adjacency bitmasks
    # adj[i] is a bitmask of neighbors of vertex i
    
    def popcount(x):
        return bin(x).count('1')
    
    def is_k23_free_adj(adj):
        """Check if graph with adjacency list adj is K2,3-free."""
        for u in range(n):
            for v in range(u + 1, n):
                if popcount(adj[u] & adj[v]) > 2:
                    return False
        return True
    
    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) keeps the graph K2,3-free."""
        if (adj[u] >> v) & 1:
            return True  # Already present
        
        # Check pair (u,v): common neighbors must be <= 2
        if popcount(adj[u] & adj[v]) > 2:
            return False
        
        # For each w in N(v), check pair (u,w): adding v as common neighbor
        # new common(u,w) = old common(u,w) + 1 (since v is now neighbor of u, and v is neighbor of w iff w in N(v))
        # We need new common <= 2, so old common <= 1
        for w in range(n):
            if w == u or w == v:
                continue
            if (adj[v] >> w) & 1:  # w in N(v)
                if popcount(adj[u] & adj[w]) > 1:
                    return False
        
        # For each w in N(u), check pair (v,w): adding u as common neighbor
        for w in range(n):
            if w == u or w == v:
                continue
            if (adj[u] >> w) & 1:  # w in N(u)
                if popcount(adj[v] & adj[w]) > 1:
                    return False
        
        return True
    
    # Upper bound: maximum possible edges in a K2,3-free graph on n vertices
    # The Zarankiewicz problem gives bounds, but for n=20, we can use a simple bound.
    # Each pair of vertices can have at most 2 common neighbors.
    # Sum over all pairs of common neighbors <= C(n,2) * 2
    # This sum equals sum over vertices of C(deg(v), 2)
    # So sum_v C(deg(v), 2) <= 2 * C(n,2) = n*(n-1)
    # To maximize edges, we want degrees as equal as possible.
    # If all degrees are d, then n * C(d,2) <= n*(n-1), so C(d,2) <= n-1 = 19
    # d(d-1)/2 <= 19, d^2 - d - 38 <= 0, d <= (1 + sqrt(153))/2 ≈ 6.7
    # So max degree is about 6-7.
    # Total edges <= n * 7 / 2 = 70
    # But this is a loose bound. Let's use a tighter one.
    
    # Actually, for the branch-and-bound, we can use:
    # Upper bound = current_edges + remaining_possible_edges
    # where remaining_possible_edges is bounded by how many more edges can fit.
    
    # Let's implement a recursive branch-and-bound with memoization-like pruning.
    
    # Order edges by some heuristic (e.g., how "constraining" they are)
    # Edges that conflict with many others should be considered later
    
    # For now, let's use a simple ordering: by (u, v)
    
    # Branch and bound:
    # We process edges in order. At each step, we decide to include or exclude the edge.
    # We maintain the current adjacency list and current edge count.
    
    # To make this efficient, we need good pruning:
    # 1. If current_edges + (m - idx) <= best_found, prune (can't beat best)
    # 2. If adding the current edge is infeasible, we must exclude it
    
    # Let's also use a greedy upper bound: from the current state, greedily add edges
    # that don't conflict, and use that as an upper bound.
    
    best_edges = [0]  # Use list to allow modification in nested function
    best_adj = [0] * n
    
    # Try multiple random orderings to escape local optima
    random.seed(42)
    
    def branch_and_bound(edge_order, start_idx=0, adj=None, current_edges=0):
        if adj is None:
            adj = [0] * n
        
        if time.time() > deadline:
            return
        
        if start_idx >= m:
            if current_edges > best_edges[0]:
                best_edges[0] = current_edges
                best_adj[:] = adj[:]
            return
        
        # Upper bound: current + remaining
        if current_edges + (m - start_idx) <= best_edges[0]:
            return
        
        u, v = all_edges[edge_order[start_idx]]
        
        # Try including the edge
        if can_add_edge(adj, u, v):
            new_adj = adj[:]
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            branch_and_bound(edge_order, start_idx + 1, new_adj, current_edges + 1)
        
        # Try excluding the edge
        branch_and_bound(edge_order, start_idx + 1, adj, current_edges)
    
    # The naive B&B is too slow for 190 edges. We need much better pruning.
    # Let's use a different approach: local search with strong initialization.
    
    # Step 1: Greedy construction
    def greedy_construct(order):
        adj = [0] * n
        edges = 0
        for idx in order:
            u, v = all_edges[idx]
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges += 1
        return adj, edges
    
    # Try multiple random orderings
    best_greedy_adj = [0] * n
    best_greedy_edges = 0
    
    for trial in range(50):
        if time.time() > deadline:
            break
        order = list(range(m))
        random.shuffle(order)
        adj, edges = greedy_construct(order)
        if edges > best_greedy_edges:
            best_greedy_edges = edges
            best_greedy_adj = adj[:]
    
    # Step 2: Local search (2-opt style)
    # Try swapping edges: remove one, add another
    current_adj = best_greedy_adj[:]
    current_edges = best_greedy_edges
    
    # Build list of current edges and non-edges
    def get_edge_list(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append((u, v))
        return edges
    
    def get_non_edge_list(adj):
        non_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if not ((adj[u] >> v) & 1):
                    non_edges.append((u, v))
        return non_edges
    
    improved = True
    iterations = 0
    max_iterations = 50000
    
    while improved and iterations < max_iterations and time.time() < deadline:
        improved = False
        iterations += 1
        
        current_edges_list = get_edge_list(current_adj)
        non_edge_list = get_non_edge_list(current_adj)
        
        # Try removing an edge and adding a different one
        for rem_idx, (ru, rv) in enumerate(current_edges_list):
            if time.time() > deadline:
                break
            
            # Remove edge (ru, rv)
            temp_adj = current_adj[:]
            temp_adj[ru] &= ~(1 << rv)
            temp_adj[rv] &= ~(1 << ru)
            
            # Try adding each non-edge
            for (au, av) in non_edge_list:
                if (au, av) == (ru, rv) or (au, av) == (rv, ru):
                    continue
                
                if can_add_edge(temp_adj, au, av):
                    new_adj = temp_adj[:]
                    new_adj[au] |= (1 << av)
                    new_adj[av] |= (1 << au)
                    
                    # Check if this is an improvement (should be same edge count, but maybe enables more)
                    # For now, accept if same count (to explore the space)
                    # Later, we can do a second pass to add more edges
                    if not is_k23_free_adj(new_adj):
                        continue
                    
                    # Try to add more edges greedily
                    final_adj = new_adj[:]
                    final_edges = current_edges  # Removed one, added one
                    for (xu, xv) in non_edge_list:
                        if (xu, xv) == (au, av) or (xu, xv) == (ru, rv) or (xu, xv) == (rv, ru):
                            continue
                        if can_add_edge(final_adj, xu, xv):
                            final_adj[xu] |= (1 << xv)
                            final_adj[xv] |= (1 << xu)
                            final_edges += 1
                    
                    if final_edges > current_edges:
                        current_adj = final_adj
                        current_edges = final_edges
                        improved = True
                        break
            
            if improved:
                break
        
        # Also try just adding edges greedily
        if not improved:
            for (au, av) in non_edge_list:
                if time.time() > deadline:
                    break
                if can_add_edge(current_adj, au, av):
                    current_adj[au] |= (1 << av)
                    current_adj[av] |= (1 << au)
                    current_edges += 1
                    improved = True
                    non_edge_list = get_non_edge_list(current_adj)
    
    # Final verification
    if not is_k23_free_adj(current_adj):
        # Fallback to greedy result
        current_adj = best_greedy_adj[:]
        current_edges = best_greedy_edges
    
    # Convert to edge list
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (current_adj[u] >> v) & 1:
                edges.append([u, v])
    
    return edges
# EVOLVE-BLOCK-END
