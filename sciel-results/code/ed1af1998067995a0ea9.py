"""K2,3-free graph maximization on 20 vertices using bitmask local search."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n = 20
    # All possible edges as bit indices
    edge_list = []
    for u in range(n):
        for v in range(u + 1, n):
            edge_list.append((u, v))
    m = len(edge_list)  # 190 edges
    
    # Map edge to index
    edge_idx = {}
    for i, (u, v) in enumerate(edge_list):
        edge_idx[(u, v)] = i
        edge_idx[(v, u)] = i
    
    # For each edge, precompute which other edges conflict with it (would create K2,3)
    # Two edges (a,b) and (c,d) conflict if they share exactly one vertex and 
    # the other two endpoints have >= 2 common neighbors when both edges are present.
    # Actually, let's think differently.
    
    # A K2,3 subgraph exists if there are two vertices u,v with at least 3 common neighbors.
    # So the constraint is: for every pair (u,v), |N(u) ∩ N(v)| <= 2.
    
    # Let's use a different approach: represent the graph as adjacency bitmasks.
    # adj[i] is a bitmask of neighbors of vertex i.
    
    # Start with a random valid graph, then do local search.
    
    def count_common_neighbors(adj, u, v):
        return bin(adj[u] & adj[v]).count('1')
    
    def is_valid(adj):
        for u in range(n):
            for v in range(u + 1, n):
                if count_common_neighbors(adj, u, v) > 2:
                    return False
        return True
    
    def num_edges(adj):
        total = 0
        for i in range(n):
            total += bin(adj[i]).count('1')
        return total // 2
    
    # Build a good initial graph: start with random and greedily add edges
    def build_initial():
        adj = [0] * n
        # Try to add edges in random order, keep if valid
        edges_order = list(range(m))
        random.shuffle(edges_order)
        for ei in edges_order:
            u, v = edge_list[ei]
            # Check if adding edge (u,v) would violate constraint
            # New common neighbors of (u,v) = N(u) ∩ N(v) plus the edge itself doesn't add to common neighbors
            # Actually adding edge (u,v): u becomes neighbor of v and v becomes neighbor of u.
            # This changes N(u) and N(v).
            # For any pair (a,b), we need to check if common neighbors increase.
            # The pairs affected are those involving u or v.
            old_u = adj[u]
            old_v = adj[v]
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            valid = True
            # Check pairs involving u: (u, w) for w != v
            for w in range(n):
                if w == v:
                    continue
                # Check pair (u, w): common neighbors = adj[u] & adj[w]
                # adj[u] now includes v. So common neighbors of (u,w) include v if v is neighbor of w.
                cn = bin(adj[u] & adj[w]).count('1')
                if cn > 2:
                    valid = False
                    break
            if valid:
                # Check pairs involving v: (v, w) for w != u
                for w in range(n):
                    if w == u:
                        continue
                    cn = bin(adj[v] & adj[w]).count('1')
                    if cn > 2:
                        valid = False
                        break
            if valid:
                # Check pairs (a, b) where neither is u nor v: not affected
                pass
            else:
                adj[u] = old_u
                adj[v] = old_v
        
        return adj
    
    # Local search: try adding edges, or swapping (add one, remove another)
    def local_search(adj, time_limit):
        adj = adj[:]
        current_edges = num_edges(adj)
        best_adj = adj[:]
        best_edges = current_edges
        
        # Precompute for faster validation
        def try_add(adj, u, v):
            """Check if we can add edge (u,v) to adj."""
            if adj[u] & (1 << v):
                return True  # already there
            # Adding (u,v): u gains neighbor v, v gains neighbor u
            # Check all pairs involving u or v
            for w in range(n):
                if w == u or w == v:
                    continue
                # Pair (u, w): new common neighbors = (adj[u] | (1<<v)) & adj[w]
                cn_uw = bin((adj[u] | (1 << v)) & adj[w]).count('1')
                if cn_uw > 2:
                    return False
                # Pair (v, w): new common neighbors = (adj[v] | (1<<u)) & adj[w]
                cn_vw = bin((adj[v] | (1 << u)) & adj[w]).count('1')
                if cn_vw > 2:
                    return False
            return True
        
        def add_edge(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        def remove_edge(adj, u, v):
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        
        improved = True
        while improved:
            improved = False
            # Try adding each missing edge
            for ei in range(m):
                u, v = edge_list[ei]
                if adj[u] & (1 << v):
                    continue  # already present
                if try_add(adj, u, v):
                    add_edge(adj, u, v)
                    current_edges += 1
                    improved = True
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]
                    if time.time() > time_limit:
                        return best_adj, best_edges
        return best_adj, best_edges
    
    # Simulated annealing for better exploration
    def simulated_annealing(adj, time_limit):
        adj = adj[:]
        current_edges = num_edges(adj)
        best_adj = adj[:]
        best_edges = current_edges
        
        def try_add(adj, u, v):
            if adj[u] & (1 << v):
                return True
            for w in range(n):
                if w == u or w == v:
                    continue
                cn_uw = bin((adj[u] | (1 << v)) & adj[w]).count('1')
                if cn_uw > 2:
                    return False
                cn_vw = bin((adj[v] | (1 << u)) & adj[w]).count('1')
                if cn_vw > 2:
                    return False
            return True
        
        def add_edge(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        def remove_edge(adj, u, v):
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        
        def valid_after_remove(adj, u, v):
            """Check if removing edge (u,v) keeps graph valid (it always does)."""
            return True
        
        temperature = 10.0
        cooling = 0.995
        
        while time.time() < time_limit:
            # Pick a random move: add or remove an edge
            if random.random() < 0.6:
                # Try to add a random missing edge
                for _ in range(20):
                    ei = random.randint(0, m - 1)
                    u, v = edge_list[ei]
                    if not (adj[u] & (1 << v)):
                        break
                else:
                    continue
                if try_add(adj, u, v):
                    add_edge(adj, u, v)
                    current_edges += 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]
                else:
                    # Can't add, try removing a random edge to enable it
                    pass
            else:
                # Remove a random edge
                present_edges = []
                for ei in range(m):
                    u, v = edge_list[ei]
                    if adj[u] & (1 << v):
                        present_edges.append(ei)
                if present_edges:
                    ei = random.choice(present_edges)
                    u, v = edge_list[ei]
                    remove_edge(adj, u, v)
                    current_edges -= 1
            
            # Occasionally do a greedy pass
            if random.random() < 0.05:
                for ei in range(m):
                    u, v = edge_list[ei]
                    if not (adj[u] & (1 << v)):
                        if try_add(adj, u, v):
                            add_edge(adj, u, v)
                            current_edges += 1
                            if current_edges > best_edges:
                                best_edges = current_edges
                                best_adj = adj[:]
            
            temperature *= cooling
        
        return best_adj, best_edges
    
    # Run multiple restarts
    best_adj_global = None
    best_edges_global = 0
    
    # First, build a good initial solution with local search
    adj = build_initial()
    adj, edges = local_search(adj, deadline - 5.0)
    best_adj_global = adj
    best_edges_global = edges
    
    if time.time() > deadline - 5.0:
        pass
    else:
        # Run simulated annealing with multiple restarts
        for restart in range(10):
            if time.time() > deadline - 3.0:
                break
            if restart == 0:
                adj_cur = best_adj_global[:]
            else:
                adj_cur = build_initial()
                adj_cur, _ = local_search(adj_cur, deadline - 5.0)
            
            adj_cur, edges_cur = simulated_annealing(adj_cur, deadline - 3.0)
            
            if edges_cur > best_edges_global:
                best_edges_global = edges_cur
                best_adj_global = adj_cur
    
    # Convert back to edge list
    result_edges = []
    for ei in range(m):
        u, v = edge_list[ei]
        if best_adj_global[u] & (1 << v):
            result_edges.append([u, v])
    
    return result_edges
# EVOLVE-BLOCK-END
