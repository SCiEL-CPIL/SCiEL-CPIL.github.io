"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    n = 20
    
    # Enumerate all possible edges
    all_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            all_edges.append((u, v))
    
    m = len(all_edges)  # 190 possible edges
    
    # Precompute for each edge (u,v) the list of other vertices
    # and for each vertex pair, their common neighbors
    
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
        
        # For each w in N(v) (w != u), check pair (u,w): 
        # adding v to N(u) means v becomes a common neighbor of u and w (since w in N(v) and v in N(u))
        # So common(u,w) increases by 1. Need old common(u,w) <= 1.
        for w in range(n):
            if w == u or w == v:
                continue
            if (adj[v] >> w) & 1:  # w in N(v)
                if popcount(adj[u] & adj[w]) > 1:
                    return False
        
        # For each w in N(u) (w != v), check pair (v,w):
        # adding u to N(v) means u becomes a common neighbor of v and w (since w in N(u) and u in N(v))
        # So common(v,w) increases by 1. Need old common(v,w) <= 1.
        for w in range(n):
            if w == u or w == v:
                continue
            if (adj[u] >> w) & 1:  # w in N(u)
                if popcount(adj[v] & adj[w]) > 1:
                    return False
        
        return True
    
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
    
    # Greedy construction with a given order
    def greedy_construct(order, adj=None):
        if adj is None:
            adj = [0] * n
        edges = 0
        for idx in order:
            u, v = all_edges[idx]
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges += 1
        return adj, edges
    
    # Local search: try to improve by removing one edge and adding another
    def local_search(current_adj, current_edges, max_iters=10000):
        improved = True
        iterations = 0
        
        while improved and iterations < max_iters and time.time() < deadline:
            improved = False
            iterations += 1
            
            current_edges_list = get_edge_list(current_adj)
            non_edge_list = get_non_edge_list(current_adj)
            
            # Try removing each edge and see if we can add a better one
            for (ru, rv) in current_edges_list:
                if time.time() > deadline:
                    break
                
                # Remove edge (ru, rv)
                temp_adj = current_adj[:]
                temp_adj[ru] &= ~(1 << rv)
                temp_adj[rv] &= ~(1 << ru)
                
                best_gain = 0
                best_add = None
                
                # Try adding each non-edge
                for (au, av) in non_edge_list:
                    if (au, av) == (ru, rv) or (au, av) == (rv, ru):
                        continue
                    
                    if can_add_edge(temp_adj, au, av):
                        # Check if this leads to improvement
                        # After adding (au,av), try to greedily add more
                        new_adj = temp_adj[:]
                        new_adj[au] |= (1 << av)
                        new_adj[av] |= (1 << au)
                        
                        # Try to add more edges greedily
                        added_more = 0
                        for (xu, xv) in non_edge_list:
                            if (xu, xv) == (au, av) or (xu, xv) == (ru, rv) or (xu, xv) == (rv, ru):
                                continue
                            if can_add_edge(new_adj, xu, xv):
                                new_adj[xu] |= (1 << xv)
                                new_adj[xv] |= (1 << xu)
                                added_more += 1
                        
                        gain = added_more  # We removed 1, added 1 + added_more, net gain = added_more
                        if gain > best_gain:
                            best_gain = gain
                            best_add = new_adj
                
                if best_gain > 0 and best_add is not None:
                    current_adj = best_add
                    current_edges += best_gain
                    improved = True
                    break
            
            # If no swap improved, try just adding edges
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
        
        return current_adj, current_edges
    
    # Main search: try multiple random orderings for greedy, then local search
    random.seed(42)
    
    best_adj = [0] * n
    best_edges = 0
    
    # Try many random orderings
    num_trials = 200
    
    for trial in range(num_trials):
        if time.time() > deadline:
            break
        
        order = list(range(m))
        random.shuffle(order)
        adj, edges = greedy_construct(order)
        
        # Local search on this result
        adj, edges = local_search(adj, edges, max_iters=5000)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
    
    # Additional local search on the best result
    best_adj, best_edges = local_search(best_adj, best_edges, max_iters=20000)
    
    # Final verification
    if not is_k23_free_adj(best_adj):
        # Fallback: rebuild from scratch
        order = list(range(m))
        random.shuffle(order)
        best_adj, best_edges = greedy_construct(order)
        best_adj, best_edges = local_search(best_adj, best_edges, max_iters=10000)
    
    # Final polish: try to add any remaining edges
    non_edges = get_non_edge_list(best_adj)
    for (au, av) in non_edges:
        if time.time() > deadline:
            break
        if can_add_edge(best_adj, au, av):
            best_adj[au] |= (1 << av)
            best_adj[av] |= (1 << au)
            best_edges += 1
    
    # Convert to edge list
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    
    return edges
# EVOLVE-BLOCK-END
