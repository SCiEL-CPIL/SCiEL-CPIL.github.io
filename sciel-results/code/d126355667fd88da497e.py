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
    
    # Compute the set of "bad" pairs: pairs (u,v) with exactly 2 common neighbors
    # Adding an edge that would make any bad pair have 3 common neighbors is forbidden
    def compute_bad_pairs(adj):
        """Return set of pairs (u,v) with exactly 2 common neighbors."""
        bad = set()
        for u in range(n):
            for v in range(u + 1, n):
                if popcount(adj[u] & adj[v]) == 2:
                    bad.add((u, v))
        return bad
    
    def can_add_fast(adj, u, v, bad_pairs):
        """Fast check using precomputed bad pairs."""
        if (adj[u] >> v) & 1:
            return True
        
        # (u,v) itself must not already have >2 common neighbors
        if popcount(adj[u] & adj[v]) > 2:
            return False
        
        # For each w in N(v) \ {u}, if (u,w) is in bad_pairs, adding (u,v) would make it bad
        # because v would become a 3rd common neighbor of u and w
        for w in range(n):
            if w == u or w == v:
                continue
            if (adj[v] >> w) & 1:
                if (min(u,w), max(u,w)) in bad_pairs:
                    return False
        
        # For each w in N(u) \ {v}, if (v,w) is in bad_pairs, adding (u,v) would make it bad
        for w in range(n):
            if w == u or w == v:
                continue
            if (adj[u] >> w) & 1:
                if (min(v,w), max(v,w)) in bad_pairs:
                    return False
        
        return True
    
    def update_bad_pairs(adj, bad_pairs, u, v, adding=True):
        """Update bad_pairs after adding or removing edge (u,v)."""
        if adding:
            # (u,v) might become bad if it now has exactly 2 common neighbors
            if popcount(adj[u] & adj[v]) == 2:
                bad_pairs.add((min(u,v), max(u,v)))
            
            # For each w in N(v) \ {u}, (u,w) gains v as common neighbor
            # If (u,w) had 1 common neighbor before, it now has 2 -> becomes bad
            for w in range(n):
                if w == u or w == v:
                    continue
                if (adj[v] >> w) & 1:
                    pair = (min(u,w), max(u,w))
                    if popcount(adj[u] & adj[w]) == 2:
                        bad_pairs.add(pair)
                    elif popcount(adj[u] & adj[w]) > 2:
                        # Should not happen if we're maintaining correctly
                        bad_pairs.discard(pair)
            
            # For each w in N(u) \ {v}, (v,w) gains u as common neighbor
            for w in range(n):
                if w == u or w == v:
                    continue
                if (adj[u] >> w) & 1:
                    pair = (min(v,w), max(v,w))
                    if popcount(adj[v] & adj[w]) == 2:
                        bad_pairs.add(pair)
                    elif popcount(adj[v] & adj[w]) > 2:
                        bad_pairs.discard(pair)
        else:
            # Removing edge (u,v)
            if (min(u,v), max(u,v)) in bad_pairs:
                bad_pairs.discard((min(u,v), max(u,v)))
            
            # For each w in N(v) \ {u}, (u,w) loses v as common neighbor
            for w in range(n):
                if w == u or w == v:
                    continue
                if (adj[v] >> w) & 1:
                    pair = (min(u,w), max(u,w))
                    if popcount(adj[u] & adj[w]) == 2:
                        bad_pairs.add(pair)
                    elif pair in bad_pairs:
                        bad_pairs.discard(pair)
            
            # For each w in N(u) \ {v}, (v,w) loses u as common neighbor
            for w in range(n):
                if w == u or w == v:
                    continue
                if (adj[u] >> w) & 1:
                    pair = (min(v,w), max(v,w))
                    if popcount(adj[v] & adj[w]) == 2:
                        bad_pairs.add(pair)
                    elif pair in bad_pairs:
                        bad_pairs.discard(pair)
    
    # Local search with bad-pair tracking
    def local_search(current_adj, current_edges, max_iters=10000):
        improved = True
        iterations = 0
        bad_pairs = compute_bad_pairs(current_adj)
        
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
                
                temp_bad = set(bad_pairs)
                update_bad_pairs(temp_adj, temp_bad, ru, rv, adding=False)
                
                best_gain = 0
                best_new_adj = None
                best_new_bad = None
                
                # Try adding each non-edge
                for (au, av) in non_edge_list:
                    if (au, av) == (ru, rv) or (au, av) == (rv, ru):
                        continue
                    
                    if can_add_fast(temp_adj, au, av, temp_bad):
                        new_adj = temp_adj[:]
                        new_adj[au] |= (1 << av)
                        new_adj[av] |= (1 << au)
                        
                        new_bad = set(temp_bad)
                        update_bad_pairs(new_adj, new_bad, au, av, adding=True)
                        
                        # Try to add more edges greedily
                        added_more = 0
                        for (xu, xv) in non_edge_list:
                            if (xu, xv) == (au, av) or (xu, xv) == (ru, rv) or (xu, xv) == (rv, ru):
                                continue
                            if can_add_fast(new_adj, xu, xv, new_bad):
                                new_adj[xu] |= (1 << xv)
                                new_adj[xv] |= (1 << xu)
                                update_bad_pairs(new_adj, new_bad, xu, xv, adding=True)
                                added_more += 1
                        
                        gain = added_more
                        if gain > best_gain:
                            best_gain = gain
                            best_new_adj = new_adj
                            best_new_bad = new_bad
                
                if best_gain > 0 and best_new_adj is not None:
                    current_adj = best_new_adj
                    current_edges += best_gain
                    bad_pairs = best_new_bad
                    improved = True
                    break
            
            # If no swap improved, try just adding edges
            if not improved:
                for (au, av) in non_edge_list:
                    if time.time() > deadline:
                        break
                    if can_add_fast(current_adj, au, av, bad_pairs):
                        current_adj[au] |= (1 << av)
                        current_adj[av] |= (1 << au)
                        update_bad_pairs(current_adj, bad_pairs, au, av, adding=True)
                        current_edges += 1
                        improved = True
                        non_edge_list = get_non_edge_list(current_adj)
        
        return current_adj, current_edges
    
    # 2-opt swap: remove two edges, add two different edges
    def two_opt_swap(current_adj, current_edges, bad_pairs, max_trials=500):
        improved = False
        edges_list = get_edge_list(current_adj)
        non_edges_list = get_non_edge_list(current_adj)
        
        for i in range(min(len(edges_list), max_trials)):
            if time.time() > deadline:
                break
            e1 = edges_list[i]
            for j in range(i+1, min(len(edges_list), i+max_trials//10)):
                if time.time() > deadline:
                    break
                e2 = edges_list[j]
                
                # Remove both edges
                temp_adj = current_adj[:]
                u1, v1 = e1
                u2, v2 = e2
                temp_adj[u1] &= ~(1 << v1)
                temp_adj[v1] &= ~(1 << u1)
                temp_adj[u2] &= ~(1 << v2)
                temp_adj[v2] &= ~(1 << u2)
                
                temp_bad = set(bad_pairs)
                update_bad_pairs(temp_adj, temp_bad, u1, v1, adding=False)
                update_bad_pairs(temp_adj, temp_bad, u2, v2, adding=False)
                
                # Try adding pairs of non-edges
                for (a1, b1) in non_edges_list:
                    if (a1, b1) in (e1, e2) or (b1, a1) in (e1, e2):
                        continue
                    if not can_add_fast(temp_adj, a1, b1, temp_bad):
                        continue
                    
                    adj1 = temp_adj[:]
                    adj1[a1] |= (1 << b1)
                    adj1[b1] |= (1 << a1)
                    bad1 = set(temp_bad)
                    update_bad_pairs(adj1, bad1, a1, b1, adding=True)
                    
                    for (a2, b2) in non_edges_list:
                        if (a2, b2) in (e1, e2) or (b2, a2) in (e1, e2):
                            continue
                        if (a2, b2) == (a1, b1) or (b2, a2) == (a1, b1):
                            continue
                        if not can_add_fast(adj1, a2, b2, bad1):
                            continue
                        
                        adj2 = adj1[:]
                        adj2[a2] |= (1 << b2)
                        adj2[b2] |= (1 << a2)
                        bad2 = set(bad1)
                        update_bad_pairs(adj2, bad2, a2, b2, adding=True)
                        
                        # Try to add more edges
                        added_more = 0
                        for (xu, xv) in non_edges_list:
                            if (xu, xv) in (e1, e2) or (xv, xu) in (e1, e2):
                                continue
                            if (xu, xv) == (a1, b1) or (xv, xu) == (a1, b1):
                                continue
                            if (xu, xv) == (a2, b2) or (xv, xu) == (a2, b2):
                                continue
                            if can_add_fast(adj2, xu, xv, bad2):
                                adj2[xu] |= (1 << xv)
                                adj2[xv] |= (1 << xu)
                                update_bad_pairs(adj2, bad2, xu, xv, adding=True)
                                added_more += 1
                        
                        # Net: removed 2, added 2 + added_more, gain = added_more
                        if added_more > 0:
                            current_adj = adj2
                            current_edges += added_more
                            bad_pairs = bad2
                            improved = True
                            return current_adj, current_edges, bad_pairs, improved
        
        return current_adj, current_edges, bad_pairs, improved
    
    # Main search: try multiple random orderings for greedy, then local search
    random.seed(42)
    
    best_adj = [0] * n
    best_edges = 0
    best_bad = set()
    
    # Try many random orderings
    num_trials = 150
    
    for trial in range(num_trials):
        if time.time() > deadline:
            break
        
        order = list(range(m))
        random.shuffle(order)
        adj, edges = greedy_construct(order)
        
        # Local search on this result
        adj, edges = local_search(adj, edges, max_iters=3000)
        
        # Try 2-opt swaps
        adj, edges, bad, improved = two_opt_swap(adj, edges, compute_bad_pairs(adj), max_trials=200)
        
        # Additional local search after 2-opt
        if improved:
            adj, edges = local_search(adj, edges, max_iters=2000)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
            best_bad = compute_bad_pairs(adj)
    
    # Additional local search on the best result
    best_adj, best_edges = local_search(best_adj, best_edges, max_iters=15000)
    
    # Try 2-opt on the best
    best_adj, best_edges, best_bad, _ = two_opt_swap(best_adj, best_edges, best_bad, max_trials=500)
    
    # Final local search
    best_adj, best_edges = local_search(best_adj, best_edges, max_iters=10000)
    
    # Final verification
    if not is_k23_free_adj(best_adj):
        # Fallback: rebuild from scratch
        order = list(range(m))
        random.shuffle(order)
        best_adj, best_edges = greedy_construct(order)
        best_adj, best_edges = local_search(best_adj, best_edges, max_iters=10000)
    
    # Final polish: try to add any remaining edges
    non_edges = get_non_edge_list(best_adj)
    bad_final = compute_bad_pairs(best_adj)
    for (au, av) in non_edges:
        if time.time() > deadline:
            break
        if can_add_fast(best_adj, au, av, bad_final):
            best_adj[au] |= (1 << av)
            best_adj[av] |= (1 << au)
            update_bad_pairs(best_adj, bad_final, au, av, adding=True)
            best_edges += 1
    
    # Convert to edge list
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    
    return edges
# EVOLVE-BLOCK-END
