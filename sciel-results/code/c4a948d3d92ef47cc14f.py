import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit
    
    n = 20
    best_edges = []
    best_score = 0
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Try multiple random seeds with greedy construction
    for seed in range(1000):
        if time.time() > deadline - 10.0:
            break
        
        edges = _greedy_construct(n, all_edges, seed)
        score = len(edges)
        
        if score > best_score:
            best_score = score
            best_edges = edges
    
    # If we still have time, try local search improvements
    if time.time() < deadline - 10.0 and best_edges:
        improved = _local_search(n, best_edges, all_edges, deadline - 5.0)
        if len(improved) > best_score:
            best_edges = improved
            best_score = len(improved)
    
    return best_edges

def _greedy_construct(n, all_edges, seed):
    """
    Greedy construction of K2,3-free graph using bitmasks for speed.
    """
    random.seed(seed)
    
    # adj[i] is a bitmask of neighbors of i
    adj = [0] * n
    edges = []
    
    shuffled_edges = all_edges[:]
    random.shuffle(shuffled_edges)
    
    for u, v in shuffled_edges:
        if _can_add_edge_fast(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.append([u, v])
    
    return edges

def _can_add_edge_fast(adj, u, v):
    """
    Check if adding edge (u,v) is valid using bitmasks.
    adj is a list of ints.
    """
    # Check pair (u, v)
    # Common neighbors of u and v are adj[u] & adj[v]
    # Note: u is not in adj[u], v is not in adj[v]
    # So the intersection is correct.
    if (adj[u] & adj[v]).bit_count() >= 3:
        return False
    
    # Check pairs (u, w) and (v, w) for w != u, v
    # We can iterate through all w, but we can also just check the bits.
    # However, we need to check if the count exceeds 2.
    # For each w, new_count_uw = count(adj[u] & adj[w]) + (1 if v in adj[w] else 0)
    # This is O(n) per check. n=20, so it's fast.
    
    mask_u = adj[u]
    mask_v = adj[v]
    
    # Iterate over all w
    for w in range(len(adj)):
        if w == u or w == v:
            continue
        
        # Check (u, w)
        # Common neighbors of u and w
        common_uw = mask_u & adj[w]
        count_uw = common_uw.bit_count()
        # If v is a neighbor of w, it becomes a common neighbor of u and w
        if (adj[w] & (1 << v)):
            count_uw += 1
        if count_uw >= 3:
            return False
            
        # Check (v, w)
        common_vw = mask_v & adj[w]
        count_vw = common_vw.bit_count()
        # If u is a neighbor of w, it becomes a common neighbor of v and w
        if (adj[w] & (1 << u)):
            count_vw += 1
        if count_vw >= 3:
            return False
            
    return True

def _local_search(n, edges, all_edges, deadline):
    """
    Local search with 1-2 moves (remove 1, add up to 2).
    """
    adj = [0] * n
    edge_set = set()
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edge_set.add((min(u,v), max(u,v)))
        
    current_edges = [list(e) for e in edges]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Try to add any single edge
        for u, v in all_edges:
            if time.time() > deadline:
                return current_edges
            if (u, v) in edge_set:
                continue
            if _can_add_edge_fast(adj, u, v):
                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_set.add((u, v))
                current_edges.append([u, v])
                improved = True
        
        if improved:
            continue # Restart adding edges as the graph changed
        
        # Phase 2: If no single edge can be added, try 1-2 moves
        # Remove one edge, try to add two
        for e in list(edge_set):
            if time.time() > deadline:
                return current_edges
            
            u1, v1 = e
            # Remove e
            adj[u1] ^= (1 << v1)
            adj[v1] ^= (1 << u1)
            edge_set.remove(e)
            
            # Try to add two edges
            added = []
            for u2, v2 in all_edges:
                if (u2, v2) in edge_set:
                    continue
                if _can_add_edge_fast(adj, u2, v2):
                    # Add first edge
                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)
                    edge_set.add((u2, v2))
                    added.append((u2, v2))
                    break # Try to add a second one now
            
            if len(added) == 1:
                u2, v2 = added[0]
                for u3, v3 in all_edges:
                    if (u3, v3) in edge_set:
                        continue
                    if _can_add_edge_fast(adj, u3, v3):
                        # Add second edge
                        adj[u3] |= (1 << v3)
                        adj[v3] |= (1 << u3)
                        edge_set.add((u3, v3))
                        added.append((u3, v3))
                        break
            
            if len(added) == 2:
                # Successful 1-2 move
                # Rebuild current_edges from edge_set to keep it clean
                current_edges = [[u, v] for u, v in edge_set]
                improved = True
                break # Restart loop as graph changed
            else:
                # Revert
                if len(added) == 1:
                    u2, v2 = added[0]
                    adj[u2] ^= (1 << v2)
                    adj[v2] ^= (1 << u2)
                    edge_set.remove((u2, v2))
                # Restore original edge
                adj[u1] |= (1 << v1)
                adj[v1] |= (1 << u1)
                edge_set.add(e)
                
    return current_edges
