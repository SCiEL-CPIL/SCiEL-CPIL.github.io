import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    n = 40
    
    # adj[v] is a bitmask of neighbors of v
    adj = [0] * n
    edges = []
    
    def has_path_len_5(u, v):
        # Check if there is a path of length 5 from u to v
        # reach[k] is bitmask of vertices reachable from u in exactly k steps
        reach = [0] * 6
        reach[0] = 1 << u
        
        for k in range(1, 6):
            mask = reach[k-1]
            union = 0
            while mask:
                l = mask & -mask
                idx = l.bit_length() - 1
                union |= adj[idx]
                mask ^= l
            reach[k] = union
        
        # Check if v is in reach[5]
        return (reach[5] >> v) & 1 == 1

    # Greedy Construction
    # We try multiple random orders to find a good packing
    best_edges = 0
    best_adj = None
    best_edge_list = None
    
    # Pre-generate a list of all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    for trial in range(50):
        if time.time() > deadline - 10.0:
            break
            
        # Reset graph
        adj = [0] * n
        edges = []
        
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if time.time() > deadline - 10.0:
                break
            # Check if adding (u, v) creates a C6
            # This requires checking path of length 5 in G \ {(u,v)}
            # Since (u,v) is not in adj yet, we just check path in current adj.
            if not has_path_len_5(u, v):
                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append((u, v))
                
        if len(edges) > best_edges:
            best_edges = len(edges)
            best_adj = adj[:]
            best_edge_list = edges[:]
            
    if best_adj is None:
        return []
        
    # Local Search / Swap Optimization
    adj = best_adj
    edges = best_edge_list
    edge_set = set(edges)
    
    def try_add(u, v):
        if (u, v) in edge_set or (v, u) in edge_set:
            return False
        if not has_path_len_5(u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edge_set.add((u, v))
            edges.append((u, v))
            return True
        return False

    def try_remove(u, v):
        if (u, v) not in edge_set:
            return False
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        edge_set.discard((u, v))
        edges.remove((u, v))
        return True

    # Try to add any missing edge
    added_any = True
    while added_any and time.time() < deadline - 5.0:
        added_any = False
        for u in range(n):
            for v in range(u+1, n):
                if (u, v) not in edge_set:
                    if try_add(u, v):
                        added_any = True
                        break
            if added_any:
                break

    # Swap search: Remove one edge, try to add two? Or just one?
    # If we remove one and add one, we maintain size.
    # If we remove one and add two, we increase size.
    # Let's try: Remove random edge, then try to add any missing edges.
    # If we can add more than 1, we net gain.
    
    # Actually, the greedy phase usually leaves no addable edges.
    # So we must remove some to unblock others.
    
    swaps = 0
    while time.time() < deadline - 2.0 and swaps < 1000:
        # Pick a random edge to remove
        if not edges:
            break
        idx = random.randint(0, len(edges)-1)
        u, v = edges[idx]
        try_remove(u, v)
        
        added_count = 0
        # Try to add edges
        # We can just run the "try to add any missing edge" logic
        # But that might add the same edge back immediately.
        # We should avoid adding (u,v) back if we just removed it?
        # Or maybe we should?
        
        # Let's try to add any edge, including (u,v)
        # If we add (u,v) back, we are in a loop.
        # So let's mark (u,v) as forbidden for this step?
        # Or just try to add others.
        
        # Simple strategy: Try to add any edge. If we add > 1, keep.
        # If we add 1 (and it's u,v), revert.
        # If we add 0, revert.
        
        # To avoid adding (u,v) back immediately, we can just check
        # if we can add any *other* edge.
        
        improved = False
        for i in range(n):
            for j in range(i+1, n):
                if (i, j) == (u, v) or (i, j) == (v, u):
                    continue
                if (i, j) not in edge_set:
                    if try_add(i, j):
                        improved = True
                        break
            if improved:
                break
        
        if not improved:
            # Revert removal
            try_add(u, v)
            swaps += 1
        else:
            # We removed 1, added 1. Net 0.
            # But maybe we can add more now?
            # Let's try to add more
            added_more = True
            while added_more:
                added_more = False
                for i in range(n):
                    for j in range(i+1, n):
                        if (i, j) not in edge_set:
                            if try_add(i, j):
                                added_more = True
                                break
                    if added_more:
                        break
            swaps += 1

    # Convert back to list of lists
    return [[u, v] for u, v in edges]

# EVOLVE-BLOCK-END
