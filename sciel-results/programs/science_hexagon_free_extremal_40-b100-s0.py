import time
import random

_START = time.time()
_DEADLINE = _START + 108

def _can_add_edge(u, v, adj_masks):
    """
    Check if adding edge (u, v) creates a C6.
    This is equivalent to checking if there is a path of length 4 
    from u to v in the graph G \ {u, v}.
    Path: u - a - b - c - d - v
    a, b, c, d must be distinct and not u or v.
    """
    # We do a DFS from u.
    # Depth 0: u
    # Depth 1: neighbors of u (excluding v)
    # Depth 2: neighbors of depth 1 (excluding u, v, and visited)
    # Depth 3: neighbors of depth 2
    # Depth 4: neighbors of depth 3. If any is v, return True.
    
    # Optimization: 
    # If u and v have no common neighbors, can we still have a path of length 4?
    # u-a-b-c-d-v. 
    # a is neighbor of u. d is neighbor of v.
    # If N(u) and N(v) are disjoint, it's possible.
    
    # Let's use a stack: (node, depth, mask)
    # mask includes u and v to prevent visiting them? 
    # Actually, we just need to ensure a,b,c,d are distinct and not u,v.
    
    stack = []
    # Start with neighbors of u
    # We can't start with v
    neighbors_u = adj_masks[u] & ~(1 << v)
    
    # If no neighbors, no path
    if not neighbors_u:
        return False
        
    # Push neighbors of u as depth 1
    # mask should contain u and v? 
    # Let's say mask contains all visited nodes.
    # Initially visited: {u, v}
    initial_mask = (1 << u) | (1 << v)
    
    while neighbors_u:
        bit = neighbors_u & -neighbors_u
        a = bit.bit_length() - 1
        neighbors_u ^= bit
        stack.append((a, 1, initial_mask | (1 << a)))
        
    while stack:
        node, depth, mask = stack.pop()
        
        if depth == 4:
            # Check if node is connected to v
            if adj_masks[node] & (1 << v):
                return True
            continue
            
        if depth == 5:
            # Should not happen if we stop at 4 and check v
            pass
            
        # Get neighbors of node
        # Exclude visited nodes
        neighbors = adj_masks[node] & ~mask
        
        # We don't need to exclude v explicitly if v is in mask? 
        # Yes, v is in initial_mask.
        
        # If depth < 4, we can go deeper
        # If depth == 3, next is depth 4.
        
        # Optimization: if depth is 3, we only care if any neighbor is v.
        # But v is in mask, so it will be excluded from `neighbors`.
        # So we must check for v separately or handle depth 4 check.
        # My logic above: at depth 4, I check if node is neighbor of v.
        # This works.
        
        # Iterate neighbors
        while neighbors:
            bit = neighbors & -neighbors
            nb = bit.bit_length() - 1
            neighbors ^= bit
            stack.append((nb, depth + 1, mask | (1 << nb)))
            
    return False

def _greedy_with_local_search(n=40, seed=0):
    rng = random.Random(seed)
    adj_masks = [0] * n
    edges = set()
    
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    rng.shuffle(all_edges)
    
    # Greedy Phase
    for u, v in all_edges:
        if not _can_add_edge(u, v, adj_masks):
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            edges.add((u, v))
            
    # Local Search
    # We maintain a list of non-edges to try
    # And a list of edges to try swapping
    
    # To speed up, we can just iterate through all edges and non-edges
    # But checking all is O(N^2).
    
    # Strategy:
    # 1. Try to add any non-edge.
    # 2. If no add, try to swap: remove edge, add non-edge.
    
    # To avoid O(N^4) in swap, we limit the search.
    
    improved = True
    # Limit iterations to prevent infinite loops or excessive time
    max_iter = 1000 
    
    while improved and time.time() < _DEADLINE - 5:
        improved = False
        
        # Phase 1: Try to add
        # Shuffle non-edges to randomize
        non_edges = [(u, v) for u, v in all_edges if (u, v) not in edges]
        rng.shuffle(non_edges)
        
        added = False
        for u, v in non_edges:
            if not _can_add_edge(u, v, adj_masks):
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                edges.add((u, v))
                added = True
                improved = True
                break # Restart add/swap loop with new state
                
        if added:
            continue
            
        # Phase 2: Try to swap
        # Remove one edge, try to add one non-edge
        edge_list = list(edges)
        rng.shuffle(edge_list)
        
        swapped = False
        for u, v in edge_list:
            # Remove u, v
            adj_masks[u] &= ~(1 << v)
            adj_masks[v] &= ~(1 << u)
            edges.remove((u, v))
            
            # Try to add a non-edge
            # We can try the same non_edges list or generate new
            # To be safe, generate new or use a subset
            # Let's try to add any edge that was previously blocked
            # We can just iterate through all non-edges again
            # This is expensive. 
            # Optimization: Only try non-edges incident to u or v? 
            # Not necessarily. The swap might open up a path elsewhere.
            # But usually, local changes help local additions.
            # Let's try all non-edges but break early.
            
            non_edges_curr = [(x, y) for x, y in all_edges if (x, y) not in edges]
            rng.shuffle(non_edges_curr)
            
            added_back = False
            for x, y in non_edges_curr:
                if not _can_add_edge(x, y, adj_masks):
                    adj_masks[x] |= (1 << y)
                    adj_masks[y] |= (1 << x)
                    edges.add((x, y))
                    added_back = True
                    swapped = True
                    improved = True
                    break
                    
            if not added_back:
                # Revert swap
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                edges.add((u, v))
            
            if swapped:
                break # Restart loop
                
    return list(edges)

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 108
    
    best_edges = []
    best_count = 0
    
    # Run multiple seeds
    # Each run takes some time. 
    # If a run is fast, we can do more.
    
    for seed in range(1000):
        if time.time() > _DEADLINE - 2:
            break
        
        try:
            edges = _greedy_with_local_search(40, seed)
            count = len(edges)
            if count > best_count:
                best_count = count
                best_edges = edges
                if count >= 110:
                    break
        except Exception:
            continue
            
    return [[u, v] for u, v in best_edges]
