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

def _greedy_with_local_search(n=40, seed=0, deadline=None):
    if deadline is None:
        deadline = _DEADLINE
    rng = random.Random(seed)
    adj_masks = [0] * n
    edges = set()
    
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    rng.shuffle(all_edges)
    
    # Greedy Phase
    for u, v in all_edges:
        if time.time() > deadline - 2:
            break
        if not _can_add_edge(u, v, adj_masks):
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            edges.add((u, v))
            
    # Local Search Phase - try to add non-edges
    non_edges = [(u, v) for u, v in all_edges if (u, v) not in edges]
    rng.shuffle(non_edges)
    
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        for u, v in non_edges:
            if (u, v) in edges:
                continue
            if not _can_add_edge(u, v, adj_masks):
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                edges.add((u, v))
                improved = True
                break # restart loop with new state
    
    # Local Search Phase - try edge swaps (remove one edge, add up to two non-edges)
    if time.time() < deadline - 3:
        for _ in range(50):
            if time.time() > deadline - 3:
                break
            # Pick a random edge to remove
            edge_list = list(edges)
            if not edge_list:
                break
            ru, rv = rng.choice(edge_list)
            adj_masks[ru] &= ~(1 << rv)
            adj_masks[rv] &= ~(1 << ru)
            edges.discard((ru, rv))
            
            # Try to add up to 2 non-edges
            added = 0
            for u, v in non_edges:
                if (u, v) in edges:
                    continue
                if not _can_add_edge(u, v, adj_masks):
                    adj_masks[u] |= (1 << v)
                    adj_masks[v] |= (1 << u)
                    edges.add((u, v))
                    added += 1
                    if added >= 2:
                        break
            # If we couldn't add 2, try more
            if added < 2:
                for u, v in non_edges:
                    if (u, v) in edges:
                        continue
                    if not _can_add_edge(u, v, adj_masks):
                        adj_masks[u] |= (1 << v)
                        adj_masks[v] |= (1 << u)
                        edges.add((u, v))
                        added += 1
                        if added >= 2:
                            break
    
    return list(edges)

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 108
    
    best_edges = []
    best_count = 0
    
    # Use multiple seeds with different strategies
    for seed in range(10000):
        if time.time() > _DEADLINE - 2:
            break
        
        try:
            edges = _greedy_with_local_search(40, seed)
            count = len(edges)
            if count > best_count:
                best_count = count
                best_edges = edges
                if count >= 105:
                    break
        except Exception:
            continue
            
    return [[u, v] for u, v in best_edges]
