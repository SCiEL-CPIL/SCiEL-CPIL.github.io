import random
import time
import sys

def is_3_colorable_fast(adj_masks, n=16):
    """
    Fast 3-colorability check using backtracking with bitmasks.
    adj_masks: list of integers where bit u is 1 if v is connected to u.
    """
    colors = [-1] * n
    
    def backtrack(v):
        if v == n:
            return True
        
        # Determine forbidden colors for vertex v
        forbidden = 0
        mask = adj_masks[v]
        # Iterate over set bits in mask
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            u = lsb.bit_length() - 1
            if colors[u] != -1:
                forbidden |= (1 << colors[u])
            m -= lsb
        
        # Try colors 0, 1, 2
        for c in range(3):
            if not (forbidden & (1 << c)):
                colors[v] = c
                if backtrack(v + 1):
                    return True
                colors[v] = -1
        return False

    return backtrack(0)

def construct():
    n = 16
    # We use a bitmask representation for edges for speed
    # edge_mask[v] is a bitmask of neighbors
    
    # 1. Initialization: Start with a known non-3-colorable triangle-free graph
    # We will construct the Grötzsch graph (11 vertices) and extend it,
    # or use a random initialization that is forced to be non-3-colorable.
    
    # Strategy: Start with a random graph, remove edges to make it triangle-free.
    # If it's 3-colorable, restart or modify.
    # However, to ensure we start valid, let's use the seed from the previous version
    # but convert it to bitmask, then optimize.
    
    # Seed edges from previous version (verified non-3-colorable)
    seed_edges = []
    for i in range(5):
        seed_edges.append((i, (i + 1) % 5))
    for i in range(5):
        u = i
        v = (i + 1) % 5
        seed_edges.append((5 + u, v))
        seed_edges.append((5 + v, u))
    for i in range(5):
        seed_edges.append((10, 5 + i))
    independent_set = [0, 2, 5, 7]
    for new_v in range(11, 16):
        for u in independent_set:
            seed_edges.append((new_v, u))
            
    # Build initial bitmask
    adj_masks = [0] * n
    for u, v in seed_edges:
        if u > v: u, v = v, u
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        
    # Verify seed is non-3-colorable
    if is_3_colorable_fast(adj_masks, n):
        # Fallback: If seed is somehow 3-colorable, try a different random init
        # But for now, assume seed is good.
        pass

    best_adj = adj_masks[:]
    best_count = sum(bin(x).count("1") for x in best_adj) // 2
    
    # Local Search
    start_time = time.time()
    time_limit = 110
    
    T = 1.0
    cooling = 0.9995
    min_T = 0.01
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    while time.time() - start_time < time_limit:
        if T < min_T:
            break
            
        # Calculate current count
        current_count = sum(bin(x).count("1") for x in adj_masks) // 2
        
        # Generate a move
        # 1. Try to add an edge (most common)
        # 2. Try to swap an edge (remove one, add another)
        
        move_type = random.random()
        
        new_adj = adj_masks[:]
        valid_move = False
        
        if move_type < 0.8:
            # Add edge
            # Pick random non-edge
            while True:
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v: continue
                if u > v: u, v = v, u
                if not (adj_masks[u] & (1 << v)):
                    break
            
            # Check if adding (u,v) creates a triangle
            # Triangle if common_neighbor(u, v) != 0
            common = adj_masks[u] & adj_masks[v]
            if common == 0:
                # Safe to add
                new_adj[u] |= (1 << v)
                new_adj[v] |= (1 << u)
                valid_move = True
                # Check 3-colorability
                if not is_3_colorable_fast(new_adj, n):
                    adj_masks = new_adj
                    # Update best
                    new_count = current_count + 1
                    if new_count > best_count:
                        best_count = new_count
                        best_adj = new_adj[:]
                    continue
                else:
                    # It became 3-colorable, reject
                    pass
            else:
                # Creates triangle, reject
                pass
                
        else:
            # Swap: Remove random edge, Add random edge
            if current_count == 0: continue
            
            # Remove random edge
            u = random.randint(0, n-1)
            # Pick a neighbor
            mask = adj_masks[u]
            if mask == 0: continue
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            
            # Remove (u,v)
            new_adj[u] &= ~(1 << v)
            new_adj[v] &= ~(1 << u)
            
            # Add random edge (w, x)
            while True:
                w = random.randint(0, n-1)
                x = random.randint(0, n-1)
                if w == x: continue
                if w > x: w, x = x, w
                if not (new_adj[w] & (1 << x)):
                    break
            
            # Check if adding (w,x) creates triangle in new_adj
            common = new_adj[w] & new_adj[x]
            if common == 0:
                new_adj[w] |= (1 << x)
                new_adj[x] |= (1 << w)
                valid_move = True
                # Check 3-colorability
                if not is_3_colorable_fast(new_adj, n):
                    adj_masks = new_adj
                    # Count didn't change, but structure did.
                    # We might have escaped a local minimum for future additions.
                    # We don't update best_count here as it's same.
                    continue
                else:
                    # Reject
                    pass
            else:
                # Reject
                pass

        # If valid_move was not accepted, we just loop.
        # To ensure progress, we might need to accept worse states (SA).
        # But for simplicity and speed, we stick to hill-climbing on valid states.
        # If stuck, we lower T? T is not used in this simplified version.
        # Let's add a small chance to accept a "worse" (3-colorable) state? No, invalid.
        
        # If we are stuck (no valid moves found in last few iterations), we might need to restart or use a different strategy.
        # For now, the random selection will eventually find a move if one exists.
        
        T *= cooling
        
    # Convert best_adj back to list of edges
    best_edges = []
    for u in range(n):
        mask = best_adj[u]
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if u < v:
                best_edges.append([u, v])
            m -= lsb
            
    return best_edges
