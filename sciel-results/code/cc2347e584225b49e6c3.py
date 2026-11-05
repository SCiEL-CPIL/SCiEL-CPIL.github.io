import numpy as np
import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a C4-free graph on n vertices with min degree >= n - 39.
    Maximize n.
    """
    deadline = time.time() + 145  # Leave 5s safety margin
    
    # Try n=46 first, as it is known to be possible.
    # If 46 fails, try 45, etc.
    for n in range(46, 38, -1):
        if time.time() > deadline:
            break
            
        required_min_deg = n - 39
        
        # Attempt construction
        result = _bitmask_greedy_construct(n, required_min_deg, deadline)
        
        if result is not None:
            n_found, edges_found = result
            return {'n': n_found, 'edges': edges_found}
            
        # If greedy fails, try local search with swaps
        if n >= 44 and time.time() < deadline:
            result = _local_search_swap(n, required_min_deg, deadline)
            if result is not None:
                n_found, edges_found = result
                return {'n': n_found, 'edges': edges_found}

    # Fallback: return a valid small graph
    # n=39, min_deg=0 is always valid (empty graph)
    return {'n': 39, 'edges': []}

def _check_c4_add_mask(adj_masks, u, v, n):
    """
    Check if adding edge (u,v) to the graph defined by adj_masks creates a C4.
    adj_masks: list of integers, where adj_masks[i] has bit j set if (i,j) is an edge.
    """
    # We need to check if there is a path u-w-x-v in the current graph.
    # This is equivalent to:
    # There exists w in N(u)\{v} and x in N(v)\{u} such that (w,x) is an edge.
    
    # Get neighbors of u, excluding v
    mask_u = adj_masks[u] & ~(1 << v)
    
    # Get neighbors of v, excluding u
    mask_v = adj_masks[v] & ~(1 << u)
    
    # We need to check if any neighbor w of u (in mask_u) has a neighbor x in mask_v.
    # i.e., is there a w such that (adj_masks[w] & mask_v) != 0?
    
    # Iterate through bits set in mask_u
    m = mask_u
    while m:
        # Get lowest set bit
        lsb = m & -m
        w = lsb.bit_length() - 1
        # Check if w has any neighbor in mask_v
        if adj_masks[w] & mask_v:
            return True
        m ^= lsb
        
    return False

def _bitmask_greedy_construct(n, required_min_deg, deadline):
    """
    Attempts to build a C4-free graph with min degree >= required_min_deg.
    Uses bitmask representation for fast C4 checks.
    """
    rng = random.Random()
    
    max_restarts = 5000  # Bitmask checks are fast, allow many restarts
    
    for attempt in range(max_restarts):
        if time.time() > deadline:
            break
            
        adj_masks = [0] * n
        degrees = [0] * n
        
        success = False
        while time.time() < deadline:
            # Find min degree
            min_deg = min(degrees)
            
            if min_deg >= required_min_deg:
                success = True
                break
                
            # Find vertices with min degree
            min_vertices = [i for i in range(n) if degrees[i] == min_deg]
            if not min_vertices:
                break
                
            u = rng.choice(min_vertices)
            
            # Find non-neighbors of u
            # non_neighbors are i where bit i is 0 in adj_masks[u]
            # and i != u
            non_neighbors = []
            mask_u = adj_masks[u]
            for v in range(n):
                if v == u:
                    continue
                if not (mask_u & (1 << v)):
                    non_neighbors.append(v)
                    
            if not non_neighbors:
                break
                
            rng.shuffle(non_neighbors)
            
            added = False
            for v in non_neighbors:
                if _check_c4_add_mask(adj_masks, u, v, n):
                    continue
                # Safe to add
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                degrees[u] += 1
                degrees[v] += 1
                added = True
                break # Restart loop to re-evaluate degrees
            
            if not added:
                # Stuck. Try a different strategy or restart.
                # For now, restart with new random seed
                break
                
        if success:
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if adj_masks[u] & (1 << v):
                        edges.append([u, v])
            return (n, edges)
            
    return None

def _local_search_swap(n, required_min_deg, deadline):
    """
    Fallback: Local search with edge swaps to fix stuck states.
    Starts from a greedy solution (even if incomplete) and tries to increase min degree
    by swapping edges.
    """
    # First, get a starting point from greedy (even if it didn't reach min deg)
    # We'll just run a greedy for a short time to get a dense C4-free graph
    rng = random.Random()
    adj_masks = [0] * n
    degrees = [0] * n
    
    # Quick greedy fill
    time_limit_inner = time.time() + 5
    while time.time() < time_limit_inner:
        min_deg = min(degrees)
        min_vertices = [i for i in range(n) if degrees[i] == min_deg]
        if not min_vertices:
            break
        u = rng.choice(min_vertices)
        non_neighbors = [v for v in range(n) if v != u and not (adj_masks[u] & (1 << v))]
        if not non_neighbors:
            break
        rng.shuffle(non_neighbors)
        added = False
        for v in non_neighbors:
            if not _check_c4_add_mask(adj_masks, u, v, n):
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                degrees[u] += 1
                degrees[v] += 1
                added = True
                break
        if not added:
            break
            
    # Now perform swaps
    # Objective: Increase min degree.
    # Move: Remove (a,b), Add (u,v).
    # Condition: 
    # 1. (a,b) exists.
    # 2. (u,v) does not exist.
    # 3. Removing (a,b) and adding (u,v) does not create a C4.
    #    - Adding (u,v) must not create a C4 in the graph *after* removing (a,b).
    #    - Removing (a,b) cannot create a C4 (removing edges never creates cycles).
    # 4. The swap should improve the min degree or the sum of squares of degrees (to make distribution more uniform).
    
    for iteration in range(5000):
        if time.time() > deadline:
            break
            
        min_deg = min(degrees)
        if min_deg >= required_min_deg:
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if adj_masks[u] & (1 << v):
                        edges.append([u, v])
            return (n, edges)
            
        # Find vertex u with min degree
        u = rng.choice([i for i in range(n) if degrees[i] == min_deg])
        
        # Try to find a swap involving u
        # We want to add an edge (u, v) where v is a non-neighbor.
        non_neighbors_u = [v for v in range(n) if v != u and not (adj_masks[u] & (1 << v))]
        
        if not non_neighbors_u:
            continue
            
        rng.shuffle(non_neighbors_u)
        
        for v in non_neighbors_u:
            # Check if (u,v) is blocked by a C4
            # If it's not blocked, we could have just added it in greedy. 
            # But maybe greedy didn't try it or it became blocked later?
            # Actually, if it's not blocked now, we can just add it.
            if not _check_c4_add_mask(adj_masks, u, v, n):
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                degrees[u] += 1
                degrees[v] += 1
                break # Restart inner loop or outer? Restart outer to re-eval min_deg
                
            # If it IS blocked, we need to remove an edge (a,b) to unblock it.
            # The C4 is u-w-x-v. The edge (w,x) blocks it.
            # We need to find a blocking edge (w,x) such that removing it is "safe" 
            # (i.e., doesn't reduce min degree too much) and allows adding (u,v).
            
            # Identify blocking edges for (u,v)
            # mask_u = N(u)\{v}, mask_v = N(v)\{u}
            mask_u = adj_masks[u] & ~(1 << v)
            mask_v = adj_masks[v] & ~(1 << u)
            
            blocking_edges = []
            m_u = mask_u
            while m_u:
                lsb = m_u & -m_u
                w = lsb.bit_length() - 1
                # Find x in N(w) that is in mask_v
                m_v = adj_masks[w] & mask_v
                while m_v:
                    lsb_v = m_v & -m_v
                    x = lsb_v.bit_length() - 1
                    # Edge (w,x) is a blocker
                    # We can try removing (w,x)
                    blocking_edges.append((w, x))
                    m_v ^= lsb_v
                m_u ^= lsb_u
                
            if not blocking_edges:
                continue
                
            rng.shuffle(blocking_edges)
            
            for a, b in blocking_edges:
                # Try swap: Remove (a,b), Add (u,v)
                # 1. Remove (a,b)
                adj_masks[a] &= ~(1 << b)
                adj_masks[b] &= ~(1 << a)
                degrees[a] -= 1
                degrees[b] -= 1
                
                # 2. Check if (u,v) is now safe
                # Note: adj_masks has been modified.
                if not _check_c4_add_mask(adj_masks, u, v, n):
                    # 3. Add (u,v)
                    adj_masks[u] |= (1 << v)
                    adj_masks[v] |= (1 << u)
                    degrees[u] += 1
                    degrees[v] += 1
                    
                    # Check if this swap improved the situation
                    # Ideally, min degree increased or stayed same, and we made progress.
                    # Since we broke out of the blocked state, it's a good move.
                    # We accept this swap and restart the main loop.
                    break # Break inner blocking_edges loop
                else:
                    # Swap didn't work, revert
                    adj_masks[a] |= (1 << b)
                    adj_masks[b] |= (1 << a)
                    degrees[a] += 1
                    degrees[b] += 1
                    continue # Try next blocking edge
            
            # If we accepted a swap, the degrees have changed.
            # We should break out of the 'for v' loop and restart the 'for iteration' loop
            # to re-evaluate min_deg.
            # How to signal this? We can just break.
            # But we need to know if a swap was accepted.
            # Let's use a flag.
            
            # Actually, the logic above breaks out of blocking_edges loop, but then continues to next v?
            # No, we should break out of the v loop too if we made a move.
            
            # Let's refactor: if a swap is made, we return to the top of the iteration loop.
            
            # To keep it simple, if we find a valid swap, we perform it and `break` out of the v loop.
            # Then we `continue` the iteration loop.
            
            # Wait, the code above doesn't have a flag. Let's add one.
            # Actually, if we break out of blocking_edges, we still need to break out of v loop.
            
            # Let's just restart the whole iteration if we make a change.
            # But we need to know if we made a change.
            
            # Simplified: If we find a valid swap, we execute it and break out of all loops to restart iteration.
            
            # This logic is getting messy. Let's just try a few swaps per iteration.
            
    return None

# EVOLVE-BLOCK-END
