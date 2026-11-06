import time
import itertools
import random

DEADLINE = 140.0  # Safety margin

def _now():
    return time.time()

def _check_not_2_colorable_bitmask(edges_masks, n):
    """
    edges_masks: list of integers, each representing an edge as a bitmask.
    n: number of vertices.
    Returns True if NOT 2-colorable.
    """
    e_count = len(edges_masks)
    if e_count == 0:
        return False # Empty hypergraph is 2-colorable
        
    # Iterate through all 2^n colorings
    # C is a bitmask where bit i is 1 if vertex i is color 1, 0 if color 0.
    for C in range(1 << n):
        valid_coloring = True
        for mask in edges_masks:
            # Check if edge is monochromatic
            # Mono 1: all vertices in mask are in C -> (mask & C) == mask
            # Mono 0: no vertices in mask are in C -> (mask & C) == 0
            if (mask & C) == mask or (mask & C) == 0:
                valid_coloring = False
                break
        if valid_coloring:
            return False # Found a valid coloring, so it IS 2-colorable
            
    return True # No valid coloring found, so NOT 2-colorable

def _greedy_reduce(n, max_trials=5):
    """
    Try to reduce the complete hypergraph K_n^(5) by greedily removing edges.
    Returns the best (smallest) non-2-colorable edge set found, or None.
    """
    all_edges = list(itertools.combinations(range(n), 5))
    # Convert to masks for fast checking
    all_masks = [0] * len(all_edges)
    for i, edge in enumerate(all_edges):
        m = 0
        for v in edge:
            m |= (1 << v)
        all_masks[i] = m
            
    best_edges = None
    best_count = len(all_edges) + 1
    
    # Check if the complete graph is even a candidate (it must be not 2-colorable)
    # For n=9, it is. For n=8, it is not.
    if not _check_not_2_colorable_bitmask(all_masks, n):
        return None
        
    rng = random.Random(42 + n)
    
    for trial in range(max_trials):
        current_indices = list(range(len(all_edges)))
        # Shuffle to break symmetry and try different removal orders
        rng.shuffle(current_indices)
        
        # We need to maintain the list of active edges
        # Using a set of indices might be slow for removal checks if we rebuild masks.
        # Instead, keep a list of active masks.
        
        active_masks = [all_masks[i] for i in current_indices]
        
        # Greedy removal
        i = 0
        while i < len(active_masks):
            if _now() > DEADLINE:
                return best_edges
            
            # Try removing active_masks[i]
            trial_masks = active_masks[:i] + active_masks[i+1:]
            
            if _check_not_2_colorable_bitmask(trial_masks, n):
                # It is still not 2-colorable, so keep it removed
                active_masks = trial_masks
                # Do not increment i, as the next element is now at i
            else:
                # It became 2-colorable, so we must keep the edge
                i += 1
                
        # End of greedy pass
        if len(active_masks) < best_count:
            best_count = len(active_masks)
            # Convert back to vertex lists for return
            # We lost the original indices, but we can just return the masks?
            # No, return format requires vertex lists.
            # We need to map masks back to vertex lists.
            # Since we started from all_edges, we can just store the vertex lists directly.
            
            # Let's refactor to store vertex lists.
            pass

    # Refactored approach: store vertex lists directly
    best_edges = None
    best_count = len(all_edges) + 1
    
    for trial in range(max_trials):
        current_edges = [list(e) for e in all_edges]
        rng.shuffle(current_edges)
        
        i = 0
        while i < len(current_edges):
            if _now() > DEADLINE:
                break
                
            # Convert current_edges to masks for check
            # This is O(E * 5). E is at most 126 for n=9.
            masks = []
            for edge in current_edges:
                if edge == current_edges[i]:
                    continue # Skip the one we're testing? No, we need to check the set WITHOUT it.
                m = 0
                for v in edge:
                    m |= (1 << v)
                masks.append(m)
            
            # Actually, building masks every time is O(E^2) if done naively in the loop.
            # Let's optimize the checker to take a list of edges and an index to skip?
            # Or just maintain the mask list.
            
            # Let's restart the loop logic with maintained masks.
            break
        
        # Better approach:
        active_masks = [all_masks[i] for i in current_indices] # Wait, current_indices was shuffled.
        # Let's just use a list of masks and a list of edges.
        
        pass

    # Simplest robust approach:
    # 1. Start with list of edges (vertex lists).
    # 2. Iterate through indices.
    # 3. To check if removing edge i works:
    #    - Build mask list from all edges except i.
    #    - Call checker.
    #    - If checker says "Not 2-colorable", remove edge i.
    #    - Else, keep it.
    
    # This is O(N^2) checks in worst case, each check O(2^n * E).
    # For n=9, E=126. 2^9=512. Check cost ~ 512*126 = 64k ops.
    # 126 removals * 64k = 8M ops per pass.
    # Python can do ~10M-100M simple ops/sec. So one pass takes < 1s.
    # We can do many passes.
    
    best_edges = None
    best_count = len(all_edges) + 1
    
    for trial in range(max_trials):
        current_edges = [list(e) for e in all_edges]
        rng.shuffle(current_edges)
        
        i = 0
        while i < len(current_edges):
            if _now() > DEADLINE:
                return best_edges
                
            # Prepare masks for all edges except current_edges[i]
            masks = []
            for j, edge in enumerate(current_edges):
                if j == i:
                    continue
                m = 0
                for v in edge:
                    m |= (1 << v)
                masks.append(m)
            
            if _check_not_2_colorable_bitmask(masks, n):
                # Safe to remove
                current_edges.pop(i)
                # Do not increment i
            else:
                i += 1
                
        if len(current_edges) < best_count:
            best_count = len(current_edges)
            best_edges = [list(e) for e in current_edges]
            
    return best_edges

def construct():
    start = _now()
    
    best_global_edges = None
    best_global_count = 1000
    
    # Try n from 9 to 15
    # n=9 is the smallest where K_n^(5) is not 2-colorable.
    for n in range(9, 16):
        if _now() > DEADLINE - 10: # Reserve time for return
            break
            
        # Limit trials for larger n as they are slower
        trials = 5 if n <= 10 else 2
        
        result = _greedy_reduce(n, trials)
        
        if result is not None:
            if len(result) < best_global_count:
                best_global_count = len(result)
                best_global_edges = result
                print(f"Found non-2-colorable with n={n}, e={best_global_count}")
                
        # If we found something very small (e.g., <= 55), we might stop early?
        # But we want to minimize. Let's keep searching other n's if time permits.
        # However, larger n usually allows for sparser non-2-colorable graphs?
        # Actually, m(k) is usually achieved at specific n.
        
    # If we found nothing (shouldn't happen for n=9), fallback to K_9^(5)
    if best_global_edges is None:
        edges = [list(e) for e in itertools.combinations(range(9), 5)]
        return {'n': 9, 'edges': edges}
        
    return {'n': n, 'edges': best_global_edges} # n from the last loop iteration? 
    # Wait, n is local to the loop. We need to store the n that produced the best result.
    
    # Fix: store n with the result.
    
    # Re-implementing construct with proper n tracking.
