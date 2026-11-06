import itertools
import random
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 140  # 10s safety margin
    
    n = 9
    all_edges = [list(e) for e in itertools.combinations(range(n), 5)]
    
    def is_not_2_colorable(edges):
        """Returns True if the hypergraph is NOT 2-colorable (no proper 2-coloring exists)."""
        m = len(edges)
        if m == 0:
            return False
        
        # Precompute edge bitmasks for fast color counting
        # For each edge, store (mask0, mask1) where mask0 has bits for vertices colored 0, mask1 for color 1
        # Actually, let's use a different approach: represent colors as two bitmasks
        # color0_mask: vertices colored 0
        # color1_mask: vertices colored 1
        # unassigned: ~(color0_mask | color1_mask) & ((1<<n)-1)
        
        full_mask = (1 << n) - 1
        
        # Precompute for each edge: the bitmask of its 5 vertices
        edge_masks = []
        for e in edges:
            mask = 0
            for v in e:
                mask |= (1 << v)
            edge_masks.append(mask)
        
        def check(color0, color1):
            """
            color0: bitmask of vertices colored 0
            color1: bitmask of vertices colored 1
            Returns True if a valid 2-coloring exists (i.e., the hypergraph IS 2-colorable)
            Returns False if no valid 2-coloring exists (NOT 2-colorable)
            """
            # Unit propagation with conflict detection
            while True:
                propagated = False
                conflict = False
                for mask in edge_masks:
                    c0 = bin(mask & color0).count('1')
                    c1 = bin(mask & color1).count('1')
                    un = 5 - c0 - c1
                    
                    if c0 == 5 or c1 == 5:
                        conflict = True
                        break
                    if c0 == 4 and un == 1:
                        # Must color the unassigned vertex as 1
                        unmask = mask & ~(color0 | color1)
                        if unmask & color1:
                            # Already colored 1, no change needed
                            pass
                        elif unmask & color0:
                            # Conflict: vertex in both colors
                            conflict = True
                            break
                        else:
                            # Assign to color1
                            color1 |= unmask
                            propagated = True
                    elif c1 == 4 and un == 1:
                        # Must color the unassigned vertex as 0
                        unmask = mask & ~(color0 | color1)
                        if unmask & color0:
                            # Already colored 0, no change needed
                            pass
                        elif unmask & color1:
                            # Conflict: vertex in both colors
                            conflict = True
                            break
                        else:
                            # Assign to color0
                            color0 |= unmask
                            propagated = True
                
                if conflict:
                    return False
                if not propagated:
                    break
            
            # Find an unassigned vertex
            unassigned = ~(color0 | color1) & full_mask
            if unassigned == 0:
                # All assigned, check for conflicts one more time
                for mask in edge_masks:
                    c0 = bin(mask & color0).count('1')
                    c1 = bin(mask & color1).count('1')
                    if c0 == 5 or c1 == 5:
                        return False
                return True  # Found a valid coloring
            
            # Pick the first unassigned vertex
            v = (unassigned & -unassigned).bit_length() - 1
            
            # Try color 0
            if check(color0 | (1 << v), color1):
                return True
            # Try color 1
            if check(color0, color1 | (1 << v)):
                return True
            
            return False
        
        return not check(0, 0)
    
    # Start with all edges
    best_edges = all_edges[:]
    best_len = len(best_edges)
    
    # Randomized Greedy Minimization with multiple restarts
    max_restarts = 50
    restart = 0
    while time.time() < deadline and restart < max_restarts:
        restart += 1
        
        current_edges = all_edges[:]
        random.shuffle(current_edges)
        
        i = 0
        while i < len(current_edges) and time.time() < deadline:
            # Try removing current_edges[i]
            if len(current_edges) <= best_len:
                # If we're already at or below best, we must try to improve
                pass
            
            test_edges = current_edges[:i] + current_edges[i+1:]
            
            if is_not_2_colorable(test_edges):
                current_edges = test_edges
                # Do not increment i
            else:
                i += 1
        
        if len(current_edges) < best_len:
            best_edges = current_edges[:]
            best_len = len(best_edges)
        
        # Early termination if we hit a good bound
        if best_len <= 28:
            break
    
    return {'n': n, 'edges': best_edges}
