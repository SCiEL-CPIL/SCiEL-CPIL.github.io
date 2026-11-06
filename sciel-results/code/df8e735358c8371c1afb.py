import random
import time
import math

def construct():
    n = 20
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # State:
    # adj: list of lists of neighbors? Or bitmask?
    # Bitmask is faster for intersection checks.
    # But for SA, we need to update violation counts incrementally.
    # Let's use:
    # adj_mask: list of 20 ints (bitmasks)
    # common_counts: 20x20 matrix of common neighbor counts.
    # violations: int (count of pairs with common_counts >= 3)
    
    def init_state():
        adj_mask = [0] * n
        common_counts = [[0] * n for _ in range(n)]
        violations = 0
        edges = []
        
        # Greedy initialization
        random.shuffle(all_edges)
        for u, v in all_edges:
            # Check if adding (u,v) keeps it valid
            # Adding (u,v) increases common_counts[u][w] by 1 for all w in adj_mask[v]
            # and common_counts[v][w] by 1 for all w in adj_mask[u]
            
            # Check if any would become 3
            valid = True
            # w in N(v)
            mask_v = adj_mask[v]
            temp_mask = mask_v
            while temp_mask:
                lsb = temp_mask & -temp_mask
                temp_mask ^= lsb
                w = lsb.bit_length() - 1
                if common_counts[u][w] >= 2:
                    valid = False
                    break
            
            if valid:
                mask_u = adj_mask[u]
                temp_mask = mask_u
                while temp_mask:
                    lsb = temp_mask & -temp_mask
                    temp_mask ^= lsb
                    w = lsb.bit_length() - 1
                    if common_counts[v][w] >= 2:
                        valid = False
                        break
            
            if valid:
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                edges.append((u, v))
                
                # Update common counts
                # For w in N(v), common_counts[u][w] += 1
                # For w in N(u), common_counts[v][w] += 1
                # Note: common_counts[u][v] and common_counts[v][u] are 0 initially, 
                # but now u and v are connected. 
                # Common neighbors of u and v? 
                # N(u) and N(v) intersection.
                # Adding edge (u,v) doesn't change N(u) \ {v} or N(v) \ {u}.
                # So common_counts[u][v] remains 0? 
                # Wait. N(u) contains v. N(v) contains u.
                # Common neighbors of u and v are nodes w such that w in N(u) and w in N(v).
                # The edge (u,v) itself is not a neighbor.
                # So common_counts[u][v] is count of w != u, v such that (u,w) and (v,w) exist.
                # This value is unchanged by adding (u,v) directly?
                # No, adding (u,v) adds v to N(u) and u to N(v).
                # Does v become a common neighbor of u and some x?
                # Yes, if x is connected to v.
                # So for x in N(v), common_counts[u][x] increases by 1 (because v is now in N(u) and v is in N(x)).
                # Similarly for x in N(u), common_counts[v][x] increases by 1.
                
                # We need to track violations.
                # If common_counts[u][w] becomes 3, violations += 1.
                # If common_counts[u][w] becomes 2, violations unchanged.
                # If common_counts[u][w] was 3 and becomes 4, violations unchanged (still >=3).
                
                # Let's update counts and violations.
                mask_v = adj_mask[v]
                temp_mask = mask_v
                while temp_mask:
                    lsb = temp_mask & -temp_mask
                    temp_mask ^= lsb
                    w = lsb.bit_length() - 1
                    old_c = common_counts[u][w]
                    common_counts[u][w] = old_c + 1
                    common_counts[w][u] = old_c + 1
                    if old_c == 2:
                        violations += 1
                
                mask_u = adj_mask[u]
                temp_mask = mask_u
                while temp_mask:
                    lsb = temp_mask & -temp_mask
                    temp_mask ^= lsb
                    w = lsb.bit_length() - 1
                    old_c = common_counts[v][w]
                    common_counts[v][w] = old_c + 1
                    common_counts[w][v] = old_c + 1
                    if old_c == 2:
                        violations += 1
                
                # common_counts[u][v] and common_counts[v][u]
                # These are the number of common neighbors of u and v.
                # This value is NOT updated by the loop above?
                # The loop updates common_counts[u][w] for w in N(v).
                # If v is in N(v)? No.
                # If u is in N(u)? No.
                # So common_counts[u][v] is not touched.
                # But wait, is common_counts[u][v] affected?
                # N(u) gains v. N(v) gains u.
                # Common neighbors of u and v: w such that w in N(u) and w in N(v).
                # Previously, w in N(u) \ {v} and w in N(v) \ {u}.
                # Now, w in N(u) and w in N(v).
                # v is in N(u) but not N(v).
                # u is in N(v) but not N(u).
                # So the set of common neighbors is unchanged.
                # So common_counts[u][v] stays the same.
                
        return adj_mask, common_counts, violations, edges

    # SA Logic
    def sa_search():
        start_time = time.time()
        time_limit = 115.0
        
        best_edges = []
        best_score = 0
        
        for restart in range(10): # Multiple restarts
            if time.time() - start_time > time_limit:
                break
                
            adj_mask, common_counts, violations, edges = init_state()
            current_edges = len(edges)
            
            # If valid, update best
            if violations == 0 and current_edges > best_score:
                best_score = current_edges
                best_edges = edges[:]
            
            T = 10.0
            T_min = 0.01
            cooling = 0.999
            
            while time.time() - start_time < time_limit:
                if T < T_min:
                    break
                
                # Pick a random edge to toggle
                u, v = random.choice(all_edges)
                
                # Calculate delta
                # If edge exists, removing it. Else adding it.
                exists = (adj_mask[u] >> v) & 1
                
                # We need to calculate change in violations and edge count.
                # Let's simulate the move on copies? No, too slow.
                # Calculate delta_violations.
                
                delta_edges = -1 if exists else 1
                delta_violations = 0
                
                if exists:
                    # Remove (u,v)
                    # For w in N(v) \ {u}, common_counts[u][w] decreases by 1.
                    # For w in N(u) \ {v}, common_counts[v][w] decreases by 1.
                    
                    mask_v = adj_mask[v] & ~(1 << u)
                    temp_mask = mask_v
                    while temp_mask:
                        lsb = temp_mask & -temp_mask
                        temp_mask ^= lsb
                        w = lsb.bit_length() - 1
                        old_c = common_counts[u][w]
                        if old_c == 3:
                            delta_violations -= 1 # Violation removed
                        # If old_c > 3, still violation.
                        # If old_c < 3, no change in violation status.
                    
                    mask_u = adj_mask[u] & ~(1 << v)
                    temp_mask = mask_u
                    while temp_mask:
                        lsb = temp_mask & -temp_mask
                        temp_mask ^= lsb
                        w = lsb.bit_length() - 1
                        old_c = common_counts[v][w]
                        if old_c == 3:
                            delta_violations -= 1
                
                else:
                    # Add (u,v)
                    # For w in N(v), common_counts[u][w] increases by 1.
                    # For w in N(u), common_counts[v][w] increases by 1.
                    
                    mask_v = adj_mask[v]
                    temp_mask = mask_v
                    while temp_mask:
                        lsb = temp_mask & -temp_mask
                        temp_mask ^= lsb
                        w = lsb.bit_length() - 1
                        old_c = common_counts[u][w]
                        if old_c == 2:
                            delta_violations += 1 # New violation created
                    
                    mask_u = adj_mask[u]
                    temp_mask = mask_u
                    while temp_mask:
                        lsb = temp_mask & -temp_mask
                        temp_mask ^= lsb
                        w = lsb.bit_length() - 1
                        old_c = common_counts[v][w]
                        if old_c == 2:
                            delta_violations += 1
                
                delta_energy = -100.0 * delta_edges + 1.0 * delta_violations
                
                if delta_energy < 0 or random.random() < math.exp(-delta_energy / T):
                    # Accept move
                    if exists:
                        # Remove
                        adj_mask[u] ^= (1 << v)
                        adj_mask[v] ^= (1 << u)
                        edges.remove((u, v))
                        
                        mask_v = adj_mask[v]
                        temp_mask = mask_v
                        while temp_mask:
                            lsb = temp_mask & -temp_mask
                            temp_mask ^= lsb
                            w = lsb.bit_length() - 1
                            common_counts[u][w] -= 1
                            common_counts[w][u] -= 1
                        
                        mask_u = adj_mask[u]
                        temp_mask = mask_u
                        while temp_mask:
                            lsb = temp_mask & -temp_mask
                            temp_mask ^= lsb
                            w = lsb.bit_length() - 1
                            common_counts[v][w] -= 1
                            common_counts[w][v] -= 1
                    else:
                        # Add
                        adj_mask[u] |= (1 << v)
                        adj_mask[v] |= (1 << u)
                        edges.append((u, v))
                        
                        mask_v = adj_mask[v]
                        temp_mask = mask_v
                        while temp_mask:
                            lsb = temp_mask & -temp_mask
                            temp_mask ^= lsb
                            w = lsb.bit_length() - 1
                            common_counts[u][w] += 1
                            common_counts[w][u] += 1
                        
                        mask_u = adj_mask[u]
                        temp_mask = mask_u
                        while temp_mask:
                            lsb = temp_mask & -temp_mask
                            temp_mask ^= lsb
                            w = lsb.bit_length() - 1
                            common_counts[v][w] += 1
                            common_counts[w][v] += 1
                    
                    current_edges += delta_edges
                    violations += delta_violations
                    
                    if violations == 0 and current_edges > best_score:
                        best_score = current_edges
                        best_edges = edges[:]
                
                T *= cooling
        
        return best_edges

    edges = sa_search()
    
    # Final verification
    def is_valid(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] & adj[v]).bit_count() >= 3:
                    return False
        return True
        
    if not is_valid(edges):
        # Fallback
        return [[i, (i+1)%20] for i in range(20)]
        
    return [[u, v] for u, v in edges]

