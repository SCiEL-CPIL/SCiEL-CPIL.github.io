"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix):
    start_time = time.time()
    n = distance_matrix.shape[0]
    
    # 1. Compute a high-quality reference tour
    # Start with Nearest Neighbor
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    tour = [0]
    current = 0
    
    for _ in range(n - 1):
        dists = distance_matrix[current, :].copy()
        dists[visited] = np.inf
        next_node = np.argmin(dists)
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
    
    # Helper to calculate tour length
    def get_tour_length(t):
        return sum(distance_matrix[t[i], t[(i + 1) % n]] for i in range(n))
    
    # 2-Opt improvement
    def two_opt(t):
        improved = True
        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1:
                        continue
                    a, b = t[i], t[i + 1]
                    c, d = t[j], t[(j + 1) % n]
                    # Check if reversing segment (i+1, ..., j) improves length
                    # Old edges: (a,b) and (c,d)
                    # New edges: (a,c) and (b,d)
                    delta = (distance_matrix[a, c] + distance_matrix[b, d]) - (distance_matrix[a, b] + distance_matrix[c, d])
                    if delta < 0:
                        t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
                        improved = True
            if time.time() - start_time > 280:
                break
        return t

    # Or-opt improvement (moves 1, 2, or 3 nodes)
    def or_opt(t):
        improved = True
        while improved:
            improved = False
            for seg_len in [1, 2, 3]:
                for i in range(n):
                    # Extract segment of length seg_len starting at i
                    # Indices in tour: i, i+1, ..., i+seg_len-1
                    # We need to remove these and insert them elsewhere
                    # To keep it simple and fast, we check insertion at all positions j
                    # Note: Standard Or-opt is O(N^2 * k) for k segment lengths
                    # For N=50, this is manageable.
                    
                    # Get the segment nodes
                    seg_nodes = [t[(i + k) % n] for k in range(seg_len)]
                    
                    # Find the edges surrounding the segment
                    # Left edge: t[i-1] -> t[i]
                    # Right edge: t[i+seg_len-1] -> t[i+seg_len]
                    # Let prev_idx = (i - 1) % n
                    # Let next_idx = (i + seg_len) % n
                    
                    prev_node = t[(i - 1) % n]
                    next_node = t[(i + seg_len) % n]
                    
                    first_seg_node = seg_nodes[0]
                    last_seg_node = seg_nodes[-1]
                    
                    # Cost of removing segment
                    # Old edges: (prev_node, first_seg_node) and (last_seg_node, next_node)
                    # New edge after removal: (prev_node, next_node)
                    remove_cost = (distance_matrix[prev_node, first_seg_node] + distance_matrix[last_seg_node, next_node]) - distance_matrix[prev_node, next_node]
                    
                    # Try inserting segment at every possible position j
                    # Position j means inserting between t[j-1] and t[j]
                    # We must skip positions that overlap with the removed segment
                    for j in range(n):
                        # Check if j is in the removed segment range
                        # The removed segment occupies indices i, i+1, ..., i+seg_len-1
                        # The insertion point j is between j-1 and j.
                        # If the insertion point is inside the removed segment, it's invalid.
                        # Specifically, if j is in [i, i+seg_len-1], it's overlapping.
                        # Actually, we are moving the segment. The destination j must not be such that the segment overlaps itself.
                        # Simple check: if j is in the range of the removed nodes, skip.
                        # The removed nodes are at indices i ... i+seg_len-1.
                        # The insertion is before index j.
                        # If j == i, we are inserting right before the start of the segment (no move).
                        # If j == i+seg_len, we are inserting right after the end (no move).
                        # If i < j < i+seg_len, it's an invalid overlap.
                        
                        # Let's just check if the insertion position j is "inside" the removed block.
                        # The removed block covers tour positions i to i+seg_len-1.
                        # Insertion before j affects edges (j-1, j).
                        # If j is in (i, i+seg_len), the edges are part of the segment or adjacent in a way that breaks the "move" logic simply.
                        # Standard Or-opt implementation usually iterates j and skips if j is in the removed range.
                        
                        if i < j < i + seg_len:
                            continue
                        if j == i or j == i + seg_len:
                            continue # No change or trivial

                        # Calculate insertion cost
                        # New edges: (t[j-1], first_seg_node) and (last_seg_node, t[j])
                        # Old edges at insertion point: (t[j-1], t[j])
                        
                        p = t[(j - 1) % n]
                        q_node = t[j]
                        
                        insert_cost = (distance_matrix[p, first_seg_node] + distance_matrix[last_seg_node, q_node]) - distance_matrix[p, q_node]
                        
                        total_delta = remove_cost + insert_cost
                        
                        if total_delta < 0:
                            # Perform the move
                            # Remove segment from t
                            # t is a list. 
                            # Remove indices i to i+seg_len-1
                            # Be careful with list slicing if i+seg_len > n, but we use modulo for nodes, 
                            # however the list indices are 0..n-1.
                            # The segment is at list indices i ... i+seg_len-1.
                            # Since we are working with a list representation of the tour, 
                            # and the tour is circular, we must handle the wrap-around carefully if i+seg_len > n.
                            # However, for Or-opt, it's often easier to work with a linear representation 
                            # and handle the circular nature by checking all j.
                            # Let's extract the nodes, remove them, and insert them.
                            
                            # To avoid complex list manipulation with wrap-around, 
                            # we can just reconstruct the tour list.
                            # But that's O(N). N=50, so it's fine.
                            
                            # Remove segment
                            # We need to remove nodes at indices i, i+1, ..., i+seg_len-1 from the list `t`.
                            # Note: `t` is 0-indexed list of node IDs.
                            # The indices in the list are 0..n-1.
                            # If i + seg_len > n, the segment wraps around.
                            # Example: N=5, i=4, seg_len=2. Nodes at 4, 0.
                            # This makes list manipulation tricky.
                            
                            # Alternative: Convert tour to a list, perform the move on the list, 
                            # but ensure we don't double count or mess up indices.
                            # Given N=50, we can just do a simple O(N) move.
                            
                            # Let's create a new list
                            new_t = []
                            # Add nodes not in the segment
                            # The segment nodes are seg_nodes.
                            # We iterate through t and skip the ones in seg_nodes? 
                            # No, seg_nodes might have duplicates? No, TSP tour has unique nodes.
                            # So we can check if node is in seg_nodes.
                            # But we need to preserve order.
                            
                            # Let's do it by indices.
                            # If no wrap-around (i + seg_len <= n):
                            if i + seg_len <= n:
                                # Remove t[i : i+seg_len]
                                remaining = t[:i] + t[i+seg_len:]
                                # Insert at position j
                                # Note: j is the index in the ORIGINAL tour.
                                # If we removed elements before j, the index shifts.
                                # This is getting complicated.
                                
                            # Simpler approach for Or-opt with N=50:
                            # Just try all moves and check length.
                            # Generate a new tour candidate.
                            # Remove segment, insert at j.
                            
                            # Let's use a helper to do the move safely.
                            def do_or_move(t_list, i, j, seg_len):
                                # i: start index of segment
                                # j: insertion index (before node at j in original list)
                                # seg_len: length of segment
                                
                                # Extract segment
                                seg = []
                                for k in range(seg_len):
                                    idx = (i + k) % n
                                    seg.append(t_list[idx])
                                
                                # Remove segment from t_list
                                # We need to remove indices i, i+1, ... i+seg_len-1
                                # If wrap around, it's harder.
                                # Let's assume we can just rebuild the list.
                                
                                # Identify indices to keep
                                keep_indices = [x for x in range(n) if x not in set([(i + k) % n for k in range(seg_len)])]
                                
                                # The kept nodes in order
                                kept_nodes = [t_list[k] for k in keep_indices]
                                
                                # Now insert seg into kept_nodes at position j
                                # But j was an index in the original list.
                                # We need to find where j falls in the new kept_nodes list.
                                # If j was in the removed segment, we handled that with skip.
                                # If j < i, the position in kept_nodes is j.
                                # If j > i + seg_len, the position in kept_nodes is j - seg_len.
                                # If i < j < i + seg_len, we skipped it.
                                
                                if j < i:
                                    insert_pos = j
                                else:
                                    insert_pos = j - seg_len
                                
                                # Insert seg into kept_nodes at insert_pos
                                new_list = kept_nodes[:insert_pos] + seg + kept_nodes[insert_pos:]
                                return new_list

                            new_tour = do_or_move(t, i, j, seg_len)
                            # Verify length
                            if get_tour_length(new_tour) < get_tour_length(t):
                                t = new_tour
                                improved = True
                                if time.time() - start_time > 280:
                                    return t
        return t

    # Apply 2-Opt
    tour = two_opt(tour)
    
    # Apply Or-Opt
    tour = or_opt(tour)
    
    # Alternate a few times to ensure convergence
    for _ in range(3):
        if time.time() - start_time > 280:
            break
        tour = two_opt(tour)
        tour = or_opt(tour)
        
    # 2. Build Heuristic Matrix
    # Base: Inverse distance
    # Add small epsilon to avoid division by zero (diagonals)
    diag_mask = np.eye(n, dtype=bool)
    safe_dist = distance_matrix.copy()
    safe_dist[diag_mask] = 1.0 # Avoid div by zero, will be masked later
    
    base = 1.0 / safe_dist
    
    # Boost edges in the reference tour
    # We want these edges to be highly attractive.
    # A factor of 1000 is used in the current code. 
    # Let's try a slightly more dynamic boost or just stick with a high constant.
    # The ACO uses pheromone * heuristic. 
    # If heuristic is very large for tour edges, ants will almost always pick them.
    # This effectively forces the ACO to find the reference tour, which is good if the reference tour is near-optimal.
    
    for i in range(n):
        j = (i + 1) % n
        base[tour[i], tour[j]] = 1000.0
        
    # Ensure diagonal is not attractive
    base[diag_mask] = 1e-9
    
    # Ensure all values are positive
    base = np.maximum(base, 1e-9)
    
    return base
# EVOLVE-BLOCK-END
