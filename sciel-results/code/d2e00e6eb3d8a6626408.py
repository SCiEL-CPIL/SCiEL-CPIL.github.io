import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # 100s safety margin
    
    n = 16
    edges = []
    edge_index = {}
    for i in range(n):
        for j in range(i + 1, n):
            edge_index[(i, j)] = len(edges)
            edges.append((i, j))
            
    num_edges = len(edges)
    
    # Pre-compute K4s and their edges
    k4s = []
    k4_edges = [] # list of list of edge indices
    for comb in itertools.combinations(range(n), 4):
        v0, v1, v2, v3 = comb
        e_indices = [
            edge_index[(v0, v1)],
            edge_index[(v0, v2)],
            edge_index[(v0, v3)],
            edge_index[(v1, v2)],
            edge_index[(v1, v3)],
            edge_index[(v2, v3)]
        ]
        k4s.append(comb)
        k4_edges.append(e_indices)
        
    num_k4s = len(k4s)
    
    # For each edge, list of K4s it belongs to
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_list in enumerate(k4_edges):
        for e_idx in e_list:
            edge_to_k4s[e_idx].append(k4_idx)
            
    # 1. Initial Coloring: 1-Factorization
    # Vertices 0..14, 15(infinity)
    # Matching k contains {15, k} and {(k+i)%15, (k-i)%15} for i=1..7
    colors = [0] * num_edges
    for k in range(15):
        # Edge (15, k)
        # Ensure i < j for edge_index
        u, v = 15, k
        if u > v: u, v = v, u
        colors[edge_index[(u, v)]] = k
        
        for i in range(1, 8):
            u = (k + i) % 15
            v = (k - i) % 15
            if u > v: u, v = v, u
            colors[edge_index[(u, v)]] = k
            
    # Initialize k4_masks
    k4_masks = [0] * num_k4s
    for k4_idx, e_list in enumerate(k4_edges):
        mask = 0
        for e_idx in e_list:
            mask |= (1 << colors[e_idx])
        k4_masks[k4_idx] = mask
        
    # Verify initial validity (should be true)
    # for m in k4_masks:
    #    if m.bit_count() < 5: return None # Fallback?

    best_colors = colors[:]
    best_max_color = max(colors)
    
    # Helper to rebuild masks from colors
    def rebuild_masks():
        for k4_idx, e_list in enumerate(k4_edges):
            mask = 0
            for e_idx in e_list:
                mask |= (1 << colors[e_idx])
            k4_masks[k4_idx] = mask

    # Fast validity check
    def can_change_edge(e_idx, new_color):
        old_color = colors[e_idx]
        if old_color == new_color:
            return True
        
        for k4_idx in edge_to_k4s[e_idx]:
            mask = k4_masks[k4_idx]
            # Remove old color, add new color
            # Note: if old_color == new_color, handled above.
            # If new_color was already present, bit_count stays same or increases?
            # If new_color in mask:
            #   mask_new = (mask & ~(1<<old)) | (1<<new)
            #   If old was unique in this K4, count decreases by 1.
            #   If old was not unique, count stays same.
            #   If new was already present, adding it doesn't change count.
            #   So count change is:
            #   -1 if old was unique and new was present? No.
            #   Let's just compute.
            new_mask = (mask & ~(1 << old_color)) | (1 << new_color)
            if new_mask.bit_count() < 5:
                return False
        return True

    # Apply change and update masks
    def apply_change(e_idx, new_color):
        old_color = colors[e_idx]
        if old_color == new_color:
            return
        colors[e_idx] = new_color
        for k4_idx in edge_to_k4s[e_idx]:
            k4_masks[k4_idx] = (k4_masks[k4_idx] & ~(1 << old_color)) | (1 << new_color)

    # Compaction
    def compact():
        unique_c = sorted(set(colors))
        if len(unique_c) < 2:
            return
        mapping = {old: new for new, old in enumerate(unique_c)}
        for i in range(num_edges):
            colors[i] = mapping[colors[i]]
        rebuild_masks()
        
    max_iter = 100000
    iter_count = 0
    
    while time.time() < deadline:
        improved = False
        # Try to improve
        # Random order of edges
        edges_to_try = list(range(num_edges))
        random.shuffle(edges_to_try)
        
        for e_idx in edges_to_try:
            if time.time() > deadline: break
            current_c = colors[e_idx]
            
            # Try to lower to 0, 1, ... current_c - 1
            # Optimization: try 0 first. If valid, set to 0.
            # This is aggressive but effective for minimization.
            min_valid = -1
            for c in range(0, current_c):
                if can_change_edge(e_idx, c):
                    min_valid = c
                    break
            
            if min_valid != -1:
                apply_change(e_idx, min_valid)
                improved = True
        
        if improved:
            current_max = max(colors)
            if current_max < best_max_color:
                best_max_color = current_max
                best_colors = colors[:]
                # Compact to tighten the space
                compact()
                # Update best_max_color after compaction?
                # Compaction maps to 0..k-1, so max becomes k-1.
                # If k-1 < best_max_color, update.
                new_max = max(colors)
                if new_max < best_max_color:
                    best_max_color = new_max
                    best_colors = colors[:]
        
        else:
            # Stuck. Kick.
            e_idx = random.randint(0, num_edges - 1)
            current_c = colors[e_idx]
            
            found_kick = False
            # Try a few random valid colors
            for _ in range(20):
                new_c = random.randint(0, best_max_color + 2)
                if new_c == current_c:
                    continue
                if can_change_edge(e_idx, new_c):
                    apply_change(e_idx, new_c)
                    found_kick = True
                    break
            
            if not found_kick:
                # Try to compact and see if that helps?
                # Compaction doesn't change validity, just values.
                # It might allow a "decrease" that was blocked by high values?
                # No, validity depends on distinctness, not values.
                # But "decrease" logic tries 0, 1...
                # If we have colors {0, 1, 10}, and we want to change 10 to something lower.
                # If we compact to {0, 1, 2}, then 10 becomes 2.
                # Then we can try to change 2 to 0 or 1.
                # So compaction is very helpful for the "decrease" phase.
                compact()
                # After compaction, we might have a new max.
                # We don't update best here because compaction doesn't change the "quality" (number of distinct colors), 
                # but it changes the max value. 
                # Wait, the score is `max(colors)`.
                # If we compact, `max(colors)` drops.
                # So we SHOULD update best after compaction.
                new_max = max(colors)
                if new_max < best_max_color:
                    best_max_color = new_max
                    best_colors = colors[:]

        iter_count += 1
        if iter_count % 100 == 0:
            # Periodic compaction
            compact()
            new_max = max(colors)
            if new_max < best_max_color:
                best_max_color = new_max
                best_colors = colors[:]

    # Final compact best
    unique_c = sorted(set(best_colors))
    if len(unique_c) > 1:
        mapping = {old: new for new, old in enumerate(unique_c)}
        for i in range(num_edges):
            best_colors[i] = mapping[best_colors[i]]
            
    return best_colors
