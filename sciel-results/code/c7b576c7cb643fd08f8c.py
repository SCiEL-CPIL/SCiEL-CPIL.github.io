import random
import math
import time

# Precompute K4 structures
# Vertices 0..15
# Edges (i,j) with i<j. Index mapping:
# We need a way to map (i,j) to an index 0..119
# and vice versa.

def get_edge_index(i, j):
    # Placeholder, will build list
    return None

# Build edge list
edges = []
edge_to_idx = {}
for i in range(16):
    for j in range(i+1, 16):
        edge_to_idx[(i,j)] = len(edges)
        edges.append((i,j))

# Build K4s
# A K4 is a set of 4 vertices.
# For each set of 4 vertices, get the 6 edges.
k4s = []
for v1 in range(16):
    for v2 in range(v1+1, 16):
        for v3 in range(v2+1, 16):
            for v4 in range(v3+1, 16):
                verts = [v1, v2, v3, v4]
                edge_indices = []
                for x in range(4):
                    for y in range(x+1, 4):
                        u, v = verts[x], verts[y]
                        if u > v: u, v = v, u
                        edge_indices.append(edge_to_idx[(u,v)])
                k4s.append(tuple(edge_indices))

# Precompute for each edge, the list of K4 indices containing it
edge_to_k4s = [[] for _ in range(120)]
for k4_idx, k4 in enumerate(k4s):
    for e_idx in k4:
        edge_to_k4s[e_idx].append(k4_idx)

def construct():
    start_time = time.time()
    max_time = 110 # Leave 10s buffer
    
    # Try K from 13 to 16
    # We will allocate time to each K value
    # If we find a solution for K, we return it immediately.
    # If time runs out, we return the best found so far (or a fallback).
    
    best_overall_colors = None
    best_overall_K = 100
    
    # Allocate time: 
    # K=13: 30s
    # K=14: 30s
    # K=15: 30s
    # K=16: 20s
    # Total 110s
    
    time_budgets = {
        13: 30,
        14: 30,
        15: 30,
        16: 20
    }
    
    for K in [13, 14, 15, 16]:
        if time.time() - start_time >= max_time:
            break
            
        budget = time_budgets[K]
        limit_time = start_time + (time.time() - start_time) + budget
        # Actually, simpler: just check against global max_time and local budget logic
        # Let's just run until global time limit or local budget exhausted.
        # To be safe, I'll calculate a specific end time for this K attempt.
        current_elapsed = time.time() - start_time
        end_time_for_K = start_time + current_elapsed + budget
        
        # Initialize
        colors = [random.randint(0, K-1) for _ in range(120)]
        
        # Initialize frequency dicts and distinct counts
        # k4_freqs[k4_idx] is a dict {color: count}
        # k4_distinct[k4_idx] is the number of distinct colors
        k4_freqs = [{} for _ in range(len(k4s))]
        k4_distinct = [0] * len(k4s)
        penalty = 0
        
        for k4_idx, k4 in enumerate(k4s):
            freq = k4_freqs[k4_idx]
            for e in k4:
                c = colors[e]
                freq[c] = freq.get(c, 0) + 1
            k4_distinct[k4_idx] = len(freq)
            if k4_distinct[k4_idx] < 5:
                penalty += 1
                
        if penalty == 0:
            return colors
            
        T = 1.0
        T_min = 0.01
        alpha = 0.999
        best_colors = colors[:]
        best_penalty = penalty
        
        while time.time() < end_time_for_K and time.time() < start_time + max_time:
            e_idx = random.randint(0, 119)
            old_color = colors[e_idx]
            
            if K == 1:
                new_color = 0
            else:
                new_color = random.randint(0, K-1)
                if new_color == old_color:
                    continue
                    
            delta = 0
            affected_k4s = edge_to_k4s[e_idx]
            
            # Store info to revert if necessary
            # We need to store enough info to undo the freq and distinct changes
            # For each affected K4, we need:
            # - old distinct count
            # - old count of old_color
            # - old count of new_color
            # - whether new_color was present (to know if distinct increased)
            
            revert_info = []
            
            for k4_idx in affected_k4s:
                freq = k4_freqs[k4_idx]
                old_d = k4_distinct[k4_idx]
                old_contrib = 1 if old_d < 5 else 0
                
                old_count = freq.get(old_color, 0)
                new_count = freq.get(new_color, 0)
                
                # Apply modification
                if old_count > 0:
                    freq[old_color] = old_count - 1
                    if old_count == 1:
                        # Color removed from freq? We keep it as 0 or delete?
                        # Keeping as 0 is fine if we check >0, but for revert we need to restore.
                        # Actually, if we set to 0, it stays in dict. That's fine.
                        pass
                # else old_count is 0, do nothing (shouldn't happen as old_color is current)
                
                if new_count > 0:
                    freq[new_color] = new_count + 1
                else:
                    freq[new_color] = 1
                    
                # Calculate new distinct
                new_d = old_d
                if old_count == 1: # old_color count went 1->0, distinct decreases
                    new_d -= 1
                if new_count == 0: # new_color count went 0->1, distinct increases
                    new_d += 1
                    
                new_contrib = 1 if new_d < 5 else 0
                delta += (new_contrib - old_contrib)
                
                # Store revert info
                # We need to restore freq[old_color] to old_count
                # and freq[new_color] to new_count (or remove if 0)
                revert_info.append((k4_idx, old_d, old_count, new_count))
                
                # Update state (tentatively)
                k4_distinct[k4_idx] = new_d
                
            # Acceptance
            if delta < 0 or random.random() < math.exp(-delta / T):
                colors[e_idx] = new_color
                penalty += delta
                if penalty < best_penalty:
                    best_penalty = penalty
                    best_colors = colors[:]
                if penalty == 0:
                    return colors
            else:
                # Revert
                for k4_idx, old_d, old_count, new_count in revert_info:
                    freq = k4_freqs[k4_idx]
                    # Revert freq
                    freq[old_color] = old_count
                    if new_count > 0:
                        freq[new_color] = new_count
                    else:
                        # new_color was not present, we added it, so remove it
                        # Or set to 0? If we set to 0, it stays in dict.
                        # For consistency, let's delete it if it was 0.
                        # But if we delete, we must ensure it's not accessed incorrectly.
                        # Since we use .get(), it's safe.
                        # However, keeping 0 is simpler and avoids delete overhead.
                        # Let's just set to 0.
                        freq[new_color] = 0
                        
                    k4_distinct[k4_idx] = old_d
                    
            T *= alpha
            if T < T_min:
                T = T_min

        if best_penalty == 0:
            return best_colors
            
        # Update overall best
        if best_overall_colors is None or K < best_overall_K:
            best_overall_K = K
            best_overall_colors = best_colors[:]

    # If we didn't find a valid solution, return the best one we found
    # If we found nothing (shouldn't happen), return a fallback
    if best_overall_colors is not None:
        return best_overall_colors
        
    # Fallback: A simple valid coloring with more colors (e.g. 17)
    # (i+j) % 17 is a known valid construction for some bounds? 
    # Let's just return a random valid-looking one or the 17-color one.
    # The task says minimize colors. If we fail, we return the best attempt.
    # But if best_overall_colors is None, it means the loop didn't run or something went wrong.
    # Let's just return a dummy valid coloring to be safe, though it will score poorly.
    # Actually, if K=16 failed, best_overall_colors should be set.
    
    # Fallback to a known valid 17-coloring (just to be safe, though 16 should work)
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
