import random
import math
import time
import sys

# Precompute K4 structures
edges = []
edge_to_idx = {}
for i in range(16):
    for j in range(i+1, 16):
        edge_to_idx[(i,j)] = len(edges)
        edges.append((i,j))

# Build K4s
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
    max_time = 115 # Leave 5s buffer for safety
    
    # We want to minimize K.
    # Try K=13, 14, 15, 16.
    # If we find a valid one, return it.
    # If we fail all, return the best invalid one? No, must be valid.
    # The task implies a valid coloring certifies f(16) <= colors.
    # If we can't find a valid one, we should probably return a valid one with higher K.
    # But the loop goes 13->16. If 16 fails, we return fallback.
    
    # Strategy:
    # Give more time to higher K if lower K fails?
    # Or just try them in order.
    # 13 is the hardest. 16 is the easiest.
    # If we start with 13 and fail, we spent time.
    # Maybe start with 16 to get a "warm start" or just a valid solution, then try to reduce?
    # But SA from scratch for each K is fine.
    
    # Let's try 13, 14, 15, 16 with time budgets.
    # If 13 succeeds, return 13.
    # If 13 fails, try 14.
    # etc.
    
    time_budgets = {
        13: 25,
        14: 25,
        15: 25,
        16: 30
    }
    
    for K in [13, 14, 15, 16]:
        if time.time() - start_time >= max_time:
            break
            
        budget = time_budgets[K]
        end_time_for_K = time.time() + budget
        
        # Run SA for this K
        best_colors, best_penalty = run_sa(K, end_time_for_K)
        
        if best_penalty == 0:
            return best_colors
            
        # If we didn't find a valid one, keep going to next K
        # We don't carry over the state because K changes (color space changes)
        
    # If we reach here, we failed to find a valid coloring for K=13..16
    # This should not happen as K=16 is very likely solvable.
    # Fallback: return a valid 17-coloring? 
    # The interface expects colors in 0..119. 
    # If we return a 17-coloring, it uses colors 0..16. 
    # Is that valid? Yes, "integer colours in 0..119".
    # But we want to minimize. If we couldn't find 16, 17 is the best we can do?
    # Actually, if K=16 SA failed, it might just be bad luck.
    # Let's try a simple greedy construction for K=16 or higher if SA fails.
    
    # Greedy construction:
    # Assign colors one by one?
    # Or use a known algebraic construction.
    # For K16, a proper edge coloring with 15 colors exists (1-factorization).
    # But we need every K4 to have >= 5 colors.
    # A 1-factorization uses 15 colors. Each color is a perfect matching.
    # In a K4, how many colors from a 1-factorization?
    # A K4 has 6 edges. In a 1-factorization of K16, each vertex is incident to exactly one edge of a given color.
    # So in a K4 (4 vertices), for a specific color, can we have more than 1 edge?
    # No, because it's a matching. So each color appears at most once in any K4?
    # Wait. A matching on 16 vertices has 8 edges.
    # A K4 has 4 vertices. The induced subgraph on these 4 vertices in a matching has at most 2 edges (if the matching pairs them up).
    # Actually, a matching has no shared vertices. So in a set of 4 vertices, a matching can have at most 2 edges (e.g., (0,1) and (2,3)).
    # So each color appears at most 2 times in a K4.
    # We have 15 colors.
    # The 6 edges of the K4 are colored with some of these 15 colors.
    # Since each color appears at most 2 times, we need at least 3 colors to cover 6 edges? No.
    # If we use only 3 colors, max edges covered is 3*2=6. So it's possible to have a K4 with only 3 colors.
    # So 1-factorization is NOT guaranteed to satisfy the "5 colors in K4" constraint.
    
    # However, if we use more colors, it's easier.
    # With K=16, we have 16 colors.
    
    # Let's just return a random 16-coloring if SA fails, and hope it's valid? No, must be valid.
    # Let's implement a simple backtracking or just return the best invalid one? 
    # "An invalid result... scores 0".
    # So we MUST return a valid coloring.
    
    # If SA for K=16 fails, we try a different heuristic for K=16.
    # Or we increase K to 17, 18... until we find one.
    
    # Let's add a fallback loop for K=17, 18...
    for K in range(17, 20):
        if time.time() - start_time >= max_time:
            break
        best_colors, best_penalty = run_sa(K, time.time() + 5)
        if best_penalty == 0:
            return best_colors
            
    # Final fallback: return a valid coloring with many colors.
    # e.g. color(i,j) = i*16 + j? No, must be < 120.
    # color(i,j) = (i*16 + j) % 120? 
    # If we use all distinct colors for each edge, every K4 has 6 distinct colors.
    # This is always valid.
    # We just need to map 120 edges to 0..119.
    # This uses 120 colors.
    # It is valid.
    # It minimizes nothing, but it's valid.
    # We should only reach this if all else fails.
    
    return [idx for idx in range(120)]

def run_sa(K, end_time):
    """
    Run Simulated Annealing for a given number of colors K.
    Returns (best_colors, best_penalty).
    """
    # Initialize
    colors = [random.randint(0, K-1) for _ in range(120)]
    
    # State: k4_distinct, k4_freqs
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
        return colors, 0
        
    best_colors = colors[:]
    best_penalty = penalty
    
    T = 1.0
    T_min = 0.01
    alpha = 0.9995 # Slower cooling
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < end_time:
        # Select edge
        # Bias towards edges in violating K4s
        # Identify violating K4s
        # To avoid iterating all K4s every step, we can maintain a list of "bad" edges?
        # Or just pick random edge. It's fast enough.
        
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
        
        # Calculate delta
        # We need to know if distinct count changes.
        # For each affected K4:
        # old_count = freq[old_color]
        # new_count = freq.get(new_color, 0)
        
        # We will tentatively apply, calculate delta, and revert if rejected.
        
        revert_info = []
        
        for k4_idx in affected_k4s:
            freq = k4_freqs[k4_idx]
            old_d = k4_distinct[k4_idx]
            old_contrib = 1 if old_d < 5 else 0
            
            old_count = freq.get(old_color, 0)
            new_count = freq.get(new_color, 0)
            
            # Apply
            if old_count > 0:
                freq[old_color] = old_count - 1
                if old_count == 1:
                    # distinct decreases
                    pass 
            # else old_count 0 (shouldn't happen)
            
            if new_count > 0:
                freq[new_color] = new_count + 1
            else:
                freq[new_color] = 1
                # distinct increases
            
            new_d = old_d
            if old_count == 1:
                new_d -= 1
            if new_count == 0:
                new_d += 1
                
            new_contrib = 1 if new_d < 5 else 0
            delta += (new_contrib - old_contrib)
            
            revert_info.append((k4_idx, old_d, old_count, new_count))
            k4_distinct[k4_idx] = new_d
            
        # Acceptance
        if delta < 0 or random.random() < math.exp(-delta / T):
            colors[e_idx] = new_color
            penalty += delta
            if penalty < best_penalty:
                best_penalty = penalty
                best_colors = colors[:]
                no_improve_count = 0
                if penalty == 0:
                    return colors, 0
            else:
                no_improve_count += 1
        else:
            # Revert
            for k4_idx, old_d, old_count, new_count in revert_info:
                freq = k4_freqs[k4_idx]
                freq[old_color] = old_count
                if new_count > 0:
                    freq[new_color] = new_count
                else:
                    # Remove or set to 0? 
                    # If we set to 0, it stays in dict. 
                    # If we delete, it's gone.
                    # Since we use .get(), setting to 0 is safe and faster than delete?
                    # Actually, deleting a key is O(1) but involves hash.
                    # Setting to 0 is just an assignment.
                    # But if we leave 0s in the dict, `len(freq)` will be wrong!
                    # Ah! `k4_distinct` is calculated as `len(freq)`.
                    # If we leave keys with value 0, `len(freq)` is incorrect.
                    # So we MUST delete the key if count becomes 0.
                    
                    # In the "Apply" phase:
                    # if old_count == 1, we set freq[old_color] = 0. 
                    # We did NOT delete it.
                    # So `len(freq)` in the next step? 
                    # We are updating `k4_distinct` manually. 
                    # But if we revert, we restore old_count.
                    # If we accept, we keep the state.
                    # If we keep the state with freq[old_color]=0, then `len(freq)` is wrong for future calculations?
                    # No, we don't use `len(freq)` for future calculations. We use `k4_distinct`.
                    # However, if we later change another edge in the same K4, we use `freq.get(...)`.
                    # If freq[old_color] is 0, `freq.get(old_color, 0)` returns 0. Correct.
                    # But if we ever need to calculate distinct from scratch, it would be wrong.
                    # We don't.
                    # BUT, if we delete the key, `freq.get` returns 0. Also correct.
                    # To keep the dict clean and small, we should delete keys with value 0.
                    
                    # Let's modify the "Apply" phase to delete keys when they become 0.
                    # And "Revert" to restore them.
                    
                    pass # Handled below
                    
            # Correct Revert Logic:
            # We need to restore freq[old_color] to old_count.
            # And freq[new_color] to new_count.
            
            for k4_idx, old_d, old_count, new_count in revert_info:
                freq = k4_freqs[k4_idx]
                
                # Restore old_color
                if old_count > 0:
                    freq[old_color] = old_count
                else:
                    # old_color was not in freq (count 0)
                    # It shouldn't be there. If it is, delete it.
                    if old_color in freq:
                        del freq[old_color]
                        
                # Restore new_color
                if new_count > 0:
                    freq[new_color] = new_count
                else:
                    # new_color was not in freq (count 0)
                    # It might be there with 0 or 1 (if we added it).
                    if new_color in freq:
                        del freq[new_color]
                        
                k4_distinct[k4_idx] = old_d
                
        T *= alpha
        if T < T_min:
            T = T_min
            
        # Restart logic
        if no_improve_count > max_no_improve:
            # Restart from best_colors
            colors = best_colors[:]
            # Rebuild state from best_colors
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
            T = 1.0
            no_improve_count = 0
            
    return best_colors, best_penalty
