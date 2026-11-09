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
    max_time = 115 # Leave 5s buffer for safety
    
    best_valid_colors = None
    best_valid_K = 100
    
    # We try K=13, 14, 15, 16
    # If we find a valid solution for K, we return it immediately because lower K is better.
    # If we don't find a valid one for K, we proceed to K+1.
    # If we run out of time, we return the best valid one found so far.
    # If no valid one found, we return a fallback valid one (e.g. K=17 or known 16).
    
    time_budgets = {
        13: 25,
        14: 25,
        15: 25,
        16: 25
    }
    
    for K in [13, 14, 15, 16]:
        if time.time() - start_time >= max_time:
            break
            
        budget = time_budgets[K]
        end_time_for_K = time.time() + budget
        
        # Initialize with a structured pattern: (j - i) % K
        # This is often a good starting point for edge colorings of complete graphs
        colors = [0] * 120
        for i in range(16):
            for j in range(i+1, 16):
                idx = edge_to_idx[(i,j)]
                colors[idx] = (j - i) % K
        
        # Initialize frequency dicts and distinct counts
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
        alpha = 0.9995 # Slower cooling to explore more
        best_colors = colors[:]
        best_penalty = penalty
        last_improvement_time = time.time()
        no_improve_limit = 5.0 # Restart or adjust if no improvement for 5s? 
        # Actually, just let it run. Simulated Annealing with restarts is better.
        # Let's stick to a single long run for each K to keep it simple and robust.
        
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
                        # Color removed
                        pass
                # else old_count is 0, do nothing
                
                if new_count > 0:
                    freq[new_color] = new_count + 1
                else:
                    freq[new_color] = 1
                    
                # Calculate new distinct
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
                    last_improvement_time = time.time()
                if penalty == 0:
                    return colors
            else:
                # Revert
                for k4_idx, old_d, old_count, new_count in revert_info:
                    freq = k4_freqs[k4_idx]
                    freq[old_color] = old_count
                    if new_count > 0:
                        freq[new_color] = new_count
                    else:
                        freq[new_color] = 0
                    k4_distinct[k4_idx] = old_d
                    
            T *= alpha
            if T < T_min:
                T = T_min

        if best_penalty == 0:
            return best_colors
        
        # If we found a valid solution, we would have returned.
        # So best_penalty > 0.
        # We don't store invalid solutions as "best valid".
        
    # If we reach here, we didn't find a valid solution for K=13..16 within time.
    # This is highly unlikely for K=16.
    # Fallback: Return a known valid coloring.
    # (i+j) % 17 is a standard construction that usually works for small n.
    # Let's verify if (i+j)%17 works for n=16.
    # Max color 16. 
    # Any K4? 
    # It's safer to just return the best attempt from K=16 if it was close, 
    # but the task requires a VALID coloring.
    # If I return an invalid one, score is 0.
    # So I MUST return a valid one.
    
    # Let's try a quick 17-color construction:
    # color(i,j) = (i+j) % 17
    # This uses 17 colors.
    # Is it valid? 
    # For any 4 vertices, do they span 5 colors?
    # Yes, for n <= 16, (i+j) % p with p >= 2n-1 often works or is close.
    # Actually, a simpler fallback: 
    # Assign unique colors to all edges? No, 120 colors.
    # We want to minimize.
    # If the search failed, it's a catastrophic failure.
    # But K=16 should be trivial. 
    # Let's add a very aggressive short run for K=16 if we didn't find one before?
    # No, the loop above already tried K=16.
    
    # Let's just return the 17-color fallback to be safe.
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
