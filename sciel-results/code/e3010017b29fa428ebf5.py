import random
import math
import time
import sys

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
    max_time = 115.0 # Leave 5s buffer for safety

    # We will try to find a solution with K=13, then 14, etc.
    # But since K=13 is very tight, we might spend most time there.
    # If we fail K=13, we try K=14.
    
    best_solution = None
    best_K = 100
    
    for K in [13, 14, 15, 16]:
        if time.time() - start_time > max_time:
            break
            
        # Time budget for this K
        # If K=13, give it 40s. If K=14, 30s. etc.
        # Actually, let's just use a fraction of remaining time.
        remaining = max_time - (time.time() - start_time)
        if remaining <= 0:
            break
        
        # Allocate time based on K
        if K == 13:
            budget = min(40.0, remaining)
        elif K == 14:
            budget = min(30.0, remaining)
        elif K == 15:
            budget = min(20.0, remaining)
        else:
            budget = min(10.0, remaining)
            
        end_time = start_time + (time.time() - start_time) + budget
        
        # Run SA for this K
        result = run_sa(K, end_time)
        
        if result is not None:
            # Check validity (should be valid if penalty 0)
            if best_solution is None or K < best_K:
                best_solution = result
                best_K = K
                
        # If we found a valid solution for K=13, we can stop?
        # The task says minimize colors. So if we find K=13, that's optimal (given lower bound 13).
        # Wait, lower bound is 13. So 13 is the best possible.
        if K == 13 and result is not None:
            return result

    # If we didn't find a valid solution for any K, we should return the best partial solution?
    # The contract says "valid colouring certifies...". If we return an invalid one, score is 0.
    # So we MUST return a valid one.
    # If SA failed to find penalty 0, we need a fallback.
    # A simple valid coloring with more colors:
    # e.g., K=17 coloring: (i+j)%17 is often valid for small n, but let's check n=16.
    # Actually, a random assignment with enough colors (e.g. 20) is almost surely valid.
    # But we want to minimize.
    # If SA failed, it's likely because it got stuck.
    # We should ensure our SA is robust enough to find K=14 or 15 easily.
    # K=14 and 15 are much easier than 13.
    
    if best_solution is not None:
        return best_solution
    
    # Fallback: Generate a random valid coloring with K=16 or higher.
    # K=16 is very likely to be solvable quickly.
    # Let's just do a quick SA for K=16 with a very short time if we got here.
    # Or just return a known valid pattern?
    # (i+j)%17 for K=17 is a standard construction for f(n) bounds.
    # Let's check if (i+j)%17 works for n=16.
    # In any K4, vertices a,b,c,d.
    # Edges: a+b, a+c, a+d, b+c, b+d, c+d mod 17.
    # Are there 5 distinct colors?
    # It's possible to have collisions.
    # Example: 0,1,2,3.
    # 1,2,3, 3,4, 5. Colors: {1,2,3,4,5}. 5 distinct.
    # Example: 0,1,3,4.
    # 1,3,4, 4,5, 7. Colors: {1,3,4,5,7}. 5 distinct.
    # It seems (i+j)%p for p>=17 is a good candidate.
    
    fallback_colors = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    # Verify fallback just in case
    if is_valid(fallback_colors):
        return fallback_colors
        
    # If even that fails (unlikely), return random with K=20
    return [random.randint(0, 19) for _ in range(120)]


def is_valid(colors):
    # Check if every K4 has >= 5 distinct colors
    for k4 in k4s:
        distinct = set()
        for e in k4:
            distinct.add(colors[e])
        if len(distinct) < 5:
            return False
    return True


def run_sa(K, end_time):
    """
    Runs Simulated Annealing to find a coloring with K colors.
    Returns the coloring if penalty 0 is reached, else None.
    """
    # Initial random assignment
    colors = [random.randint(0, K-1) for _ in range(120)]
    
    # Data structures for fast penalty calculation
    # k4_freqs[k4_idx] = {color: count}
    k4_freqs = [{} for _ in range(len(k4s))]
    k4_distinct = [0] * len(k4s)
    penalty = 0 # Number of K4s with < 5 distinct colors
    
    for k4_idx, k4 in enumerate(k4s):
        freq = k4_freqs[k4_idx]
        for e in k4:
            c = colors[e]
            freq[c] = freq.get(c, 0) + 1
        k4_distinct[k4_idx] = len(freq)
        if k4_distinct[k4_idx] < 5:
            penalty += 1
            
    if penalty == 0:
        return colors[:]

    best_colors = colors[:]
    best_penalty = penalty
    
    T = 1.0
    T_min = 0.01
    alpha = 0.995 # Faster cooling
    step = 0
    
    while time.time() < end_time:
        # Select an edge to change
        # Strategy: 50% chance pick random edge, 50% chance pick edge from a bad K4
        if random.random() < 0.5:
            e_idx = random.randint(0, 119)
        else:
            # Pick a random bad K4, then a random edge in it
            bad_k4s = [i for i in range(len(k4s)) if k4_distinct[i] < 5]
            if not bad_k4s:
                break # Should not happen if penalty > 0
            bad_k4_idx = random.choice(bad_k4s)
            e_idx = random.choice(k4s[bad_k4_idx])
            
        old_color = colors[e_idx]
        
        # Choose new color
        # Try a few random colors? Or just one?
        # Just one is standard SA.
        new_color = random.randint(0, K-1)
        if new_color == old_color:
            continue
            
        affected_k4s = edge_to_k4s[e_idx]
        
        delta = 0
        revert_info = []
        
        for k4_idx in affected_k4s:
            freq = k4_freqs[k4_idx]
            old_d = k4_distinct[k4_idx]
            old_contrib = 1 if old_d < 5 else 0
            
            old_count = freq.get(old_color, 0)
            new_count = freq.get(new_color, 0)
            
            # Update freqs
            if old_count > 0:
                freq[old_color] = old_count - 1
                if old_count == 1:
                    # Color effectively removed from support
                    pass
            # else old_count is 0 (shouldn't happen for old_color)
            
            freq[new_color] = new_count + 1
            
            # Update distinct count
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
                if best_penalty == 0:
                    return best_colors
        else:
            # Revert
            for k4_idx, old_d, old_count, new_count in revert_info:
                freq = k4_freqs[k4_idx]
                # Revert old_color
                freq[old_color] = old_count
                # Revert new_color
                if new_count == 0:
                    # It was not present, we added it. Remove it.
                    if freq[new_color] == 1:
                        del freq[new_color]
                    else:
                        freq[new_color] -= 1
                else:
                    freq[new_color] = new_count
                    
                k4_distinct[k4_idx] = old_d
                
        T *= alpha
        if T < T_min:
            T = T_min
            
        # Restart logic?
        # If we are stuck (no improvement for many steps), restart temperature?
        # Or just let it cool.
        
    return None
