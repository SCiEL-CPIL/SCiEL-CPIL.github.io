import time
import random
import sys

def _count_mono_triangles_fast(colors, n):
    """
    Count monochromatic triangles efficiently.
    colors: list of length n*(n-1)//2
    """
    # Build adjacency sets for each color
    adj = [set() for _ in range(4)]
    for i in range(n):
        for j in range(i + 1, n):
            idx = i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
            c = colors[idx]
            adj[c].add((i, j))
            adj[c].add((j, i)) # For faster lookup? No, just use sets of neighbors
    
    # Actually, for triangle counting, we need:
    # For each color c, for each edge (u,v) in color c, count common neighbors w in color c.
    # Let's use a different structure:
    # adj[c][u] = set of v such that (u,v) is color c
    
    adj = [ [set() for _ in range(n)] for _ in range(4) ]
    for i in range(n):
        for j in range(i + 1, n):
            idx = i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
            c = colors[idx]
            adj[c][i].add(j)
            adj[c][j].add(i)
            
    count = 0
    for c in range(4):
        for i in range(n):
            for j in adj[c][i]:
                if j > i: # Count each edge once? No, triangle (i,j,k) is counted 3 times if we sum degrees of common neighbors?
                    # Triangle (i,j,k) exists if k is in adj[c][i] and k is in adj[c][j]
                    # Sum over edges (i,j) of |adj[c][i] & adj[c][j]| counts each triangle 3 times (once for each edge).
                    common = adj[c][i].intersection(adj[c][j])
                    count += len(common)
    return count // 3

def _local_search_fast(colors, n, time_limit, start_time, seed=0):
    """
    Local search with fast triangle delta calculation.
    """
    rng = random.Random(seed)
    
    # Precompute indices for fast access
    # idx(i, j) for i < j
    def idx(i, j):
        if i > j: i, j = j, i
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
        
    # Build adjacency structures
    adj = [ [set() for _ in range(n)] for _ in range(4) ]
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx(i, j)]
            adj[c][i].add(j)
            adj[c][j].add(i)
            
    # Initial triangle count
    tri_count = 0
    for c in range(4):
        for i in range(n):
            for j in adj[c][i]:
                if j > i:
                    tri_count += len(adj[c][i].intersection(adj[c][j]))
    tri_count //= 3
    
    best_colors = colors[:]
    best_tri = tri_count
    
    if tri_count == 0:
        return colors, 0
        
    # Simulated Annealing / Hill Climbing
    temp = 10.0
    it = 0
    while time.time() - start_time < time_limit:
        # Pick random edge
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j: continue
        if i > j: i, j = j, i
        
        e_idx = idx(i, j)
        old_c = colors[e_idx]
        new_c = rng.randint(0, 3)
        if new_c == old_c: continue
        
        # Calculate delta
        # Remove (i,j) from old_c, add to new_c
        # Triangles in old_c containing (i,j): common neighbors of i and j in old_c
        # But (i,j) is an edge in old_c.
        # Common neighbors k such that (i,k) in old_c and (j,k) in old_c.
        # Note: k cannot be i or j.
        
        common_old = adj[old_c][i].intersection(adj[old_c][j])
        # Remove i and j from common if they are there? 
        # adj[old_c][i] contains j. adj[old_c][j] contains i.
        # Intersection will contain... well, i is not in adj[old_c][i] (no self loops).
        # j is in adj[old_c][i]. i is in adj[old_c][j].
        # So intersection does not contain i or j? 
        # adj[old_c][i] has j. adj[old_c][j] has i.
        # Is j in adj[old_c][j]? No.
        # Is i in adj[old_c][i]? No.
        # So intersection is correct.
        
        delta_old = len(common_old)
        
        # Add (i,j) to new_c
        # Temporarily add to adj to calculate new common?
        # We need common neighbors in new_c AFTER adding (i,j).
        # (i,j) is an edge.
        # Common neighbors k such that (i,k) in new_c and (j,k) in new_c.
        # Note: If we add (i,j) to new_c, it doesn't affect the neighbors of i or j in terms of other vertices k.
        # So common_new is just adj[new_c][i].intersection(adj[new_c][j])
        
        common_new = adj[new_c][i].intersection(adj[new_c][j])
        delta_new = len(common_new)
        
        delta = delta_new - delta_old
        
        if delta <= 0 or (rng.random() < 1.0 / (1.0 + temp) and delta < 20):
            # Accept
            # Update adj
            adj[old_c][i].remove(j)
            adj[old_c][j].remove(i)
            adj[new_c][i].add(j)
            adj[new_c][j].add(i)
            
            colors[e_idx] = new_c
            tri_count += delta
            
            if tri_count < best_tri:
                best_tri = tri_count
                best_colors = colors[:]
                if best_tri == 0:
                    return best_colors, 0
            temp = max(0.1, temp * 0.999)
        else:
            # Reject
            pass
            
        it += 1
        if it % 1000 == 0:
            temp = max(0.1, temp * 0.9)

    return best_colors, best_tri

def _build_seed32():
    """Build the Greenwood-Gleason 32-vertex 4-coloring."""
    # Hardcoded or generated
    # Using GF(16) logic
    def mul_gf16(a, b):
        r = 0
        for i in range(4):
            if (b >> i) & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if (r >> i) & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log = {}
    x = 1
    for e in range(15):
        log[x] = e
        x = mul_gf16(x, 2)
        
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                colors.append(log[(i % 16) ^ (j % 16)] % 3)
            else:
                colors.append(3)
    return colors

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    # 1. Try to find a valid 50-vertex coloring
    best_n = 32
    best_colors = _build_seed32()
    
    # Verify seed
    if _count_mono_triangles_fast(best_colors, 32) != 0:
        # Should not happen
        pass
        
    # Try N=50
    for target_n in [50, 49, 48, 45, 40]:
        if time.time() - start_time > time_limit - 10:
            break
            
        found = False
        # Multiple restarts
        for seed in range(5):
            if time.time() - start_time > time_limit - 10:
                break
            remaining = time_limit - (time.time() - start_time)
            if remaining < 5: break
            
            # Random init
            rng = random.Random(seed)
            colors = [rng.randint(0, 3) for _ in range(target_n * (target_n - 1) // 2)]
            
            colors, tri = _local_search_fast(colors, target_n, remaining - 5, start_time, seed=seed)
            
            if tri == 0:
                best_n = target_n
                best_colors = colors[:]
                found = True
                break
        
        if found:
            break # Found valid coloring for this n, try higher? No, we go down.
            # Actually, if we found 50, we are done.
            # If we found 49, we might try 50 again? No, time is tight.
            
    # If we didn't find 50, we have best_n (likely 32 or higher if lucky)
    # Return best result
    return {'n': best_n, 'colors': best_colors}
