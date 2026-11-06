import random
import time

def construct():
    start_time = time.time()
    deadline = 115.0  # Leave 5s margin
    
    # We will try to find a coloring for n=50
    n = 50
    num_edges = n * (n - 1) // 2
    
    # Helper to get edge index
    def edge_idx(i, j):
        if i > j: i, j = j, i
        return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)
        
    # This indexing is complex. Let's use a list of pairs.
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    idx_map = {e: i for i, e in enumerate(edges)}
    
    # Initialize colors
    # Use a heuristic: 
    # Color 0: Edges where (i+j) % 4 == 0 ? 
    # Better: Start with a random assignment or a structured one.
    # Let's use the Clebsch graph logic if possible, but for 50 it's hard.
    # Let's just use random initialization with a bias or pure random.
    
    best_colors = None
    best_tris = float('inf')
    
    # Precompute triangles for each edge?
    # An edge (u, v) is in a triangle with (u, w) and (v, w) if colors match.
    # For each edge e, we need to know the other edges that form a triangle with it.
    # For edge (u, v), the third vertex w can be any other vertex.
    # The triangle is (u, v, w). The other edges are (u, w) and (v, w).
    # So for each edge e=(u,v), the set of "conflict pairs" is {(idx(u,w), idx(v,w)) for w != u,v}.
    
    conflict_pairs = []
    for (u, v) in edges:
        pairs = []
        for w in range(n):
            if w != u and w != v:
                e1 = idx_map[(min(u, w), max(u, w))]
                e2 = idx_map[(min(v, w), max(v, w))]
                pairs.append((e1, e2))
        conflict_pairs.append(pairs)
        
    # Function to count triangles
    def count_triangles(colors):
        count = 0
        for e_idx, (u, v) in enumerate(edges):
            c = colors[e_idx]
            for e1, e2 in conflict_pairs[e_idx]:
                if colors[e1] == c and colors[e2] == c:
                    count += 1
        return count // 3 # Each triangle counted 3 times
        
    # Local Search
    def local_search(n, max_iter=100000):
        # Initialize
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        # Heuristic init: Try to make it structured
        # For now, random is fine for SA.
        
        tris = count_triangles(colors)
        best_local = colors[:]
        best_local_tris = tris
        
        # Simulated Annealing
        temp = 100.0
        cooling = 0.9999
        min_temp = 0.1
        
        for i in range(max_iter):
            if time.time() > deadline:
                break
            
            # Pick a random edge
            e = random.randint(0, num_edges - 1)
            old_c = colors[e]
            new_c = random.randint(0, 3)
            if new_c == old_c:
                continue
                
            colors[e] = new_c
            
            # Calculate delta in triangles
            # Only triangles involving edge e are affected.
            # A triangle involving e=(u,v) and w has edges (u,w), (v,w).
            # If old_c == colors[u,w] == colors[v,w], we had a triangle.
            # If new_c == colors[u,w] == colors[v,w], we have a triangle.
            
            delta = 0
            u, v = edges[e]
            for w in range(n):
                if w != u and w != v:
                    e1 = idx_map[(min(u, w), max(u, w))]
                    e2 = idx_map[(min(v, w), max(v, w))]
                    
                    # Check old state
                    if old_c == colors[e1] == colors[e2]:
                        delta -= 1
                    # Check new state
                    if new_c == colors[e1] == colors[e2]:
                        delta += 1
            
            new_tris = tris + delta
            
            if new_tris < tris or random.random() < pow(2.71828, -new_tris / temp):
                tris = new_tris
                if tris < best_local_tris:
                    best_local_tris = tris
                    best_local = colors[:]
            else:
                colors[e] = old_c
                
            temp *= cooling
            if temp < min_temp:
                break
                
        return best_local, best_local_tris

    # Try to find solution for n=50
    found = False
    for attempt in range(20):
        if time.time() > deadline:
            break
        colors, tris = local_search(n, max_iter=50000)
        if tris == 0:
            best_colors = colors
            found = True
            break
        if tris < best_tris:
            best_tris = tris
            best_colors = colors
            
    if found:
        return {'n': n, 'colors': best_colors}
        
    # If n=50 not found, try n=49? Or just return best found for n=50 if it's valid? 
    # No, must be valid.
    # If we didn't find 0 triangles for n=50, we should fall back to a known valid solution.
    # The seed program has n=32.
    # We can return the n=32 solution if we don't find n=50.
    
    # Fallback to n=32 (Greenwood-Gleason)
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    colors32 = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors32.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    return {'n': 32, 'colors': colors32}
