import time
import random
import sys

def construct():
    # Fallback to the known n=32 solution if we fail
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
    colors_32 = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors_32.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    fallback = {'n': 32, 'colors': colors_32}
    
    n = 50
    num_edges = n * (n - 1) // 2
    
    # Map (i, j) to index in colors array
    # i < j
    # idx = i * n - i * (i+1) // 2 + (j - i - 1) ?
    # Simpler: precompute index map
    idx_map = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            idx_map[(i, j)] = idx
            idx += 1
            
    # Precompute triangles for each edge
    # edge (u, v) is in triangles (u, v, w) for w != u, v
    edge_triangles = [[] for _ in range(num_edges)]
    for u in range(n):
        for v in range(u + 1, n):
            idx_uv = idx_map[(u, v)]
            for w in range(n):
                if w == u or w == v:
                    continue
                # Triangle (u, v, w)
                # Need indices of (u, w) and (v, w)
                if u < w:
                    idx_uw = idx_map[(u, w)]
                else:
                    idx_uw = idx_map[(w, u)]
                if v < w:
                    idx_vw = idx_map[(v, w)]
                else:
                    idx_vw = idx_map[(w, v)]
                edge_triangles[idx_uv].append((idx_uw, idx_vw))
                
    # Greedy Initialization
    colors = [0] * num_edges
    # We build the graph vertex by vertex
    # When adding vertex v, we set colors for edges (v, u) for u < v
    # We want to minimize mono-triangles involving v
    # A mono-triangle (u, v, w) with u, w < v is formed if colors[(u,v)] == colors[(v,w)] == colors[(u,w)]
    
    # Initialize colors for edges among 0..n-1
    # Actually, greedy:
    # Start with vertex 0.
    # Add vertex 1. Edge (0,1). No triangles yet.
    # Add vertex 2. Edges (0,2), (1,2). Triangle (0,1,2).
    #   Choose colors for (0,2) and (1,2) to avoid mono triangle with (0,1).
    
    # Let's just do random init and SA, it's simpler and often works if T is high enough.
    # But greedy is better.
    
    # Let's try random init first for simplicity, then SA.
    random.seed(42)
    colors = [random.randint(0, 3) for _ in range(num_edges)]
    
    # Count mono triangles
    def count_mono_triangles(cols):
        counts = [0, 0, 0, 0]
        for u in range(n):
            for v in range(u + 1, n):
                c_uv = cols[idx_map[(u, v)]]
                for w in range(v + 1, n):
                    c_vw = cols[idx_map[(v, w)]]
                    if c_uv != c_vw:
                        continue
                    if u < w:
                        c_uw = cols[idx_map[(u, w)]]
                    else:
                        c_uw = cols[idx_map[(w, u)]]
                    if c_uw == c_uv:
                        counts[c_uv] += 1
        return sum(counts)
        
    # This O(N^3) count is too slow for every step in SA.
    # We need incremental update.
    
    # State:
    # colors: list of ints
    # mono_count: total number of mono triangles
    # color_counts: [c0, c1, c2, c3] counts of mono triangles per color
    
    def get_initial_state(cols):
        mono_count = 0
        color_counts = [0, 0, 0, 0]
        # We can just run the O(N^3) once
        for u in range(n):
            for v in range(u + 1, n):
                c_uv = cols[idx_map[(u, v)]]
                for w in range(v + 1, n):
                    c_vw = cols[idx_map[(v, w)]]
                    if c_uv != c_vw:
                        continue
                    if u < w:
                        c_uw = cols[idx_map[(u, w)]]
                    else:
                        c_uw = cols[idx_map[(w, u)]]
                    if c_uw == c_uv:
                        mono_count += 1
                        color_counts[c_uv] += 1
        return mono_count, color_counts

    current_colors = colors[:]
    current_mono, current_cc = get_initial_state(current_colors)
    best_colors = current_colors[:]
    best_mono = current_mono
    
    start_time = time.time()
    timeout = 110.0 # Leave margin
    
    T = 1.0
    T_min = 0.001
    cooling = 0.999
    
    # SA Loop
    while time.time() - start_time < timeout:
        # Pick random edge
        e_idx = random.randint(0, num_edges - 1)
        old_c = current_colors[e_idx]
        new_c = random.choice([c for c in range(4) if c != old_c])
        
        # Calculate delta
        delta = 0
        # Triangles containing this edge
        for idx_uw, idx_vw in edge_triangles[e_idx]:
            c_uw = current_colors[idx_uw]
            c_vw = current_colors[idx_vw]
            
            # Check if this triangle was mono in old_c
            # It was mono in old_c if c_uw == old_c and c_vw == old_c
            if c_uw == old_c and c_vw == old_c:
                delta -= 1 # Remove one mono triangle of old_c
                
            # Check if this triangle becomes mono in new_c
            # It becomes mono in new_c if c_uw == new_c and c_vw == new_c
            if c_uw == new_c and c_vw == new_c:
                delta += 1 # Add one mono triangle of new_c
                
        new_mono = current_mono + delta
        
        if new_mono < current_mono or random.random() < pow(2.718281828, -delta / T):
            # Accept
            current_colors[e_idx] = new_c
            current_mono = new_mono
            
            if current_mono < best_mono:
                best_mono = current_mono
                best_colors = current_colors[:]
                
            if best_mono == 0:
                break
        T *= cooling
        
    if best_mono == 0:
        return {'n': n, 'colors': best_colors}
    else:
        return fallback

# EVOLVE-BLOCK-END
