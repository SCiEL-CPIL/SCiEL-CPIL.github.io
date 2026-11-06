import time
import sys
import random

def construct():
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer for overhead and return

    # Fallback: The seed program's 32-vertex coloring
    def get_fallback():
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
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
        return 32, colors

    best_n, best_colors = get_fallback()

    # Try to find a 50-vertex coloring using Simulated Annealing
    # This is a structural change from Backtracking to Local Search
    
    n = 50
    # Edge list
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    # Map edge to index
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx

    # Precompute for each edge, the other edges that form a triangle with it
    # For edge (i, j), the third vertex k can be any other vertex.
    # Triangle is (i, j, k). Edges: (i,j), (i,k), (j,k).
    # We need to check if all 3 have the same color.
    
    # For fast update, for each edge e=(i,j), we store the list of (k, edge_ik, edge_jk)
    # Actually, simpler: for each edge e, store the list of pairs (e1, e2) such that {e, e1, e2} is a triangle.
    # If color(e) == color(e1) == color(e2), it's a mono-triangle.
    
    # This precomputation is O(n^3) which is 50^3 = 125k, fine.
    triangle_pairs = [[] for _ in range(num_edges)]
    for i in range(n):
        for j in range(i + 1, n):
            e_ij = edge_idx[(i, j)]
            for k in range(n):
                if k == i or k == j:
                    continue
                e_ik = edge_idx[(min(i, k), max(i, k))]
                e_jk = edge_idx[(min(j, k), max(j, k))]
                # Add pair to triangle_pairs[e_ij]
                # To avoid duplicates in the list? No, each triangle has 3 edges.
                # We just need to count mono-triangles.
                # If we change color of e_ij, we only affect triangles containing e_ij.
                # For each such triangle (e_ij, e_ik, e_jk), if they were mono, and we change e_ij, 
                # the triangle is no longer mono (unless e_ik==e_jk==new_color? No, e_ij changed).
                # Actually, if e_ik == e_jk == c, and e_ij was c, it was 1 mono.
                # If we change e_ij to d != c, it becomes 0 mono.
                # If we change e_ij to d == c, it stays 1 mono.
                # If e_ik == e_jk == d, and e_ij was c != d, it was 0 mono.
                # If we change e_ij to d, it becomes 1 mono.
                
                # So, for each edge e, we can store for each color c, the number of "potential" mono-triangles
                # that would be formed if e had color c, given the current colors of the other edges.
                # But this depends on the other edges' colors, which change.
                
                # Simpler: Just recompute the count for affected triangles.
                # Number of triangles containing an edge is n-2 = 48.
                # So changing one edge affects 48 triangles.
                # Recalculating the total mono-triangle count from scratch is O(n^3) = 125k.
                # With 115 seconds, we can do many iterations.
                
                # Let's just store the list of (e1, e2) for each e.
                triangle_pairs[e_ij].append((e_ik, e_jk))

    def count_mono_triangles(colors):
        # colors is a list of length num_edges
        count = 0
        # Iterate over all triangles
        # To speed up, we can iterate over edges and check, but that's O(E * (n-2)).
        # E ~ 1250, n-2 ~ 48. 1250 * 48 = 60k operations.
        # 115s / 60k ops per eval ~ 1.9M evals.
        # Python is slow. ~1M ops/s? So ~1000 evals/s?
        # 115s * 1000 = 115k evals.
        # This might be too slow for random restarts.
        
        # Optimization: Incremental update.
        # Keep a global count.
        # When changing edge e from c_old to c_new:
        # For each (e1, e2) in triangle_pairs[e]:
        #   c1 = colors[e1], c2 = colors[e2]
        #   If c_old == c1 == c2: count -= 1
        #   If c_new == c1 == c2: count += 1
        #   (If c_old == c1 == c2 AND c_new == c1 == c2, net 0, which is correct)
        
        pass

    # Initialize random colors
    colors = [random.randint(0, 3) for _ in range(num_edges)]
    
    # Calculate initial count
    mono_count = 0
    for e in range(num_edges):
        for e1, e2 in triangle_pairs[e]:
            if colors[e] == colors[e1] == colors[e2]:
                mono_count += 1
    # Each triangle is counted 3 times (once for each edge).
    # So actual_mono_triangles = mono_count // 3
    
    actual_mono = mono_count // 3
    
    if actual_mono == 0:
        return {'n': n, 'colors': colors}

    # Simulated Annealing
    T = 1.0
    cooling = 0.999
    min_T = 0.001
    
    # Deadline guard
    deadline = start_time + time_limit
    
    while time.time() < deadline and T > min_T:
        # Pick random edge
        e = random.randint(0, num_edges - 1)
        old_color = colors[e]
        new_color = random.randint(0, 3)
        if old_color == new_color:
            continue
            
        # Calculate delta
        delta = 0
        for e1, e2 in triangle_pairs[e]:
            c1 = colors[e1]
            c2 = colors[e2]
            if old_color == c1 == c2:
                delta -= 1
            if new_color == c1 == c2:
                delta += 1
        
        # delta is change in mono_count (which is 3 * actual_mono)
        # We want to minimize mono_count.
        
        if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
            colors[e] = new_color
            mono_count += delta
            actual_mono = mono_count // 3
            if actual_mono == 0:
                return {'n': n, 'colors': colors}
        
        T *= cooling

    # If SA failed, return fallback
    return {'n': best_n, 'colors': best_colors}
