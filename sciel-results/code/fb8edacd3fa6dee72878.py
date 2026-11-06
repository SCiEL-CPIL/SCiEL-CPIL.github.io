import time
import random

def construct():
    """
    Attempts to find a 4-coloring of K_n with no monochromatic triangles.
    Returns the largest n found (up to 51) or falls back to 32.
    """
    
    def idx(i, j, n):
        """Calculate index in the flat colors list for edge (i, j) with i < j."""
        # Number of edges before row i is sum_{k=0}^{i-1} (n-1-k)
        # Edges in row i before j is (j - i - 1)
        # Total = sum_{k=0}^{i-1} (n-1-k) + (j-i-1)
        # Sum is i*n - i*(i+1)/2 - i + j - i - 1 ? 
        # Let's use a simpler mapping or precompute offsets.
        # offset[i] = sum_{k=0}^{i-1} (n-1-k)
        # index = offset[i] + (j - i - 1)
        return offsets[i] + (j - i - 1)

    def count_triangles(colors, n):
        """Count total monochromatic triangles."""
        # For each vertex w, and each pair (u, v) of neighbors?
        # Easier: For each edge (u, v) with color c, count common neighbors w 
        # such that color(u, w) == c and color(v, w) == c.
        # Sum over all edges. Divide by 3 (each triangle counted 3 times, once per edge).
        
        # Optimization: Precompute adjacency lists for each color?
        # Or just iterate.
        # n=50, edges=1225.
        # For each edge, check 48 other vertices. 1225 * 48 ~ 60k ops. Fast.
        
        total = 0
        for u in range(n):
            for v in range(u + 1, n):
                c = colors[idx(u, v, n)]
                count = 0
                for w in range(n):
                    if w == u or w == v:
                        continue
                    # Check if (u, w) and (v, w) have color c
                    i_uw = idx(min(u, w), max(u, w), n)
                    i_vw = idx(min(v, w), max(v, w), n)
                    if colors[i_uw] == c and colors[i_vw] == c:
                        count += 1
                total += count
        return total // 3

    def delta_triangles(colors, n, u, v, old_c, new_c):
        """
        Calculate change in number of monochromatic triangles when 
        changing edge (u, v) from old_c to new_c.
        """
        if old_c == new_c:
            return 0
        
        delta = 0
        # Triangles containing (u, v)
        for w in range(n):
            if w == u or w == v:
                continue
            
            i_uw = idx(min(u, w), max(u, w), n)
            i_vw = idx(min(v, w), max(v, w), n)
            c_uw = colors[i_uw]
            c_vw = colors[i_vw]
            
            # If (u, w) and (v, w) were both old_c, they formed a triangle with (u, v).
            # Now (u, v) is new_c, so that triangle is broken.
            if c_uw == old_c and c_vw == old_c:
                delta -= 1
            
            # If (u, w) and (v, w) are both new_c, they form a new triangle with (u, v).
            if c_uw == new_c and c_vw == new_c:
                delta += 1
                
        return delta

    def search(n, time_limit):
        """
        Local search to find a valid coloring for K_n.
        Returns (success, colors)
        """
        if n < 2:
            return True, []
            
        num_edges = n * (n - 1) // 2
        # Precompute offsets for idx
        nonlocal offsets
        offsets = [0] * n
        offset = 0
        for i in range(n):
            offsets[i] = offset
            offset += n - 1 - i
            
        # Initialize with random colors
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        
        current_triangles = count_triangles(colors, n)
        best_colors = colors[:]
        best_triangles = current_triangles
        
        start_time = time.time()
        no_improve_count = 0
        max_no_improve = 5000
        
        while time.time() - start_time < time_limit:
            # Pick a random edge
            edge_idx = random.randint(0, num_edges - 1)
            
            # Find u, v from edge_idx
            # Inverse of idx:
            # Find i such that offsets[i] <= edge_idx < offsets[i+1]
            # Simple linear search or binary search. n is small, linear is fine.
            u = 0
            for i in range(n):
                if offsets[i] <= edge_idx < (offsets[i+1] if i+1 < n else num_edges + 1):
                    u = i
                    break
            v = u + 1 + (edge_idx - offsets[u])
            
            old_c = colors[edge_idx]
            
            # Try all 4 colors
            best_delta = 0
            best_new_c = old_c
            for new_c in range(4):
                if new_c == old_c:
                    continue
                d = delta_triangles(colors, n, u, v, old_c, new_c)
                if d < best_delta:
                    best_delta = d
                    best_new_c = new_c
            
            # Accept the best move (Greedily)
            if best_new_c != old_c:
                colors[edge_idx] = best_new_c
                current_triangles += best_delta
                
                if current_triangles == 0:
                    return True, colors
                
                if current_triangles < best_triangles:
                    best_triangles = current_triangles
                    best_colors = colors[:]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    # If stuck, maybe randomize?
                    if no_improve_count > max_no_improve:
                        # Perturbation: flip a random edge
                        rand_idx = random.randint(0, num_edges - 1)
                        colors[rand_idx] = random.randint(0, 3)
                        current_triangles = count_triangles(colors, n)
                        no_improve_count = 0

        return False, best_colors

    # Deadline guard
    global_start = time.time()
    total_limit = 140.0 # 150s limit, keep margin
    
    # Try n=51 first (new record)
    # Allocate 60s for 51
    time_left = total_limit - (time.time() - global_start)
    if time_left > 60:
        success, colors_51 = search(51, 60)
        if success:
            return {'n': 51, 'colors': colors_51}
    
    # Try n=50 (known record)
    # Allocate remaining time, min 30s
    time_left = total_limit - (time.time() - global_start)
    alloc_50 = min(time_left, 80)
    if alloc_50 > 5:
        success, colors_50 = search(50, alloc_50)
        if success:
            return {'n': 50, 'colors': colors_50}
    
    # Fallback to seed n=32
    # Re-implement seed logic for fallback
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
    return {'n': 32, 'colors': colors}

# Global variable for offsets, initialized in search
offsets = []
