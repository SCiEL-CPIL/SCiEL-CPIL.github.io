# EVOLVE-BLOCK-START
import time
import random
import sys

def _mul(a, b):
    """Multiply in GF(16) with primitive polynomial 0b10011."""
    r = 0
    for i in range(4):
        if b >> i & 1:
            r ^= a << i
    for i in range(7, 3, -1):
        if r >> i & 1:
            r ^= 0b10011 << (i - 4)
    return r

def _build_seed_32():
    """Greenwood-Gleason colouring of K32 (two copies of K16 with colour 3 between)."""
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = _mul(x, 2)
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                colors.append(log[(i % 16) ^ (j % 16)] % 3)
            else:
                colors.append(3)
    return colors

def _color_index(n, i, j):
    """Index in the flat colors list for pair (i,j), i<j, n vertices."""
    return i * (i - 1) // 2 + (j - i - 1)

def _count_mono_triangles(colors, n):
    """Count monochromatic triangles."""
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            cij = colors[_color_index(n, i, j)]
            for k in range(j + 1, n):
                if colors[_color_index(n, i, k)] == cij and colors[_color_index(n, j, k)] == cij:
                    count += 1
    return count

def _find_mono_triangles(colors, n, limit=1000):
    """Find monochromatic triangles. Returns list of (i,j,k)."""
    tri = []
    for i in range(n):
        for j in range(i + 1, n):
            cij = colors[_color_index(n, i, j)]
            for k in range(j + 1, n):
                if len(tri) >= limit:
                    return tri
                if colors[_color_index(n, i, k)] == cij and colors[_color_index(n, j, k)] == cij:
                    tri.append((i, j, k))
    return tri

def _local_search(colors, n, deadline, rng):
    """
    Local search to minimize monochromatic triangles.
    Returns (colors, triangle_count).
    """
    cur_colors = colors[:]
    cur_count = _count_mono_triangles(cur_colors, n)
    best_colors = cur_colors[:]
    best_count = cur_count
    
    # Simulated Annealing parameters
    temp = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        if cur_count == 0:
            break
            
        # Pick a random monochromatic triangle
        found = False
        for _ in range(100): # Try 100 times to find a mono triangle
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            k = rng.randint(0, n - 1)
            if i == j or j == k or i == k:
                continue
            a, b, c = sorted([i, j, k])
            c_ab = cur_colors[_color_index(n, a, b)]
            c_ac = cur_colors[_color_index(n, a, c)]
            c_bc = cur_colors[_color_index(n, b, c)]
            if c_ab == c_ac == c_bc:
                found = True
                break
        
        if not found:
            # If we can't find a mono triangle by random sampling, it might be 0 or very low.
            # Let's do a full count to be sure.
            cur_count = _count_mono_triangles(cur_colors, n)
            if cur_count == 0:
                break
            # If count > 0 but we didn't find one, keep trying or restart?
            # Let's just continue, random sampling should hit it eventually if count is high.
            # If count is low, full count is needed.
            # For safety, if we do many searches without finding one, do a full count.
            continue

        # We found a mono triangle (a,b,c) with color c_ab
        # Try to change color of one of the edges to break the triangle.
        # Changing (a,b) to c_ab' != c_ab.
        
        # Let's try changing (a,b)
        old_c = c_ab
        new_c = (old_c + 1) % 4
        
        # Estimate impact:
        # The triangle (a,b,c) is broken. (-1)
        # New triangles formed:
        # 1. (a,b,x) where x is such that c_ax == new_c and c_bx == new_c.
        # 2. (x,a,c) where x is such that c_xa == new_c and c_xc == c_ab (old color of ab? No, c_ac is fixed).
        #    Wait, c_ac is fixed. For (x,a,c) to be mono with new_c, we need c_xa == new_c, c_ac == new_c, c_xc == new_c.
        #    But c_ac is old_c. So if new_c != old_c, (x,a,c) cannot be mono with new_c on (x,a) unless c_ac is also new_c.
        #    So triangles involving the changed edge (a,b) with new color new_c:
        #    - (a,b,x): need c_ax == new_c and c_bx == new_c.
        #    - (x,a,c): need c_xa == new_c and c_ac == new_c and c_xc == new_c. But c_ac=old_c. So if new_c != old_c, impossible.
        #    - (x,b,c): need c_xb == new_c and c_bc == new_c and c_xc == new_c. But c_bc=old_c. So if new_c != old_c, impossible.
        
        # So only new triangles are of the form (a,b,x) where c_ax == new_c and c_bx == new_c.
        # And we lose the triangle (a,b,c).
        # Also, we might break other triangles that were using (a,b) with old_c?
        # Triangles (a,b,y) where c_ay == old_c and c_by == old_c. These are now broken.
        
        # Let's calculate delta precisely.
        # Old triangles broken:
        # 1. (a,b,c) itself.
        # 2. Any (a,b,y) where y != c, c_ay == old_c, c_by == old_c.
        # New triangles created:
        # 1. Any (a,b,x) where x != c, c_ax == new_c, c_bx == new_c.
        
        # Note: x=c is excluded from "new" because c_ac=old_c != new_c (assuming new_c != old_c).
        # So we just count vertices y (y != c) such that c_ay == old_c and c_by == old_c. (Broken)
        # And vertices x (x != c) such that c_ax == new_c and c_bx == new_c. (Created)
        
        broken = 0
        created = 0
        
        for v in range(n):
            if v == a or v == b or v == c:
                continue
            c_av = cur_colors[_color_index(n, a, v)] if a < v else cur_colors[_color_index(n, v, a)]
            c_bv = cur_colors[_color_index(n, b, v)] if b < v else cur_colors[_color_index(n, v, b)]
            
            if c_av == old_c and c_bv == old_c:
                broken += 1
            if c_av == new_c and c_bv == new_c:
                created += 1
        
        delta = created - (broken + 1) # +1 for the triangle (a,b,c) itself
        
        # Accept if delta < 0 or (delta == 0 and rng.random() < 0.5) or (delta > 0 and rng.random() < temp)
        accept = False
        if delta < 0:
            accept = True
        elif delta == 0:
            if rng.random() < 0.5:
                accept = True
        else:
            if rng.random() < temp:
                accept = True
        
        if accept:
            idx = _color_index(n, a, b)
            cur_colors[idx] = new_c
            cur_count += delta
            temp *= cooling
            if cur_count < best_count:
                best_count = cur_count
                best_colors = cur_colors[:]
                
    return best_colors, best_count

def _greedy_construct(n, deadline, rng):
    """
    Greedy randomized construction: iteratively add vertices,
    selecting edge colors to minimize immediate triangle creation.
    Returns (colors, triangle_count) or None if deadline exceeded.
    """
    colors = [0] * (n * (n - 1) // 2)
    
    # For each new vertex v (from 1 to n-1), assign colors to edges (u, v) for u in 0..v-1.
    # We want to minimize the number of monochromatic triangles formed.
    # A triangle (u, w, v) is mono if color(u,w) == color(u,v) == color(w,v).
    # When assigning color(u,v), we look at all w < u to see if it creates a triangle.
    
    # Actually, let's do a greedy assignment with some randomization.
    # For each new vertex v, for each edge (u, v), try all 4 colors and pick the one
    # that minimizes the number of new mono triangles formed.
    
    # To be faster, we can precompute for each color c, how many w < u have color(u,w) == c.
    # Then, for a candidate color c for edge (u,v), the number of new triangles is
    # the number of w < u such that color(u,w) == c AND color(w,v) == c.
    # But color(w,v) is not yet assigned (since w < u < v, but we're assigning in order u=0..v-1).
    # Wait, if we assign edges (0,v), (1,v), ..., (v-1,v) in order, then when assigning (u,v),
    # colors (w,v) for w < u are already assigned.
    
    # So for each u from 0 to v-1:
    #   For each candidate color c in 0..3:
    #     count_new = 0
    #     for w in 0..u-1:
    #       if color(u,w) == c and color(w,v) == c:
    #         count_new += 1
    #     # Also, we don't form triangles with w > u because those edges aren't assigned yet.
    #   Pick the c with minimum count_new (randomly break ties).
    
    # This is O(n^3) total, which is fine for n=50.
    
    for v in range(1, n):
        for u in range(v):
            best_c = 0
            best_count = float('inf')
            # For each candidate color, count new triangles
            for c in range(4):
                cnt = 0
                for w in range(u):
                    # color(u,w) is at index _color_index(n, w, u) since w < u
                    # color(w,v) is at index _color_index(n, w, v) since w < v
                    if colors[_color_index(n, w, u)] == c and colors[_color_index(n, w, v)] == c:
                        cnt += 1
                if cnt < best_count:
                    best_count = cnt
                    best_c = c
                elif cnt == best_count:
                    # Random tie-breaking
                    if rng.random() < 0.5:
                        best_c = c
            colors[_color_index(n, u, v)] = best_c
    
    # Count total mono triangles
    tri_count = _count_mono_triangles(colors, n)
    
    return colors, tri_count

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Keep margin before 150s kill
    
    rng = random.Random(42)
    
    # Strategy: Try to find a valid coloring for n=50, 49, ... down to 32.
    # Start with n=50.
    
    best_result = {'n': 32, 'colors': _build_seed_32()}
    
    # We can also try to grow from 32, but local search from random n=50 is often faster if it works.
    # Let's try n from 50 down to 33.
    for n in range(50, 32, -1):
        if time.time() > deadline:
            break
            
        # Try multiple random restarts for this n
        found = False
        attempts = 0
        max_attempts = 10 # Limit attempts per n to ensure we try smaller n if stuck
        
        while not found and attempts < max_attempts and time.time() < deadline:
            attempts += 1
            # Try greedy construction first
            if attempts == 1:
                colors, tri_count = _greedy_construct(n, deadline, rng)
            else:
                # Random initialization
                colors = [rng.randint(0, 3) for _ in range(n * (n - 1) // 2)]
            
            # Run local search
            res_colors, res_count = _local_search(colors, n, deadline, rng)
            
            if res_count == 0:
                # Valid coloring found!
                if n > best_result['n']:
                    best_result = {'n': n, 'colors': res_colors}
                found = True
                break
            else:
                # If we are close, maybe try again?
                # If res_count is very low, it might be worth another try with a different seed?
                # But we are limited by time.
                pass
        
        # If we found a valid n, we can stop?
        # The task says "Maximize n". So if we found 50, we are done.
        # If we found 49, we should try 50 again?
        # The loop goes 50 down to 33.
        # If we found 50, we break.
        if best_result['n'] == n:
            break
            
    # If we didn't find anything better than 32, return 32.
    # We should also verify the best result.
    if best_result['n'] >= 3:
        tri = _find_mono_triangles(best_result['colors'], best_result['n'], limit=1)
        if tri:
            # Fallback
            best_result = {'n': 32, 'colors': _build_seed_32()}
            
    return best_result
# EVOLVE-BLOCK-END
