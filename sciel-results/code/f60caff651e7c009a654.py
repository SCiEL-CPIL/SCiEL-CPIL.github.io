# EVOLVE-BLOCK-START
import time
import random
import sys
import math
import numpy as np

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
    Uses a more efficient delta calculation.
    Returns (colors, triangle_count).
    """
    cur_colors = colors[:]
    cur_count = _count_mono_triangles(cur_colors, n)
    best_colors = cur_colors[:]
    best_count = cur_count
    
    # Precompute adjacency lists for faster delta? No, let's just be careful.
    
    # Simulated Annealing parameters
    temp = 5.0
    cooling = 0.9995
    min_temp = 0.1
    
    # Precompute indices for all pairs? No, just use the function.
    
    while time.time() < deadline:
        if cur_count == 0:
            break
            
        # Pick a random monochromatic triangle to break
        # We maintain a list of mono triangles? Or just sample?
        # Sampling is O(1) but might miss. 
        # Since n <= 62, total edges ~ 1900. 
        # We can afford to check all edges for mono triangles? 
        # If we pick a random edge (u,v) with color c, and find a w such that (u,w) and (v,w) are also c,
        # then (u,v,w) is a mono triangle.
        
        found = False
        # Try to find a mono triangle quickly
        for _ in range(50):
            u = rng.randint(0, n - 1)
            v = rng.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            c = cur_colors[_color_index(n, u, v)]
            # Find w
            for w in range(n):
                if w == u or w == v:
                    continue
                c_uw = cur_colors[_color_index(n, min(u,w), max(u,w))]
                c_vw = cur_colors[_color_index(n, min(v,w), max(v,w))]
                if c_uw == c and c_vw == c:
                    # Found mono triangle (u, v, w) with color c
                    found = True
                    break
            if found:
                break
        
        if not found:
            # Recalculate count to be sure
            cur_count = _count_mono_triangles(cur_colors, n)
            if cur_count == 0:
                break
            # If we couldn't find one by sampling, pick a random edge and flip?
            # Or just continue with random move
            u = rng.randint(0, n - 1)
            v = rng.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            old_c = cur_colors[_color_index(n, u, v)]
            new_c = (old_c + rng.randint(1, 3)) % 4
        else:
            # We have (u, v, w) mono with color c
            # Choose one of the three edges to change
            edges = [(u, v), (u, w), (v, w)]
            # Sort edges for consistent indexing
            edges = [(min(a,b), max(a,b)) for a,b in edges]
            idx = rng.randint(0, 2)
            u, v = edges[idx]
            old_c = cur_colors[_color_index(n, u, v)]
            new_c = (old_c + rng.randint(1, 3)) % 4
        
        # Calculate delta for changing edge (u, v) from old_c to new_c
        # A triangle (u, v, x) is mono with color c iff c_ux == c, c_vx == c, c_uv == c.
        # Changing c_uv affects only triangles containing (u, v).
        # Triangles broken: those where c_ux == old_c, c_vx == old_c (and c_uv was old_c).
        # Triangles created: those where c_ux == new_c, c_vx == new_c.
        
        broken = 0
        created = 0
        for x in range(n):
            if x == u or x == v:
                continue
            c_ux = cur_colors[_color_index(n, min(u,x), max(u,x))]
            c_vx = cur_colors[_color_index(n, min(v,x), max(v,x))]
            if c_ux == old_c and c_vx == old_c:
                broken += 1
            if c_ux == new_c and c_vx == new_c:
                created += 1
        
        delta = created - broken
        
        # Accept if delta < 0, or with probability exp(-delta/temp)
        accept = False
        if delta < 0:
            accept = True
        else:
            if delta == 0:
                if rng.random() < 0.3:
                    accept = True
            else:
                # Avoid overflow
                if delta * math.log(temp) > -700: # log(exp(-d/t)) = -d/t
                    prob = math.exp(-delta / temp)
                else:
                    prob = 0.0
                if rng.random() < prob:
                    accept = True
        
        if accept:
            idx = _color_index(n, u, v)
            cur_colors[idx] = new_c
            cur_count += delta
            if cur_count < best_count:
                best_count = cur_count
                best_colors = cur_colors[:]
            if temp > min_temp:
                temp *= cooling
            else:
                # Reheat?
                temp = 5.0
                
    return best_colors, best_count

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Keep margin before 150s kill
    
    rng = random.Random(42)
    np.random.seed(42)
    
    # Known record is 50. We want to see if we can beat it or at least match it efficiently.
    # The seed is 32.
    # Strategy:
    # 1. Start with n=50.
    # 2. Use a good initialization. 
    #    - Random is bad.
    #    - Spectral is okay.
    #    - Greedy is good.
    #    - Let's try a hybrid: Start with a random 4-coloring of a smaller graph (e.g. 40) that is known to be valid? 
    #      We don't have a valid 40 in seed.
    #    - Let's use the "growth" strategy mentioned in inspirations: 
    #      Start with K_32 (valid). Try to add vertices one by one.
    #      When adding vertex v, for each u < v, we must pick a color for (u,v) such that 
    #      no mono triangle is formed with existing vertices.
    #      Specifically, for a fixed u, color c is forbidden if there exists w < u such that 
    #      color(u,w) == c and color(v,w) == c.
    #      So for each u, the set of valid colors for (u,v) is {0,1,2,3} \ Forbidden(u).
    #      We want to pick colors that keep the graph "extendable" or minimize future conflicts.
    
    best_result = {'n': 32, 'colors': _build_seed_32()}
    
    # Try to grow from 32 to 50
    current_n = 32
    current_colors = best_result['colors']
    
    # We will try to extend current_colors.
    # Represent colors as a list. When we add a vertex, we append new edges.
    
    # To make it efficient, let's use a matrix or a list of lists for adjacency colors.
    # But the interface requires a flat list.
    # Let's convert to a matrix for internal manipulation during growth.
    
    adj = [[0] * current_n for _ in range(current_n)]
    idx = 0
    for i in range(current_n):
        for j in range(i + 1, current_n):
            c = current_colors[idx]
            adj[i][j] = c
            adj[j][i] = c
            idx += 1
    
    # Try to add vertices
    for new_v in range(current_n, 50):
        if time.time() > deadline:
            break
        
        # Try multiple times to add this vertex with different random strategies
        success = False
        for attempt in range(10):
            # Try to assign colors to edges (u, new_v) for u in 0..new_v-1
            # For each u, valid colors are those c such that no w < u has adj[u][w]==c and adj[w][new_v]==c.
            # But adj[w][new_v] is not known yet.
            
            # This is a CSP. 
            # Greedy approach:
            # For each u, calculate the set of "dangerous" colors.
            # A color c is dangerous for edge (u, new_v) if it creates a mono triangle (u, w, new_v) for some w.
            # This depends on the choice of color for (w, new_v) if w < u? No, w is arbitrary.
            # Actually, if we assign color c to (u, new_v), then for any w, if adj[u][w] == c, 
            # then we MUST NOT assign color c to (w, new_v).
            
            # So, we can process vertices in some order.
            # Let's try random order of u.
            order = list(range(new_v))
            rng.shuffle(order)
            
            new_adj = [[0] * (new_v + 1) for _ in range(new_v + 1)]
            for i in range(new_v):
                for j in range(new_v + 1):
                    if i < j < new_v:
                        new_adj[i][j] = adj[i][j]
                    elif j < i < new_v:
                        new_adj[i][j] = adj[j][i]
                    else:
                        new_adj[i][j] = 0 # To be filled or unused
            
            # Constraints:
            # For each u, v (new), w: not (new_adj[u][v] == new_adj[u][w] == new_adj[v][w])
            
            # Let's use backtracking or just greedy with lookahead?
            # Greedy:
            # For each u in order:
            #   Determine valid colors for new_adj[u][new_v].
            #   A color c is valid if:
            #     For all w already assigned (i.e., w is in the set of processed vertices? No, w is any existing vertex).
            #     Wait, if we assign c to (u, new_v), then for ANY w (0..new_v-1), 
            #     if new_adj[u][w] == c, then new_adj[w][new_v] must NOT be c.
            #     If new_adj[w][new_v] is already assigned, we must check.
            #     If it's not assigned yet, we create a constraint.
            
            # This is getting complex. Let's try a simpler heuristic:
            # Randomly assign colors, then check for mono triangles. If any, restart.
            # But that's slow for n=50.
            
            # Better: 
            # For each u, the color c is "blocked" if there exists w such that 
            # adj[u][w] == c and the color we *want* to assign to (w, new_v) is c.
            # But we don't know (w, new_v) yet.
            
            # Let's try: 
            # For each u, count for each color c, how many w have adj[u][w] == c.
            # If we pick c for (u, new_v), then for all those w, we cannot pick c for (w, new_v).
            
            # Heuristic: 
            # Sort u by the number of "constraints" they impose.
            # Or just random.
            
            # Let's try a random greedy assignment.
            assigned = {}
            possible = {u: set([0,1,2,3]) for u in range(new_v)}
            
            # For each u, the color c is forbidden if picking c would force (w, new_v) != c for all w with adj[u][w]==c.
            # This is a global constraint.
            
            # Let's just try to find a valid assignment using backtracking with pruning.
            # Since new_v is up to 50, and we are adding 1 vertex, it's a 50-variable CSP.
            # Might be slow.
            
            # Alternative: Use the local search on the extended graph.
            # Create a graph of size new_v+1.
            # Initialize edges (u, new_v) randomly.
            # Run local search for a short time.
            
            if attempt == 0:
                # Random init for new edges
                init_colors = current_colors[:]
                for u in range(new_v):
                    init_colors.append(rng.randint(0, 3))
            else:
                # Greedy init
                init_colors = current_colors[:]
                # Convert to matrix again? No, just use the flat list.
                # For u in 0..new_v-1:
                #   Pick color c for (u, new_v) that minimizes the number of w such that 
                #   color(u,w)==c and color(w,new_v)==c.
                #   But color(w,new_v) is not known.
                #   So just pick c that minimizes the number of w with color(u,w)==c.
                #   This is the "least used" color for u.
                
                # We need to store the current colors in a structure to query quickly.
                # Let's use the matrix adj for the current graph.
                # For the new vertex, we will build a new matrix.
                
                pass # This branch is too complicated to implement quickly in the loop.
            
            # Let's just use the random init + local search approach for simplicity and robustness.
            # If it fails, we try again with a different seed.
            
            res_colors, res_count = _local_search(init_colors, new_v + 1, min(deadline, time.time() + 5), rng)
            
            if res_count == 0:
                current_n = new_v + 1
                current_colors = res_colors
                success = True
                break
        
        if not success:
            # Could not extend. Try to repair the current graph? 
            # Or just stop.
            # If we are at n=32, we can stop.
            # If we are at n=40, we can stop.
            # Let's try to extend a bit more with more attempts?
            # No, let's break to save time.
            break
    
    # If we successfully extended to new_n, update best_result
    if current_n > best_result['n']:
        best_result = {'n': current_n, 'colors': current_colors}
        
    # Final verification
    if best_result['n'] >= 3:
        tri = _find_mono_triangles(best_result['colors'], best_result['n'], limit=1)
        if tri:
            # Fallback to seed
            best_result = {'n': 32, 'colors': _build_seed_32()}
            
    return best_result
# EVOLVE-BLOCK-END
