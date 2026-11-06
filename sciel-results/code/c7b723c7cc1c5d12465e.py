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

def _has_mono_triangle_with_new_edge(colors, n, new_vertex, new_colors, deadline=None):
    """
    Check if adding a new vertex 'new_vertex' with given colors to existing vertices 
    creates a monochromatic triangle.
    colors: list of colors for edges among existing vertices (0 to n-1)
    new_vertex: index of the new vertex (n)
    new_colors: list of length n, color of edge (v, new_vertex) for v in 0..n-1
    Returns True if a mono triangle is created.
    """
    # A triangle (u, v, new_vertex) is mono if:
    # color(u, v) == color(u, new_vertex) == color(v, new_vertex)
    # We need to check all pairs (u, v) in existing vertices.
    for u in range(n):
        cu_new = new_colors[u]
        for v in range(u + 1, n):
            cv_new = new_colors[v]
            if cu_new == cv_new:
                # Check if edge (u, v) has the same color
                idx = _color_index(n, u, v)
                if colors[idx] == cu_new:
                    return True
    return False

def _greedy_construct(start_colors, start_n, target_n, deadline, rng):
    """
    Greedily add vertices to start_colors until target_n is reached or deadline passes.
    Returns (colors, n) or None if failed.
    """
    current_colors = start_colors[:]
    current_n = start_n
    
    while current_n < target_n:
        if time.time() > deadline:
            return None
        
        new_vertex = current_n
        # Try to assign colors to edges (v, new_vertex) for v in 0..current_n-1
        # We can try random assignments or greedy backtracking.
        # For speed, let's try a few random full assignments.
        
        success = False
        # Try up to 50 random assignments for the new vertex's edges
        for _ in range(50):
            new_edge_colors = [rng.randint(0, 3) for _ in range(current_n)]
            
            # Check validity
            if not _has_mono_triangle_with_new_edge(current_colors, current_n, new_vertex, new_edge_colors):
                # Valid! Append to colors
                for v in range(current_n):
                    current_colors.append(new_edge_colors[v])
                current_n += 1
                success = True
                break
        
        if not success:
            # Failed to add vertex with random attempts. 
            # This is a dead end. Return None to signal failure.
            return None
            
    return current_colors, current_n

def _local_search_refine(colors, n, deadline, rng):
    """
    If greedy construction gets stuck or we want to try to fix a near-miss,
    we can use local search. But since greedy maintains validity, we only 
    need local search if we start from an invalid state (which we don't here).
    However, if greedy fails at n=45, maybe a different random seed for the 
    new vertex edges would have worked? 
    The current _greedy_construct already tries 50 randoms.
    
    Let's add a simple backtracking step: if greedy fails, try to remove 
    the last vertex and try different colors for it? 
    Actually, the main failure mode is that the existing graph is too constrained.
    
    For now, this function is a placeholder for potential refinement if needed,
    but the primary strategy is pure greedy growth from the seed.
    """
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    rng = random.Random(12345)
    
    # Start with the known valid K32 coloring
    seed_colors = _build_seed_32()
    seed_n = 32
    
    best_result = {'n': 32, 'colors': seed_colors}
    
    # Try to grow from 32 up to 50
    # We can try multiple times with different random seeds for the greedy steps
    # to see if we can reach higher n.
    
    max_target = 50
    
    # Try greedy growth
    # We can also try to start from a slightly perturbed seed? 
    # No, the seed is fixed valid.
    
    # Let's try multiple runs of greedy growth
    for run in range(20): # 20 attempts
        if time.time() > deadline:
            break
            
        # Reset RNG for each run? Or use global rng?
        # Using global rng ensures different paths.
        
        res = _greedy_construct(seed_colors, seed_n, max_target, deadline, rng)
        
        if res is not None:
            res_colors, res_n = res
            if res_n > best_result['n']:
                best_result = {'n': res_n, 'colors': res_colors}
                
                # If we reached 50, we are done (record)
                if res_n >= 50:
                    break
        else:
            # Greedy failed. 
            # This means we couldn't add the next vertex with 50 random tries.
            # We can try to increase the number of random tries for the next vertex 
            # or accept that this seed is a dead end for this run.
            # Since we have multiple runs, we just continue.
            
            # Optional: If we failed at a specific n, we could try to "relax" 
            # by doing a local search on the existing graph to change some edges 
            # to make it easier to add the next vertex? 
            # But that risks breaking validity.
            # For this structural change, we stick to pure greedy.
            pass

    # Verification
    if best_result['n'] >= 3:
        # Simple check for mono triangles in the final result
        # To save time, we only check if n > 32
        if best_result['n'] > 32:
            colors = best_result['colors']
            n = best_result['n']
            valid = True
            # Efficient check: for each pair (i,j), check if there's a k such that 
            # (i,j,k) is mono.
            # We can just iterate all triangles. For n=50, this is ~20k triangles.
            # 20k * 3 checks is fast.
            for i in range(n):
                for j in range(i + 1, n):
                    cij = colors[_color_index(n, i, j)]
                    for k in range(j + 1, n):
                        if colors[_color_index(n, i, k)] == cij and colors[_color_index(n, j, k)] == cij:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if not valid:
                # Fallback to seed
                best_result = {'n': 32, 'colors': _build_seed_32()}

    return best_result
# EVOLVE-BLOCK-END
