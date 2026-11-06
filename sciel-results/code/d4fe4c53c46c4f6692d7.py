"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def _construct_32():
    """Greenwood-Gleason / Clebsch-based 4-colouring of K32 (two copies of K16 with colour 3 between)."""
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

def _check_valid(n, colors):
    """Verify no monochromatic triangle."""
    if len(colors) != n * (n - 1) // 2:
        return False
    # Build adjacency per colour
    adj = [set() for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            if c < 0 or c > 3:
                return False
            adj[c].add((i, j))
            idx += 1
    # For each colour, check no triangle
    for c in range(4):
        # Build adjacency list for this colour
        neigh = [set() for _ in range(n)]
        for (i, j) in adj[c]:
            neigh[i].add(j)
            neigh[j].add(i)
        for i in range(n):
            for j in neigh[i]:
                if j <= i:
                    continue
                for k in neigh[j]:
                    if k > j and k in neigh[i]:
                        return False
    return True

def construct():
    """
    Try to find a 4-colouring of K_n without monochromatic triangles, maximizing n.
    Start from the known K32 construction. Attempt local search / hill-climbing
    to extend to larger n, but given the difficulty (R(3,3,3,3)=51 is open, best known 50),
    we likely can't beat 32 quickly. However, we try a few strategies within time budget.
    """
    start = time.time()
    deadline = start + 120.0  # 120s wall-clock limit
    
    # Baseline: K32
    best = _construct_32()
    best_n = best['n']
    
    # Try to extend to K33, K34, ... via local search
    # Strategy: start with the K32 colouring, add vertices one by one,
    # and use local search to fix monochromatic triangles.
    
    def _try_extend(n_target, max_time=60.0):
        """Attempt to find a valid 4-colouring of K_{n_target}."""
        t0 = time.time()
        rng = random.Random(42)
        
        # Initialize: use K32 colours for first 32 vertices, random for new ones
        if n_target <= 32:
            # Already have it
            if n_target == 32:
                return best
            # Need to shrink - just return best
            return best
        
        # Build initial colouring
        colors = list(best['colors'])  # first 32*31/2 edges
        # Add edges involving new vertices
        for i in range(32, n_target):
            for j in range(i + 1, n_target):
                colors.append(rng.randint(0, 3))
            for j in range(i):
                colors.append(rng.randint(0, 3))
        
        # Wait, the order is lexicographic (i,j) with i<j.
        # Let's rebuild properly.
        colors = []
        for i in range(n_target):
            for j in range(i + 1, n_target):
                if i < 32 and j < 32:
                    # Use existing colour from best
                    idx = i * 31 - i * (i + 1) // 2 + (j - i - 1)
                    colors.append(best['colors'][idx])
                else:
                    colors.append(rng.randint(0, 3))
        
        # Local search: iteratively find monochromatic triangles and recolor
        # Use a smarter approach: for each edge, if it's part of a mono triangle, try flipping
        
        def _find_mono_triangle(colors, n):
            """Return (i, j, k, c) if a monochromatic triangle exists, else None."""
            # Build adjacency per colour
            adj = [[] for _ in range(4)]
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[idx]
                    adj[c].append((i, j))
                    idx += 1
            # Check each colour for triangles
            for c in range(4):
                # Build neighbor sets
                neigh = [set() for _ in range(n)]
                for (i, j) in adj[c]:
                    neigh[i].add(j)
                    neigh[j].add(i)
                for i in range(n):
                    for j in neigh[i]:
                        if j <= i:
                            continue
                        common = neigh[i] & neigh[j]
                        for k in common:
                            if k > j:
                                return (i, j, k, c)
            return None
        
        # Local search loop
        best_local_n = 32
        for iteration in range(1000000):
            if time.time() - t0 > max_time:
                break
            tri = _find_mono_triangle(colors, n_target)
            if tri is None:
                return {'n': n_target, 'colors': colors}
            i, j, k, c = tri
            # Try to recolor one of the three edges to break the triangle
            # Find the index of edge (i,j), (i,k), (j,k)
            def _edge_idx(a, b, n):
                if a > b:
                    a, b = b, a
                return a * (n - 1) - a * (a + 1) // 2 + (b - a - 1)
            
            ei_j = _edge_idx(i, j, n_target)
            ei_k = _edge_idx(i, k, n_target)
            ej_k = _edge_idx(j, k, n_target)
            
            # Try all 4 colours for each of the 3 edges, pick one that doesn't create new triangles
            # Simplified: just try random recolors
            edges = [ei_j, ei_k, ej_k]
            rng.shuffle(edges)
            for eidx in edges:
                old_c = colors[eidx]
                new_c = (old_c + 1 + rng.randint(0, 2)) % 4
                colors[eidx] = new_c
                # Check if this creates a new triangle (quick check)
                # For efficiency, just accept and let the next iteration catch issues
                break
        
        # If we didn't succeed, return None
        return None
    
    # Try to extend to 33, 34, etc.
    for n_target in range(33, 38):
        if time.time() > deadline - 10:
            break
        result = _try_extend(n_target, max_time=30.0)
        if result is not None and _check_valid(result['n'], result['colors']):
            best = result
            best_n = result['n']
    
    # Verify the best result
    if not _check_valid(best_n, best['colors']):
        # Fallback to known valid K32
        best = _construct_32()
        best_n = 32
    
    return best
# EVOLVE-BLOCK-END
