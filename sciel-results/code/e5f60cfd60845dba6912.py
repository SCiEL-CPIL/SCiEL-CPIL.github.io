"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
_DEADLINE = None
_START_TIME = None

def _init_time():
    global _DEADLINE, _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        _DEADLINE = _START_TIME + 115.0  # safety margin before 120s limit

def _time_left():
    return _DEADLINE - time.time()

def construct():
    _init_time()
    
    # Greenwood-Gleason colouring of K16 by cubic-residue classes of GF(16)
    # This gives a 3-coloring of K16 with no monochromatic triangle.
    # We use this as a base and try to extend it.
    
    # First, build the base K16 coloring using GF(16) cubic residues
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
    
    # Build K16 coloring: color(i,j) = log[i^j] % 3 for i != j in GF(16)
    base_colors = {}
    for i in range(16):
        for j in range(i+1, 16):
            base_colors[(i,j)] = log[i ^ j] % 3
    
    # Now try to extend to larger n using local search
    # Start with n=16 and try to add vertices
    
    def make_full_coloring(n, base_n=16):
        """Create a full coloring for n vertices, extending the base K16."""
        colors = [0] * (n*(n-1)//2)
        # Fill in base edges
        for i in range(base_n):
            for j in range(i+1, base_n):
                idx = i*n - (i+1)//2 + (j - i - 1)
                colors[idx] = base_colors[(i,j)]
        # New vertices (base_n to n-1) get random initial colors
        for i in range(base_n, n):
            for j in range(i+1, n):
                idx = i*n - (i+1)//2 + (j - i - 1)
                colors[idx] = random.randint(0, 3)
        # Edges between base and new vertices
        for i in range(base_n):
            for j in range(base_n, n):
                idx = i*n - (i+1)//2 + (j - i - 1)
                colors[idx] = random.randint(0, 3)
        return colors
    
    def get_color(colors, n, i, j):
        if i > j:
            i, j = j, i
        return colors[i*n - (i+1)//2 + (j - i - 1)]
    
    def set_color(colors, n, i, j, c):
        if i > j:
            i, j = j, i
        colors[i*n - (i+1)//2 + (j - i - 1)] = c
    
    def count_mono_triangles(colors, n):
        """Count monochromatic triangles."""
        count = 0
        for c in range(4):
            # For each color c, count triangles where all 3 edges have color c
            for i in range(n):
                for j in range(i+1, n):
                    if get_color(colors, n, i, j) == c:
                        for k in range(j+1, n):
                            if get_color(colors, n, i, k) == c and get_color(colors, n, j, k) == c:
                                count += 1
        return count
    
    def local_search(colors, n, max_iters=50000):
        """Improve coloring by flipping edges to reduce mono triangles."""
        for it in range(max_iters):
            if _time_left() < 5.0:
                break
            if count_mono_triangles(colors, n) == 0:
                break
            # Pick a random edge that's in a mono triangle and flip it
            # First find edges in mono triangles
            bad_edges = []
            for c in range(4):
                for i in range(n):
                    for j in range(i+1, n):
                        if get_color(colors, n, i, j) == c:
                            # Check if this edge is in a mono triangle of color c
                            for k in range(n):
                                if k != i and k != j:
                                    if get_color(colors, n, min(i,k), max(i,k)) == c and \
                                       get_color(colors, n, min(j,k), max(j,k)) == c:
                                        bad_edges.append((i, j))
                                        break
            if not bad_edges:
                break
            i, j = random.choice(bad_edges)
            old_c = get_color(colors, n, i, j)
            new_c = (old_c + 1) % 4
            set_color(colors, n, i, j, new_c)
        return colors
    
    # Try to find a valid coloring for increasing n
    best_n = 16
    best_colors = None
    
    # First verify base K16 is valid
    base_full = make_full_coloring(16, 16)
    if count_mono_triangles(base_full, 16) == 0:
        best_colors = base_full[:]
    
    # Try to extend
    for target_n in range(17, 33):
        if _time_left() < 10.0:
            break
        found = False
        for attempt in range(50):
            if _time_left() < 5.0:
                break
            colors = make_full_coloring(target_n, 16)
            colors = local_search(colors, target_n, max_iters=20000)
            if count_mono_triangles(colors, target_n) == 0:
                best_n = target_n
                best_colors = colors[:]
                found = True
                break
        if not found and target_n > 16:
            # If we couldn't extend further, stop
            break
    
    # If we found a valid coloring, return it
    if best_colors is not None:
        return {'n': best_n, 'colors': best_colors}
    
    # Fallback: return the base K16
    return {'n': 16, 'colors': base_full}
# EVOLVE-BLOCK-END
