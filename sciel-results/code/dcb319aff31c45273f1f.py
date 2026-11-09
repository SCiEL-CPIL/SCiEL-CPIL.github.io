"""
5-coloring of K_n without monochromatic C4, maximising n.
Uses the affine plane AG(2,5) construction for n=25,
then attempts to extend to n=26 and n=27 via randomized search.
"""
import time
import random
import itertools

def construct():
    start = time.time()
    DEADLINE = 145.0  # safety margin before 150s limit

    # Start with the deterministic AG(2,5) construction for n=25
    n25, colors25 = _ag25_coloring()

    best_n = n25
    best_colors = colors25

    # Try to extend to n=26
    if time.time() < DEADLINE - 30:
        result = _try_extend(n25, colors25, start, DEADLINE - 30)
        if result is not None:
            best_n = result[0]
            best_colors = result[1]

    # Try to extend to n=27
    if best_n == 26 and time.time() < DEADLINE - 10:
        result = _try_extend(26, best_colors, start, DEADLINE - 10)
        if result is not None:
            best_n = result[0]
            best_colors = result[1]

    return {'n': best_n, 'colors': best_colors}


def _ag25_coloring():
    """
    Deterministic 5-coloring of K_25 using the affine plane AG(2,5).
    
    Vertices: the 25 points of GF(5)^2, labeled (x,y) with x,y in {0,...,4}.
    
    Edge coloring: For two points P=(x1,y1) and Q=(x2,y2), compute the slope of PQ.
    In GF(5), there are 6 possible slopes: 0,1,2,3,4, and infinity (vertical).
    We map these 6 slopes to 5 colors using a carefully chosen mapping that 
    preserves C4-freeness.
    
    Key insight: For C4-freeness, we need that in each color class, no two vertices 
    share 2+ common neighbors. Using the slope-based coloring with a proper mapping 
    of the 6 slopes to 5 colors achieves this.
    
    One valid approach: use the fact that in AG(2,5), any two non-parallel lines 
    intersect in exactly one point. The dual construction (vertices = lines) gives 
    a natural 6-coloring. We merge two parallel classes carefully.
    
    Alternative simpler approach: Use a known valid 5-coloring derived from 
    the Paley graph or another construction.
    
    Let me use a direct verified construction:
    """
    n = 25
    colors = []
    
    # For each pair of points, determine the slope class
    # Slope classes: 0,1,2,3,4 (finite slopes) and 5 (infinity/vertical)
    # Map 6 slope classes to 5 colors: merge classes 4 and 5 (infinity)
    # But we need to verify this doesn't create C4s.
    
    # Actually, let's use a different mapping: assign each slope class to a color
    # such that the merged class is C4-free.
    
    # For vertical edges (x1==x2): assign color based on x1
    # For horizontal edges (y1==y2, i.e., slope 0): assign color 0
    # For other edges: assign based on slope
    
    # Let me try: color = slope for slopes 0-3, and color 4 for both slope 4 and infinity
    # Verify: In color 4, we have edges with slope 4 AND vertical edges.
    # A C4 in color 4 would need 4 points where consecutive edges have slope 4 or are vertical.
    # This is complex to verify manually, so let's just compute and verify.
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = i // 5, i % 5
            x2, y2 = j // 5, j % 5
            if x1 == x2:
                # Vertical: slope = infinity
                c = 4  # Merge with slope 4
            else:
                dy = (y2 - y1) % 5
                dx = (x2 - x1) % 5
                inv = pow(dx, 3, 5)
                slope = (dy * inv) % 5
                if slope == 4:
                    c = 4
                else:
                    c = slope
            colors.append(c)
    
    # Verify
    if _verify_c4_free(n, colors):
        return (n, colors)
    
    # If merging slope 4 and infinity doesn't work, try other mergers
    # Try merging slope 0 and infinity
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = i // 5, i % 5
            x2, y2 = j // 5, j % 5
            if x1 == x2:
                c = 0  # Merge with slope 0
            else:
                dy = (y2 - y1) % 5
                dx = (x2 - x1) % 5
                inv = pow(dx, 3, 5)
                slope = (dy * inv) % 5
                if slope == 0:
                    c = 0
                else:
                    c = slope
            colors.append(c)
    
    if _verify_c4_free(n, colors):
        return (n, colors)
    
    # Try all possible mergers of two slope classes into one color
    # There are C(6,2)=15 ways to choose which two slopes to merge
    slope_to_color_options = []
    for merge_a in range(6):
        for merge_b in range(merge_a + 1, 6):
            # Create mapping: all slopes except merge_a,merge_b map to their own color
            # merge_a and merge_b both map to color 0 (or any fixed color)
            # Remaining 4 slopes map to colors 1,2,3,4
            mapping = {}
            other_slopes = [s for s in range(6) if s != merge_a and s != merge_b]
            mapping[merge_a] = 0
            mapping[merge_b] = 0
            for k, s in enumerate(other_slopes):
                mapping[s] = k + 1
            
            colors = []
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    x1, y1 = i // 5, i % 5
                    x2, y2 = j // 5, j % 5
                    if x1 == x2:
                        slope_class = 5  # infinity
                    else:
                        dy = (y2 - y1) % 5
                        dx = (x2 - x1) % 5
                        inv = pow(dx, 3, 5)
                        slope_class = (dy * inv) % 5
                    colors.append(mapping[slope_class])
            
            if _verify_c4_free(n, colors):
                return (n, colors)
    
    # Fallback: use a known valid smaller coloring
    n = 11
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    return (n, colors)


def _try_extend(n_base, colors_base, start, deadline):
    """
    Try to extend a valid n_base coloring to n_base+1 by adding one vertex.
    Uses randomized greedy with backtracking.
    """
    rng = random.Random(42 + n_base)
    
    # Build color_of for the base coloring
    color_of = {}
    idx = 0
    for i in range(n_base):
        for j in range(i + 1, n_base):
            color_of[(i, j)] = colors_base[idx]
            idx += 1
    
    # Build adjacency per color: adj[c][v] = set of neighbors of v in color c
    adj = [dict() for _ in range(n_base)]
    for v in range(n_base):
        adj[v] = {c: set() for c in range(5)}
    for (u, v), c in color_of.items():
        adj[u][c].add(v)
        adj[v][c].add(u)
    
    # Common neighbor count: cn[c][u][v] = number of common neighbors of u,v in color c
    cn = [[0] * n_base for _ in range(5)]
    for c in range(5):
        for u in range(n_base):
            for v in adj[u][c]:
                if v > u:
                    # Count common neighbors of u and v in color c
                    common = len(adj[u][c] & adj[v][c])
                    cn[c][u][v] = common
                    cn[c][v][u] = common
    
    # Try to assign colors to edges (new_vertex, v) for v in 0..n_base-1
    # new_vertex = n_base
    
    for restart in range(2000):
        if time.time() > deadline:
            break
        
        order = list(range(n_base))
        rng.shuffle(order)
        
        color_new = {}
        # vertices_in_color[c] = set of v such that (new, v) is color c
        vertices_in_color = [set() for _ in range(5)]
        
        success = True
        for w in order:
            valid_colors = []
            for c in range(5):
                # Check if assigning color c to (new, w) creates a C4
                # C4 involving new: new-a-x-b-new where all edges color c
                # a,b are in vertices_in_color[c], x is common neighbor of a,b in color c
                # Also: new-w-x-a-new where w is new's neighbor, x is common neighbor of w,a
                # 
                # More precisely: after adding (new,w) in color c, check for any pair 
                # (u,v) where u,v are both neighbors of new in color c (including w),
                # and u,v share a common neighbor in color c.
                # 
                # Simple check: for each a in vertices_in_color[c], check if a and w
                # share a common neighbor in color c (excluding new).
                conflict = False
                for a in vertices_in_color[c]:
                    # Check if a and w have >= 1 common neighbor in color c
                    # (they already have 0 since the base coloring is C4-free,
                    #  but adding new doesn't change their common neighbors)
                    # Actually, the C4 would be new-a-x-w-new where (new,a),(a,x),(x,w),(w,new) all color c
                    # This requires x to be a common neighbor of a and w in color c.
                    # Since the base coloring is C4-free, a and w share at most 1 common neighbor.
                    # If they share exactly 1, then new-a-x-w-new is a C4.
                    common_aw = len(adj[a][c] & adj[w][c])
                    if common_aw >= 1:
                        conflict = True
                        break
                
                if not conflict:
                    valid_colors.append(c)
            
            if not valid_colors:
                success = False
                break
            
            # Pick a color with some heuristic
            # Prefer colors where w has fewer neighbors (less constraint)
            scores = []
            for c in valid_colors:
                score = len(adj[w][c]) + len(vertices_in_color[c])
                scores.append((score, c))
            scores.sort()
            top = scores[:min(3, len(scores))]
            c = rng.choice(top)[1]
            
            color_new[w] = c
            vertices_in_color[c].add(w)
        
        if success:
            # Build the new coloring
            new_colors = list(colors_base)
            for v in range(n_base):
                new_colors.append(color_new[v])
            
            if _verify_c4_free(n_base + 1, new_colors):
                return (n_base + 1, new_colors)
    
    return None


# Removed: replaced by _try_extend above


def _verify_c4_free(n, colors):
    """Verify that no color class contains a C4."""
    # Build color_of
    color_of = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            color_of[(i, j)] = colors[idx]
            idx += 1
    
    # For each color, check for C4
    for c in range(5):
        # Build adjacency for color c
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if color_of.get((i, j), -1) == c:
                    adj[i].add(j)
                    adj[j].add(i)
        
        # Check for C4: no two vertices should share 2+ common neighbors in color c
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u] & adj[v]
                if len(common) >= 2:
                    return False
    return True


# Keep helper functions for compatibility
def _construct_cyclic():
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

def _check_c4(n, colors):
    color_of = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            color_of[(i, j)] = colors[idx]
            idx += 1
    return _verify_c4_free(n, colors)

def _get_idx(i, j, n):
    return i * (2 * n - i - 1) // 2 + (j - i - 1)

def _validate_coloring(result):
    n = result['n']
    colors = result['colors']
    expected_len = n * (n - 1) // 2
    if len(colors) != expected_len:
        return False
    if not all(0 <= c <= 4 for c in colors):
        return False
    return _check_c4(n, colors)
