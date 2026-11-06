import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Try n=26 first (known algebraic construction exists)
    if time.time() < deadline - 5:
        result = try_affine_extension_26(deadline)
        if result is not None:
            return result

    # Fallback: n=25 affine plane construction
    if time.time() < deadline - 2:
        result = affine_plane_25()
        if result is not None:
            return result

    # Final fallback: small valid coloring
    return {'n': 11, 'colors': [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]}


def affine_plane_25():
    """
    Construct a valid 5-coloring of K_25 using AG(2,5).
    
    Points are (x, y) for x, y in {0, 1, 2, 3, 4}.
    The 6 parallel classes:
      - Slope 0 (horizontal): lines {y = c} for c in 0..4
      - Slope 1: lines {y - x = c}
      - Slope 2: lines {y - 2x = c}
      - Slope 3: lines {y - 3x = c}
      - Slope 4: lines {y - 4x = c}
      - Vertical: lines {x = c}
    
    Color each parallel class with one of the 5 colors.
    Each color class is a disjoint union of K5's, which is C4-free.
    """
    n = 25
    # Vertex index: (x, y) -> x * 5 + y
    def idx(x, y):
        return x * 5 + y

    # Build the 6 parallel classes
    parallel_classes = []
    # Slope 0: horizontal lines y = c
    for c in range(5):
        line = [idx(x, c) for x in range(5)]
        parallel_classes.append(line)
    # Slope 1: y - x = c
    for c in range(5):
        line = [idx(x, (x + c) % 5) for x in range(5)]
        parallel_classes.append(line)
    # Slope 2: y - 2x = c
    for c in range(5):
        line = [idx(x, (2 * x + c) % 5) for x in range(5)]
        parallel_classes.append(line)
    # Slope 3: y - 3x = c
    for c in range(5):
        line = [idx(x, (3 * x + c) % 5) for x in range(5)]
        parallel_classes.append(line)
    # Slope 4: y - 4x = c
    for c in range(5):
        line = [idx(x, (4 * x + c) % 5) for x in range(5)]
        parallel_classes.append(line)
    # Vertical: x = c
    for c in range(5):
        line = [idx(c, y) for y in range(5)]
        parallel_classes.append(line)

    # Verify: each pair of vertices appears in exactly one parallel class
    pair_to_class = {}
    for ci, line in enumerate(parallel_classes):
        for a in range(5):
            for b in range(a + 1, 5):
                u, v = line[a], line[b]
                key = (min(u, v), max(u, v))
                if key in pair_to_class:
                    return None  # invalid
                pair_to_class[key] = ci

    if len(pair_to_class) != 300:
        return None

    # Assign colors to parallel classes: use 5 colors, one parallel class gets no color
    # Actually we have 6 parallel classes but only 5 colors.
    # Each parallel class has 5 lines (cliques). We can assign colors to individual lines.
    # But simpler: assign each parallel class a color, but we have 6 classes and 5 colors.
    # 
    # Alternative: each color class is a union of some parallel classes' lines.
    # But a color class can't contain two lines from the same parallel class (they'd share a vertex? No, lines in same parallel class are disjoint).
    # Actually lines in the same parallel class are disjoint, so their union is a matching of K5's, which is C4-free.
    # So we can put multiple parallel classes in the same color.
    #
    # But if we put two different parallel classes in the same color, do we get a C4?
    # A line from parallel class A and a line from parallel class B share exactly one vertex.
    # So the union of a line from A and a line from B is two K5's sharing one vertex - no C4.
    # In fact, any collection of lines from distinct parallel classes, where each line is a K5,
    # the union is C4-free because any two lines from different parallel classes share at most one vertex.
    #
    # Wait, but a color class could contain multiple lines. Let's just assign each line a color.
    # There are 30 lines total (6 parallel classes * 5 lines). Assign 5 colors.
    # Each color gets 6 lines. As long as no two lines in the same color share more than one vertex,
    # we're fine. Lines from the same parallel class are disjoint (share 0 vertices).
    # Lines from different parallel classes share exactly 1 vertex.
    # So any assignment works! Let's just assign colors round-robin.
    
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    colors = [0] * len(edges)
    
    # Assign each line a color
    line_colors = []
    for ci in range(6):
        for li in range(5):
            line_colors.append(ci * 5 + li)  # line index
    
    # Actually, let's assign colors to lines. 30 lines, 5 colors.
    # Line index: ci * 5 + li where ci is parallel class, li is line within class.
    # Assign color = (ci * 5 + li) % 5 = li % 5... that's just li.
    # Or color = (ci + li) % 5. Let's use color = (ci * 3 + li) % 5 to spread out.
    
    line_color = {}
    for ci in range(6):
        for li in range(5):
            line_color[(ci, li)] = (ci * 3 + li) % 5
    
    # Now assign colors to edges
    for (u, v), ci in pair_to_class.items():
        # Find which line in parallel class ci contains both u and v
        line = parallel_classes[ci]
        for li in range(5):
            # Check if u and v are both in this line
            # The line is parallel_classes[ci], which is a list of 5 vertices
            # li identifies the line within the parallel class
            # Wait, I structured parallel_classes as a list of 6 elements, each a list of 5 lines? No.
            # parallel_classes[ci] is a list of 5 vertices (one line).
            # I need to restructure.
            pass
    
    # Let me restructure: parallel_classes[ci] should be a list of 5 lines, each a list of 5 vertices.
    # Let me rebuild.
    
    parallel_classes2 = []
    # Slope 0: horizontal lines y = c
    lines = []
    for c in range(5):
        line = [idx(x, c) for x in range(5)]
        lines.append(line)
    parallel_classes2.append(lines)
    # Slope 1
    lines = []
    for c in range(5):
        line = [idx(x, (x + c) % 5) for x in range(5)]
        lines.append(line)
    parallel_classes2.append(lines)
    # Slope 2
    lines = []
    for c in range(5):
        line = [idx(x, (2 * x + c) % 5) for x in range(5)]
        lines.append(line)
    parallel_classes2.append(lines)
    # Slope 3
    lines = []
    for c in range(5):
        line = [idx(x, (3 * x + c) % 5) for x in range(5)]
        lines.append(line)
    parallel_classes2.append(lines)
    # Slope 4
    lines = []
    for c in range(5):
        line = [idx(x, (4 * x + c) % 5) for x in range(5)]
        lines.append(lines[-1])  # placeholder
    # Fix slope 4
    lines = []
    for c in range(5):
        line = [idx(x, (4 * x + c) % 5) for x in range(5)]
        lines.append(line)
    parallel_classes2[4] = lines
    # Vertical
    lines = []
    for c in range(5):
        line = [idx(c, y) for y in range(5)]
        lines.append(line)
    parallel_classes2.append(lines)
    
    # Now build pair to (ci, li)
    pair_to_line = {}
    for ci in range(6):
        for li in range(5):
            line = parallel_classes2[ci][li]
            for a in range(5):
                for b in range(a + 1, 5):
                    u, v = line[a], line[b]
                    key = (min(u, v), max(u, v))
                    if key in pair_to_line:
                        return None
                    pair_to_line[key] = (ci, li)
    
    if len(pair_to_line) != 300:
        return None
    
    # Assign colors
    colors = [0] * len(edges)
    for e_idx, (u, v) in enumerate(edges):
        ci, li = pair_to_line[(u, v)]
        colors[e_idx] = (ci * 3 + li) % 5
    
    # Verify
    if is_valid_coloring(n, colors):
        return {'n': n, 'colors': colors}
    
    return None


def is_valid_coloring(n, colors):
    """Check if a coloring has no monochromatic C4."""
    neighbors = [[0] * n for _ in range(5)]
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    for idx, (i, j) in enumerate(edges):
        c = colors[idx]
        neighbors[c][i] |= (1 << j)
        neighbors[c][j] |= (1 << i)
    
    for c in range(5):
        for u in range(n):
            for v in range(u + 1, n):
                common = neighbors[c][u] & neighbors[c][v]
                if common.bit_count() > 1:
                    return False
    return True


def try_affine_extension_26(deadline):
    """
    Try to extend the AG(2,5) construction to n=26 by adding one vertex.
    Use delta-based local search to fix C4 violations.
    """
    n = 26
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    # Build the base coloring for vertices 0..24 using affine plane
    base = affine_plane_25()
    if base is None:
        return None
    
    # Initialize colors: first 300 edges from base, last 25 edges random
    colors = base['colors'][:]
    # The last 25 edges are (25, j) for j in 0..24
    for j in range(25):
        e_idx = num_edges - 25 + j
        colors[e_idx] = random.randint(0, 4)
    
    # Build neighbor bitmasks
    neighbors = [[0] * n for _ in range(5)]
    for idx, (i, j) in enumerate(edges):
        c = colors[idx]
        neighbors[c][i] |= (1 << j)
        neighbors[c][j] |= (1 << i)
    
    # Count violations
    def count_violations():
        total = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = neighbors[c][u] & neighbors[c][v]
                    cnt = common.bit_count()
                    if cnt > 1:
                        total += cnt * (cnt - 1) // 2
        return total
    
    current_violations = count_violations()
    best_violations = current_violations
    best_colors = colors[:]
    
    start_time = time.time()
    temperature = 1.0
    cooling_rate = 0.9995
    
    while time.time() < deadline - 2:
        # Pick an edge to change, preferably one involving vertex 25
        if random.random() < 0.7:
            e_idx = num_edges - 25 + random.randint(0, 24)
        else:
            e_idx = random.randint(0, num_edges - 1)
        
        old_color = colors[e_idx]
        new_color = random.randint(0, 4)
        if new_color == old_color:
            continue
        
        # Delta computation
        i, j = edges[e_idx]
        delta = 0
        
        # Remove edge (i,j) from old_color
        # For each vertex u != i,j, check if u is neighbor of both i and j in old_color
        # If so, removing the edge might reduce violations involving (i,j) pairs
        # Actually, the violation count is over pairs (u,v) of vertices with >1 common neighbor.
        # Changing color of edge (i,j) affects:
        # 1. The common neighbors of (i,j) in old_color and new_color
        # 2. For each vertex u, the common neighbors of (u,i) and (u,j)
        
        # This is complex. Let's just recompute locally.
        # Actually, let's use a simpler delta: recompute violations involving i and j.
        
        # For efficiency, let's just do full recompute if violations are low,
        # otherwise use delta.
        
        if current_violations < 200:
            # Full recompute
            neighbors[old_color][i] &= ~(1 << j)
            neighbors[old_color][j] &= ~(1 << i)
            neighbors[new_color][i] |= (1 << j)
            neighbors[new_color][j] |= (1 << i)
            colors[e_idx] = new_color
            new_violations = count_violations()
            delta = new_violations - current_violations
            
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current_violations = new_violations
                if new_violations < best_violations:
                    best_violations = new_violations
                    best_colors = colors[:]
                    if best_violations == 0:
                        return {'n': n, 'colors': colors}
            else:
                # Revert
                neighbors[new_color][i] &= ~(1 << j)
                neighbors[new_color][j] &= ~(1 << i)
                neighbors[old_color][i] |= (1 << j)
                neighbors[old_color][j] |= (1 << i)
                colors[e_idx] = old_color
        else:
            # Use delta approximation: only check violations involving i or j
            # This is an approximation but faster
            neighbors[old_color][i] &= ~(1 << j)
            neighbors[old_color][j] &= ~(1 << i)
            neighbors[new_color][i] |= (1 << j)
            neighbors[new_color][j] |= (1 << i)
            colors[e_idx] = new_color
            
            # Compute delta by checking pairs involving i or j
            delta = 0
            for c in range(5):
                # Pairs (u, i) for u != i
                for u in range(n):
                    if u == i:
                        continue
                    common = neighbors[c][u] & neighbors[c][i]
                    cnt = common.bit_count()
                    if cnt > 1:
                        delta += cnt * (cnt - 1) // 2
                # Pairs (u, j) for u != j, u != i (to avoid double counting (i,j))
                for u in range(n):
                    if u == j or u == i:
                        continue
                    common = neighbors[c][u] & neighbors[c][j]
                    cnt = common.bit_count()
                    if cnt > 1:
                        delta += cnt * (cnt - 1) // 2
                # Pair (i, j)
                common = neighbors[c][i] & neighbors[c][j]
                cnt = common.bit_count()
                if cnt > 1:
                    delta += cnt * (cnt - 1) // 2
            
            # This delta is not accurate because we're double counting and not subtracting old.
            # Let's just use full recompute for simplicity, but limit iterations.
            # Revert and use full recompute
            neighbors[new_color][i] &= ~(1 << j)
            neighbors[new_color][j] &= ~(1 << i)
            neighbors[old_color][i] |= (1 << j)
            neighbors[old_color][j] |= (1 << i)
            colors[e_idx] = old_color
            
            # Full recompute
            neighbors[old_color][i] &= ~(1 << j)
            neighbors[old_color][j] &= ~(1 << i)
            neighbors[new_color][i] |= (1 << j)
            neighbors[new_color][j] |= (1 << i)
            colors[e_idx] = new_color
            new_violations = count_violations()
            delta = new_violations - current_violations
            
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current_violations = new_violations
                if new_violations < best_violations:
                    best_violations = new_violations
                    best_colors = colors[:]
                    if best_violations == 0:
                        return {'n': n, 'colors': colors}
            else:
                neighbors[new_color][i] &= ~(1 << j)
                neighbors[new_color][j] &= ~(1 << i)
                neighbors[old_color][i] |= (1 << j)
                neighbors[old_color][j] |= (1 << i)
                colors[e_idx] = old_color
        
        temperature *= cooling_rate
        if temperature < 0.05:
            temperature = 0.05
    
    # If we found a valid coloring, return it
    if best_violations == 0:
        return {'n': n, 'colors': best_colors}
    
    return None
