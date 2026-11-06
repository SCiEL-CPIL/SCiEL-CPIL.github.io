"""Improved program for edge-colourings of K16 with >=5 colours per K4."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

def _idx(i, j):
    if i > j:
        i, j = j, i
    return i * 16 - i * (i + 1) // 2 + j - i - 1

def _validate_list(colors_list):
    """Check every K4 spans at least 5 distinct colours using flat list."""
    for quad in itertools.combinations(range(16), 4):
        a, b, c, d = quad
        e1 = colors_list[_idx(a, b)]
        e2 = colors_list[_idx(a, c)]
        e3 = colors_list[_idx(a, d)]
        e4 = colors_list[_idx(b, c)]
        e5 = colors_list[_idx(b, d)]
        e6 = colors_list[_idx(c, d)]
        cols = {e1, e2, e3, e4, e5, e6}
        if len(cols) < 5:
            return False
    return True

def _count_bad_k4(colors_list):
    """Count number of K4s with fewer than 5 distinct colours."""
    bad = 0
    for quad in itertools.combinations(range(16), 4):
        a, b, c, d = quad
        e1 = colors_list[_idx(a, b)]
        e2 = colors_list[_idx(a, c)]
        e3 = colors_list[_idx(a, d)]
        e4 = colors_list[_idx(b, c)]
        e5 = colors_list[_idx(b, d)]
        e6 = colors_list[_idx(c, d)]
        cols = {e1, e2, e3, e4, e5, e6}
        if len(cols) < 5:
            bad += 1
    return bad

def _make_f4_geometry_coloring():
    """
    Map vertices to points in F_4^2 and color edges by line slope.
    F_4 = {0, 1, ω, ω+1} where ω^2 = ω + 1
    This gives 16 vertices and 5 slopes (parallel classes).
    """
    # F_4 elements: 0, 1, w, w+1 where w^2 = w+1
    # Represent as: 0, 1, 2 (for w), 3 (for w+1)
    # Addition table in F_4:
    #   0+0=0, 0+1=1, 0+2=2, 0+3=3
    #   1+0=1, 1+1=0, 1+2=3, 1+3=2
    #   2+0=2, 2+1=3, 2+2=0, 2+3=1
    #   3+0=3, 3+1=2, 3+2=1, 3+3=0
    add_f4 = [
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 1, 0]
    ]
    
    # Multiplication table in F_4:
    #   0*x=0 for all x
    #   1*x=x for all x
    #   2*0=0, 2*1=2, 2*2=3, 2*3=1
    #   3*0=0, 3*1=3, 3*2=1, 3*3=2
    mul_f4 = [
        [0, 0, 0, 0],
        [0, 1, 2, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2]
    ]
    
    # Map vertex index (0-15) to (x, y) in F_4^2
    # vertex v -> (v // 4, v % 4)
    def vertex_to_point(v):
        return (v // 4, v % 4)
    
    def f4_sub(a, b):
        # a - b in F_4
        return add_f4[a][b]  # In F_4, subtraction is same as addition (char 2)
    
    def f4_div(a, b):
        # a / b in F_4 (b != 0)
        if b == 0:
            return None
        if b == 1:
            return a
        if b == 2:
            # 2 * 3 = 1, so 2^-1 = 3
            return mul_f4[3][a]
        if b == 3:
            # 3 * 2 = 1, so 3^-1 = 2
            return mul_f4[2][a]
        return None
    
    # Compute slope for edge (p1, p2)
    # If x1 != x2: slope = (y2 - y1) / (x2 - x1)
    # If x1 == x2: slope = infinity (vertical line)
    def compute_slope(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = f4_sub(x2, x1)
        dy = f4_sub(y2, y1)
        if dx == 0:
            return 4  # infinity
        slope = f4_div(dy, dx)
        return slope  # 0, 1, 2, or 3
    
    # Build coloring
    colors = [0] * 120
    for i in range(16):
        for j in range(i + 1, 16):
            p1 = vertex_to_point(i)
            p2 = vertex_to_point(j)
            slope = compute_slope(p1, p2)
            colors[_idx(i, j)] = slope
    
    return colors

def _make_1factorization_cyclic():
    """Standard cyclic 1-factorization of K16. Returns list of 120 colors."""
    colors = [0] * 120
    for i in range(15):
        colors[_idx(15, i)] = i
        for j in range(1, 8):
            a = (i + j) % 15
            b = (i - j) % 15
            if a > b:
                a, b = b, a
            colors[_idx(a, b)] = i
    return colors

def _local_search_1factor(colors, max_time, seed=42):
    """Local search: swap edges between factors to reduce bad K4 count."""
    rng = random.Random(seed)
    t0 = time.time()
    deadline = max_time
    
    factor = [[] for _ in range(15)]
    for idx in range(120):
        c = colors[idx]
        factor[c].append(idx)
    
    bad_count = _count_bad_k4(colors)
    
    if bad_count == 0:
        return colors, 0
    
    all_k4s = list(itertools.combinations(range(16), 4))
    k4_edges = []
    for a, b, c, d in all_k4s:
        k4_edges.append([_idx(a,b), _idx(a,c), _idx(a,d), _idx(b,c), _idx(b,d), _idx(c,d)])
    
    edge_to_k4s = [[] for _ in range(120)]
    for ki, edges in enumerate(k4_edges):
        for e in edges:
            edge_to_k4s[e].append(ki)
    
    best_colors = colors[:]
    best_bad = bad_count
    
    iterations = 0
    while time.time() - t0 < deadline:
        iterations += 1
        f1 = rng.randint(0, 14)
        f2 = rng.randint(0, 14)
        if f1 == f2:
            continue
        if not factor[f1] or not factor[f2]:
            continue
        e1 = factor[f1][rng.randint(0, len(factor[f1])-1)]
        e2 = factor[f2][rng.randint(0, len(factor[f2])-1)]
        
        old_c1 = colors[e1]
        old_c2 = colors[e2]
        colors[e1] = f2
        colors[e2] = f1
        
        total_bad = _count_bad_k4(colors)
        
        if total_bad < best_bad:
            best_bad = total_bad
            best_colors = colors[:]
            bad_count = total_bad
            # Update factor membership
            factor[f1].remove(e1)
            factor[f2].append(e1)
            factor[f2].remove(e2)
            factor[f1].append(e2)
            if best_bad == 0:
                return best_colors, 0
        else:
            # Revert
            colors[e1] = old_c1
            colors[e2] = old_c2
    
    return best_colors, best_bad

def _try_1factorization_with_swaps(max_time, seed=42):
    """Try cyclic 1-factorization then local search."""
    colors = _make_1factorization_cyclic()
    bad = _count_bad_k4(colors)
    
    if bad == 0:
        return colors, 0
    
    result, bad = _local_search_1factor(colors, max_time, seed)
    return result, bad

def _make_17coloring():
    """The (i+j) % 17 construction - guaranteed valid, uses 17 colors."""
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]

def construct():
    t0 = time.time()
    deadline = 145  # safety margin before 150s kill
    
    # Strategy 1: Try F4 geometry coloring (should give 5 colors if it works)
    colors_f4 = _make_f4_geometry_coloring()
    bad_f4 = _count_bad_k4(colors_f4)
    if bad_f4 == 0:
        return colors_f4
    
    # Strategy 2: Try cyclic 1-factorization + local search
    remaining = deadline - (time.time() - t0)
    if remaining > 5:
        colors15, bad = _try_1factorization_with_swaps(min(30, remaining), seed=42)
        if bad == 0:
            return colors15
    
    # Strategy 3: If F4 geometry has few bad K4s, try local search on it
    remaining = deadline - (time.time() - t0)
    if remaining > 5 and bad_f4 > 0 and bad_f4 < 100:
        # Convert F4 coloring to a format suitable for local search
        # The F4 coloring uses colors 0-4, so we need to adapt
        # For now, just try to improve it
        pass
    
    # Fallback: 17-coloring (guaranteed valid)
    return _make_17coloring()

# EVOLVE-BLOCK-END
