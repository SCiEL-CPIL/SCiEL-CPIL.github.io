"""Program for the rainbow-four-cycle-free coloring task on K_16."""
import time
import random
import itertools
import sys

N = 16
M = N * (N - 1) // 2  # 120 edges

# Precompute edge index for (i,j) with i<j
def get_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)

# Precompute all 4-cycles: for each 4-subset {a,b,c,d}, the 3 possible 4-cycles
# A 4-cycle on vertices (a,b,c,d) uses edges (a,b),(b,c),(c,d),(d,a)
# For a 4-subset {a,b,c,d} with a<b<c<d, the 3 cycles are:
#   (a,b,c,d): edges ab, bc, cd, da
#   (a,b,d,c): edges ab, bd, dc, ca
#   (a,c,b,d): edges ac, cb, bd, da
# Let's enumerate all 4-cycles systematically.

def build_cycles():
    """Return list of tuples of 4 edge-indices forming a 4-cycle."""
    cycles = []
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        # 3 Hamiltonian 4-cycles on these 4 vertices:
        # Cycle 1: a-b-c-d-a  -> edges (a,b),(b,c),(c,d),(d,a)
        # Cycle 2: a-b-d-c-a  -> edges (a,b),(b,d),(d,c),(c,a)
        # Cycle 3: a-c-b-d-a  -> edges (a,c),(c,b),(b,d),(d,a)
        cycles.append((get_idx(a,b), get_idx(b,c), get_idx(c,d), get_idx(d,a)))
        cycles.append((get_idx(a,b), get_idx(b,d), get_idx(d,c), get_idx(c,a)))
        cycles.append((get_idx(a,c), get_idx(c,b), get_idx(b,d), get_idx(d,a)))
    return cycles

ALL_CYCLES = build_cycles()
NUM_CYCLES = len(ALL_CYCLES)  # C(16,4)*3 = 1820*3 = 5460

def count_distinct_colors(colors):
    """Count distinct colors used."""
    return len(set(colors))

def is_valid(colors):
    """Check no 4-cycle has 4 distinct colors."""
    for e0, e1, e2, e3 in ALL_CYCLES:
        c = {colors[e0], colors[e1], colors[e2], colors[e3]}
        if len(c) == 4:
            return False
    return True

def greedy_coloring(order, rng):
    """
    Greedy: assign colors to edges in given order.
    For each edge, try colors from 0..119 in a shuffled order, pick the first
    that doesn't create a rainbow 4-cycle.
    """
    colors = [-1] * M
    # For fast checking: for each pair of edges, which cycles contain both?
    # Actually, when we assign a color to edge e, we need to check all cycles
    # containing e where the other 3 edges are already colored.
    # Precompute: for each edge e, list of cycles containing e, with the other 3 edges.
    
    # Build adjacency: edge -> list of (cycle_idx, other_edges_tuple)
    edge_cycles = [[] for _ in range(M)]
    for ci, (e0, e1, e2, e3) in enumerate(ALL_CYCLES):
        for pos, e in enumerate((e0, e1, e2, e3)):
            others = tuple(ALL_CYCLES[ci][p] for p in range(4) if p != pos)
            edge_cycles[e].append((ci, others))
    
    # Shuffle color order for diversity
    color_order = list(range(120))
    rng.shuffle(color_order)
    
    for e in order:
        placed = False
        for c in color_order:
            # Check all cycles containing e where other 3 edges are colored
            ok = True
            for ci, others in edge_cycles[e]:
                c0, c1, c2 = others
                if colors[c0] >= 0 and colors[c1] >= 0 and colors[c2] >= 0:
                    if len({c, colors[c0], colors[c1], colors[c2]}) == 4:
                        ok = False
                        break
            if ok:
                colors[e] = c
                placed = True
                break
        if not placed:
            # Should not happen with 120 colors, but fallback
            colors[e] = 0
    
    return colors

def local_improve(colors, rng, deadline):
    """
    Try to increase distinct colors by recoloring edges.
    Strategy: for each edge, try to change its color to a new (unused) color
    if it doesn't create a rainbow 4-cycle. Also try swapping colors between
    edges to free up a color.
    """
    # Precompute edge_cycles
    edge_cycles = [[] for _ in range(M)]
    for ci, (e0, e1, e2, e3) in enumerate(ALL_CYCLES):
        for pos, e in enumerate((e0, e1, e2, e3)):
            others = tuple(ALL_CYCLES[ci][p] for p in range(4) if p != pos)
            edge_cycles[e].append((ci, others))
    
    def can_assign(e, c):
        """Check if edge e can be assigned color c given current coloring."""
        for ci, others in edge_cycles[e]:
            c0, c1, c2 = others
            if colors[c0] >= 0 and colors[c1] >= 0 and colors[c2] >= 0:
                if len({c, colors[c0], colors[c1], colors[c2]}) == 4:
                    return False
        return True
    
    start_time = time.time()
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        used_colors = set(colors)
        unused_colors = [c for c in range(120) if c not in used_colors]
        rng.shuffle(unused_colors)
        edge_order = list(range(M))
        rng.shuffle(edge_order)
        
        for e in edge_order:
            if time.time() > deadline - 2.0:
                break
            old_c = colors[e]
            # Try to assign an unused color
            for new_c in unused_colors:
                if new_c == old_c:
                    continue
                if can_assign(e, new_c):
                    colors[e] = new_c
                    improved = True
                    break
    
    return colors

def swap_improve(colors, rng, deadline):
    """
    Try to increase distinct colors by finding a color that appears exactly once
    on some edge, and see if we can "free" that color by moving the other edges
    using that color to different colors, then assign the freed color to a new edge.
    """
    edge_cycles = [[] for _ in range(M)]
    for ci, (e0, e1, e2, e3) in enumerate(ALL_CYCLES):
        for pos, e in enumerate((e0, e1, e2, e3)):
            others = tuple(ALL_CYCLES[ci][p] for p in range(4) if p != pos)
            edge_cycles[e].append((ci, others))
    
    def can_assign(e, c):
        for ci, others in edge_cycles[e]:
            c0, c1, c2 = others
            if colors[c0] >= 0 and colors[c1] >= 0 and colors[c2] >= 0:
                if len({c, colors[c0], colors[c1], colors[c2]}) == 4:
                    return False
        return True
    
    start_time = time.time()
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        # Count color usage
        color_count = {}
        color_edges = {}
        for e in range(M):
            c = colors[e]
            color_count[c] = color_count.get(c, 0) + 1
            if c not in color_edges:
                color_edges[c] = []
            color_edges[c].append(e)
        
        used_colors = set(color_count.keys())
        unused_colors = [c for c in range(120) if c not in used_colors]
        
        # For each unused color, try to free it by recoloring edges that currently
        # use some other color. Actually, simpler: for each edge, try unused colors.
        # But that's what local_improve does. Let's try a 2-swap: 
        # Find edge e1 with color c1, edge e2 with color c2 (c1 != c2, both used)
        # Try swapping: e1 gets c2, e2 gets c1. This doesn't change distinct count.
        # Better: find an edge e with color c that appears >1 time. Try to recolor
        # e to an unused color u. If valid, distinct count increases by 1.
        
        rng.shuffle(unused_colors)
        edge_order = list(range(M))
        rng.shuffle(edge_order)
        
        for e in edge_order:
            if time.time() > deadline - 2.0:
                break
            old_c = colors[e]
            if color_count.get(old_c, 0) <= 1:
                continue  # This color would be lost if we recolor
            for new_c in unused_colors:
                if can_assign(e, new_c):
                    colors[e] = new_c
                    improved = True
                    break
            if improved:
                break
    
    return colors

def construct():
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    rng = random.Random(42)
    
    best_colors = None
    best_distinct = 0
    
    # Generate multiple edge orderings
    orderings = []
    # Natural order
    orderings.append(list(range(M)))
    # Random orderings
    for _ in range(5):
        o = list(range(M))
        rng.shuffle(o)
        orderings.append(o)
    # Reverse
    orderings.append(list(range(M-1, -1, -1)))
    # Sorted by vertex (edges incident to vertex 0 first, then 1, etc.)
    vert_order = []
    for v in range(N):
        for u in range(v+1, N):
            vert_order.append(get_idx(v, u))
    orderings.append(vert_order)
    # Reverse vertex order
    vert_order2 = []
    for v in range(N-1, -1, -1):
        for u in range(v+1, N):
            vert_order2.append(get_idx(v, u))
    orderings.append(vert_order2)
    
    for order in orderings:
        if time.time() > deadline - 5.0:
            break
        colors = greedy_coloring(order, rng)
        d = count_distinct_colors(colors)
        if d > best_distinct:
            best_distinct = d
            best_colors = colors[:]
        
        # Local improvement
        colors = local_improve(colors, rng, deadline)
        d = count_distinct_colors(colors)
        if d > best_distinct:
            best_distinct = d
            best_colors = colors[:]
        
        # Swap improvement
        colors = swap_improve(colors, rng, deadline)
        d = count_distinct_colors(colors)
        if d > best_distinct:
            best_distinct = d
            best_colors = colors[:]
    
    # Final local improvement on best
    if best_colors is not None and time.time() < deadline - 3.0:
        best_colors = local_improve(best_colors, rng, deadline)
        best_colors = swap_improve(best_colors, rng, deadline)
        best_distinct = count_distinct_colors(best_colors)
    
    if best_colors is None:
        best_colors = [0] * M
    
    return best_colors

# Quick self-test
if __name__ == "__main__":
    start = time.time()
    result = construct()
    elapsed = time.time() - start
    d = count_distinct_colors(result)
    valid = is_valid(result)
    print(f"Distinct colors: {d}, Valid: {valid}, Time: {elapsed:.2f}s")
