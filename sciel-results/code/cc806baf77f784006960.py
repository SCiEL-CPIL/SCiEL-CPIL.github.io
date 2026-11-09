"""Gallai coloring of K_16 maximizing distinct edge colors, with local-search polish."""
import time
import random

N = 16
NUM_EDGES = N * (N - 1) // 2

def _edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def _build_edge_map():
    """Return list of (i,j) for each edge index in lexicographic order."""
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((i, j))
    return edges

def _build_cycles_per_edge():
    """
    For each edge e, build list of 3-tuples (e1, e2, e3) such that {e, e1, e2, e3}
    forms a 4-cycle (i.e., the four edges form a cycle on 4 vertices).
    A 4-cycle on vertices {a,b,c,d} has edges (a,b),(b,c),(c,d),(d,a) — but in K_n,
    any 4 vertices form 3 distinct 4-cycles (the three ways to pair them into opposite edges).
    Actually, for 4 vertices {a,b,c,d}, the three 4-cycles are:
      (a,b,c,d), (a,b,d,c), (a,c,b,d) — but as edge sets:
      Cycle 1: {ab, bc, cd, da}
      Cycle 2: {ab, bd, dc, ca}
      Cycle 3: {ac, cb, bd, da}
    Each edge belongs to multiple 4-cycles.
    
    For edge (u,v), a 4-cycle containing it uses two more vertices x,y such that
    the cycle is u-v-x-y-u or u-v-y-x-u. The four edges are (u,v), (v,x), (x,y), (y,u)
    or (u,v), (v,y), (y,x), (x,u).
    """
    edges = _build_edge_map()
    edge_to_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_to_idx[(i, j)] = idx
        edge_to_idx[(j, i)] = idx
    
    cycles_per_edge = [[] for _ in range(NUM_EDGES)]
    
    for u in range(N):
        for v in range(u + 1, N):
            e_uv = edge_to_idx[(u, v)]
            # Choose two more vertices x, y from remaining N-2 vertices
            remaining = [w for w in range(N) if w != u and w != v]
            for xi in range(len(remaining)):
                for yi in range(xi + 1, len(remaining)):
                    x = remaining[xi]
                    y = remaining[yi]
                    # Two possible 4-cycles through edge (u,v) with vertices x,y:
                    # Cycle A: u-v-x-y-u  => edges (u,v),(v,x),(x,y),(y,u)
                    # Cycle B: u-v-y-x-u  => edges (u,v),(v,y),(y,x),(x,u)
                    # But these are the same cycle traversed differently? No:
                    # Cycle A edges: uv, vx, xy, yu
                    # Cycle B edges: uv, vy, yx, xu
                    # These are different edge sets unless {vx,xy,yu} == {vy,yx,xu}
                    # which happens when x and y are symmetric. Let's just add both and dedup.
                    
                    for (e1, e2, e3) in [
                        (edge_to_idx[(v, x)], edge_to_idx[(x, y)], edge_to_idx[(y, u)]),
                        (edge_to_idx[(v, y)], edge_to_idx[(y, x)], edge_to_idx[(x, u)]),
                    ]:
                        cycles_per_edge[e_uv].append((e1, e2, e3))
    
    # Deduplicate: for each edge, remove duplicate 3-tuples
    for e in range(NUM_EDGES):
        seen = set()
        unique = []
        for t in cycles_per_edge[e]:
            key = tuple(sorted(t))
            if key not in seen:
                seen.add(key)
                unique.append(t)
        cycles_per_edge[e] = unique
    
    return cycles_per_edge

def _count_distinct(colors):
    return len(set(colors))

def _validate(colors):
    """Check no 4-cycle has 4 distinct colors. Returns True if valid."""
    edges = _build_edge_map()
    edge_to_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_to_idx[(i, j)] = idx
        edge_to_idx[(j, i)] = idx
    
    # Check all 4-vertex subsets and all 3 cycles on each
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # Three 4-cycles on {a,b,c,d}
                    cycles = [
                        [(a, b), (b, c), (c, d), (d, a)],
                        [(a, b), (b, d), (d, c), (c, a)],
                        [(a, c), (c, b), (b, d), (d, a)],
                    ]
                    for cyc in cycles:
                        cols = set()
                        for (u, v) in cyc:
                            cols.add(colors[edge_to_idx[(u, v)]])
                        if len(cols) == 4:
                            return False
    return True

def _construct_base():
    """Fixed Gallai coloring with 20 distinct colors."""
    colors = [0] * NUM_EDGES
    color_counter = [0]

    def get_new_color():
        c = color_counter[0]
        color_counter[0] += 1
        return c

    def color_all_distinct(vertices):
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                u, v = vertices[i], vertices[j]
                colors[_edge_idx(u, v)] = get_new_color()

    def color_node(part1, part2):
        c = get_new_color()
        for u in part1:
            for v in part2:
                colors[_edge_idx(u, v)] = c

    leaves = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [9, 10, 11],
        [12, 13, 14],
        [15]
    ]

    for leaf in leaves:
        if len(leaf) >= 2:
            color_all_distinct(leaf)

    color_node(leaves[0], leaves[1])
    color_node(leaves[2], leaves[3])
    part2a = leaves[0] + leaves[1]
    part2b = leaves[2] + leaves[3]
    color_node(part2a, part2b)
    color_node(leaves[4], leaves[5])
    part_root_a = leaves[0] + leaves[1] + leaves[2] + leaves[3]
    part_root_b = leaves[4] + leaves[5]
    color_node(part_root_a, part_root_b)

    return colors

def _try_local_search(colors, cycles_per_edge, deadline):
    """
    Try to increase distinct color count by reassigning individual edges to new colors.
    Only accept a change if it doesn't create a rainbow 4-cycle.
    """
    best_colors = list(colors)
    best_count = _count_distinct(best_colors)
    
    # Precompute: for each edge, which new colors (not currently used) would be safe?
    # We'll iterate: pick an edge, try a new color, check validity.
    
    used_colors = set(best_colors)
    max_color = max(best_colors) if best_colors else 0
    next_new_color = max_color + 1
    
    # For efficiency, track for each edge the set of colors that would be invalid
    # Actually, just do direct check.
    
    edge_order = list(range(NUM_EDGES))
    random.shuffle(edge_order)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for e in edge_order:
            if time.time() >= deadline:
                break
            current_color = best_colors[e]
            # Try assigning a new color
            new_color = next_new_color
            # Check if this change would create a rainbow 4-cycle
            valid = True
            for (e1, e2, e3) in cycles_per_edge[e]:
                c1 = best_colors[e1]
                c2 = best_colors[e2]
                c3 = best_colors[e3]
                if len({new_color, c1, c2, c3}) == 4:
                    valid = False
                    break
            if valid:
                best_colors[e] = new_color
                next_new_color += 1
                best_count += 1
                improved = True
    
    return best_colors, best_count

def construct():
    """
    Construct a Gallai coloring of K_16 maximizing distinct edge colors.
    Starts with a fixed 20-color base and attempts local-search improvement.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Base coloring
    colors = _construct_base()
    base_count = _count_distinct(colors)
    
    # Build cycle structures for local search
    cycles_per_edge = _build_cycles_per_edge()
    
    # Verify base is valid
    if not _validate(colors):
        # Should not happen, but fallback
        return colors
    
    # Local search
    try:
        improved_colors, improved_count = _try_local_search(colors, cycles_per_edge, deadline)
        if improved_count > base_count:
            # Final validation
            if _validate(improved_colors):
                return improved_colors
    except Exception:
        pass
    
    return colors
