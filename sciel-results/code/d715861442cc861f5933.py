import time
import random
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def _edge_idx(i, j):
    return i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)

# Precompute edge indices for all pairs
EDGES = []
for i in range(N):
    for j in range(i + 1, N):
        EDGES.append((i, j))

# Map edge index to (u, v)
EDGE_MAP = {idx: (i, j) for idx, (i, j) in enumerate(EDGES)}

# Precompute 4-cycles containing each edge
# A 4-cycle containing edge (u,v) is determined by two other vertices x, y.
# The cycle is u-v-x-y-u.
# Edges: (u,v), (v,x), (x,y), (y,u).
# We need the indices of these 4 edges.

EDGE_CYCLES = [[] for _ in range(NUM_EDGES)]

# For each edge e = (u,v), find all pairs (x,y) such that x,y != u,v and x < y
# The cycle is u-v-x-y-u.
# We need to ensure we don't duplicate cycles?
# The problem says "all three cycles on every four-vertex subset are checked".
# My check `len({c1,c2,c3,c4})==4` handles any specific 4-cycle.
# I need to check ALL 4-cycles that contain the edge.
# For a fixed edge (u,v), the 4-cycles containing it are formed by choosing 2 other vertices from the remaining 14.
# Let the other vertices be x, y.
# The cycle must use (u,v). The other edges must connect v to one of x,y, and u to the other, and x to y.
# Possible cycles:
# 1. u-v-x-y-u: Edges (u,v), (v,x), (x,y), (y,u)
# 2. u-v-y-x-u: Edges (u,v), (v,y), (y,x), (x,u) -> Same set of edges as 1?
#    (v,x) and (x,v) are same edge. (y,u) and (u,y) are same.
#    So for a set {u,v,x,y}, there is only ONE 4-cycle containing (u,v)?
#    Let's check.
#    Vertices u,v,x,y.
#    Cycles containing (u,v):
#    Path from v to u not using (u,v).
#    Paths of length 3 from v to u:
#    v-x-y-u (Edges vx, xy, yu)
#    v-y-x-u (Edges vy, yx, xu)
#    These are two distinct cycles if x != y.
#    So for each pair {x,y}, there are TWO 4-cycles containing (u,v).
#    Wait, the cycle v-x-y-u uses edges (v,x), (x,y), (y,u).
#    The cycle v-y-x-u uses edges (v,y), (y,x), (x,u).
#    Are these the same 4-cycle? No, they are different cycles on the same 4 vertices.
#    So for each pair {x,y}, we have 2 cycles.
#    Total cycles per edge: 14 * 13 / 2 * 2 = 182.
#    Wait, $\binom{14}{2} = 91$ pairs. $91 \times 2 = 182$ cycles.
#    Let's verify total cycles in K16.
#    Total 4-cycles in K16: $\binom{16}{4} \times 3 = 1820 \times 3 = 5460$.
#    Each cycle has 4 edges.
#    Total edge-cycle incidences: $5460 \times 4 = 21840$.
#    Total edges: 120.
#    Incidences per edge: $21840 / 120 = 182$.
#    Matches.

for e_idx, (u, v) in enumerate(EDGES):
    other_vertices = [x for x in range(N) if x != u and x != v]
    for x, y in combinations(other_vertices, 2):
        # Cycle 1: u-v-x-y-u
        # Edges: (u,v), (v,x), (x,y), (y,u)
        e1 = _edge_idx(u, v)
        e2 = _edge_idx(v, x) if v < x else _edge_idx(x, v)
        e3 = _edge_idx(x, y) if x < y else _edge_idx(y, x)
        e4 = _edge_idx(y, u) if y < u else _edge_idx(u, y)
        EDGE_CYCLES[e_idx].append((e1, e2, e3, e4))
        
        # Cycle 2: u-v-y-x-u
        # Edges: (u,v), (v,y), (y,x), (x,u)
        f1 = _edge_idx(u, v)
        f2 = _edge_idx(v, y) if v < y else _edge_idx(y, v)
        f3 = _edge_idx(y, x) if y < x else _edge_idx(x, y)
        f4 = _edge_idx(x, u) if x < u else _edge_idx(u, x)
        EDGE_CYCLES[e_idx].append((f1, f2, f3, f4))

def construct():
    deadline = time.time() + 55  # Safety margin
    best_colors = [0] * NUM_EDGES
    best_count = 1
    
    def check_validity(colors, edge_idx, new_color):
        # Check all cycles containing edge_idx
        for cyc in EDGE_CYCLES[edge_idx]:
            c0 = colors[cyc[0]] if cyc[0] != edge_idx else new_color
            c1 = colors[cyc[1]] if cyc[1] != edge_idx else new_color
            c2 = colors[cyc[2]] if cyc[2] != edge_idx else new_color
            c3 = colors[cyc[3]] if cyc[3] != edge_idx else new_color
            if len({c0, c1, c2, c3}) == 4:
                return False
        return True

    # Greedy construction with multiple restarts
    while time.time() < deadline:
        colors = [0] * NUM_EDGES
        used_colors = {0}
        next_color = 1
        
        # Phase 1: Greedy assignment of new colors
        # We iterate through edges multiple times to allow late edges to take new colors
        # that might have been blocked earlier by different assignments
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle edges for randomization
            edge_order = list(range(NUM_EDGES))
            random.shuffle(edge_order)
            
            for e in edge_order:
                # Try to assign next_color
                if check_validity(colors, e, next_color):
                    colors[e] = next_color
                    used_colors.add(next_color)
                    next_color += 1
                    improved = True
        
        # Phase 2: Local Search / Hill Climbing
        # Try to swap colors or introduce new colors on specific edges
        # This is optional if Phase 1 is strong enough, but helps break plateaus
        # Given time, we might skip complex local search and just do more restarts.
        
        current_count = len(used_colors)
        if current_count > best_count:
            best_count = current_count
            best_colors = colors[:]
            
        # If we hit a high number, we can try to optimize further
        # But for now, restarts are fast.

    return best_colors
