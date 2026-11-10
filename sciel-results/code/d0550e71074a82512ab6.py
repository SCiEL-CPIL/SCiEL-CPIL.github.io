"""Construct a coloring of K16 edges with as many distinct colors as possible,
subject to: no 4-cycle has 4 distinct edge colors (the 3 possible 4-cycles on
every 4-vertex subset are checked)."""
import time
import random

N = 16
E = N * (N - 1) // 2  # 120

# Precompute edge index map
def _edge_index():
    idx = {}
    k = 0
    for i in range(N):
        for j in range(i + 1, N):
            idx[(i, j)] = k
            k += 1
    return idx

EDGE_IDX = _edge_idx()
EDGE_LIST = []
for i in range(N):
    for j in range(i + 1, N):
        EDGE_LIST.append((i, j))

# For each vertex, list of (other_vertex, edge_index)
ADJ = [[] for _ in range(N)]
for (u, v), ei in EDGE_IDX.items():
    ADJ[u].append((v, ei))
    ADJ[v].append((u, ei))

# For each vertex, its incident edge indices
INC = [[] for _ in range(N)]
for (u, v), ei in EDGE_IDX.items():
    INC[u].append(ei)
    INC[v].append(ei)

# Precompute all 4-cycles: each is a set of 4 vertices {a,b,c,d}
# The three 4-cycles are: (a-b-c-d), (a-b-d-c), (a-c-b-d)
# i.e., the 3 perfect matchings of K4 give the 3 "opposite pairs".
# A 4-cycle on vertices {a,b,c,d} uses 4 edges. The 3 cycles are:
#   cycle1: a-b, b-c, c-d, d-a  (edges: (a,b),(b,c),(c,d),(d,a))
#   cycle2: a-b, b-d, d-c, c-a  (edges: (a,b),(b,d),(c,d),(a,c))
#   cycle3: a-c, c-b, b-d, d-a  (edges: (a,c),(b,c),(b,d),(a,d))
FOUR_CYCLES = []
def _add_cycle(e1, e2, e3, e4):
    FOUR_CYCLES.append((e1, e2, e3, e4))

for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            for d in range(c + 1, N):
                # Cycle 1: a-b-c-d-a
                _add_cycle(EDGE_IDX[(a, b)], EDGE_IDX[(b, c)], EDGE_IDX[(c, d)], EDGE_IDX[(a, d)])
                # Cycle 2: a-b-d-c-a
                _add_cycle(EDGE_IDX[(a, b)], EDGE_IDX[(b, d)], EDGE_IDX[(c, d)], EDGE_IDX[(a, c)])
                # Cycle 3: a-c-b-d-a
                _add_cycle(EDGE_IDX[(a, c)], EDGE_IDX[(b, c)], EDGE_IDX[(b, d)], EDGE_IDX[(a, d)])

# For each edge, list of cycles it belongs to (for fast validity)
EDGE_CYCLES = [[] for _ in range(E)]
for ci, cyc in enumerate(FOUR_CYCLES):
    for ei in cyc:
        EDGE_CYCLES[ei].append(ci)

def count_distinct(colors):
    return len(set(colors))

def is_valid(colors):
    """Check no 4-cycle has 4 distinct colors."""
    for c1, c2, c3, c4 in FOUR_CYCLES:
        vals = (colors[c1], colors[c2], colors[c3], colors[c4])
        if len(set(vals)) == 4:
            return False
    return True

def _cycle_has_rainbow(colors, cyc):
    return len(set(colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]])) == 4

def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin before 90s kill
    best = None
    best_score = 0

    rng = random.Random(12345)

    # --- Phase 1: Greedy randomized construction ---
    # Strategy: assign colors in a random edge order, trying to use a new color
    # when possible, else reusing an existing safe color.
    # We try many random seeds/orders.

    for attempt in range(400):
        if time.time() > deadline - 5.0:
            break

        # Random edge order
        order = list(range(E))
        rng.shuffle(order)

        colors = [-1] * E
        used_colors = set()
        # Track for each color, the set of edges using it (for fast check)
        # Actually for the "w" check we need: for color c, for each vertex w,
        # how many of w's incident edges use c. A new edge (u,v) with color c is
        # safe iff for every w not in {u,v}, w does not have 2 incident edges of
        # color c. Equivalently, w has at most 1 edge of color c among all its
        # incident edges EXCEPT we're adding (u,v) which is incident to u and v,
        # not w. So the condition is: for all w not in {u,v}, count_c[w] <= 1
        # where count_c[w] = number of incident edges of w currently colored c.

        # Maintain: for each color, an array of length N with counts per vertex.
        # To keep it fast, store color -> list of (u,v) edges, and a per-vertex count.
        color_edges = {}  # color -> list of edge indices
        color_vcount = {}  # color -> list of N counts

        # Order edges by a heuristic: maybe assign "hub" edges first? Let's just
        # do random order but with a twist: prioritize edges to spread colors.
        success = True
        for ei in order:
            u, v = EDGE_LIST[ei]
            # Try to assign a new color first (maximize distinct)
            placed = False
            # Try new colors from a random range to encourage diversity
            # We'll try a new color: pick a fresh color id
            # A new color is always safe (no existing edge uses it), so it's always valid!
            # Wait — a new color used only on (u,v): no other edge has it, so no
            # 4-cycle can have 4 distinct colors involving this edge with a 4th
            # distinct color? Actually a 4-cycle containing (u,v) has 3 other edges.
            # If (u,v) has a unique color, the 4-cycle has at most 3 other colors + 1
            # unique = up to 4 distinct. So a NEW color CAN create a rainbow!
            # So we must check.
            #
            # Condition for edge (u,v) with color c (possibly new):
            # For every 4-cycle containing edge (u,v), the other 3 edges must not
            # all have distinct colors different from c.
            #
            # Simpler global check: assign c, then check all cycles containing ei.
            #
            # To maximize distinct colors, we want to use a new color whenever safe.
            # Let's check if a new color is safe.

            # Check if assigning a brand-new color to (u,v) is safe.
            # A new color c_new appears only on (u,v). A 4-cycle containing (u,v)
            # becomes rainbow iff its other 3 edges have 3 distinct colors, all
            # different from c_new (they are, since c_new is new). So we need:
            # no 4-cycle containing (u,v) has its other 3 edges all distinct.
            safe_new = True
            for ci in EDGE_CYCLES[ei]:
                cyc = FOUR_CYCLES[ci]
                other = [x for x in cyc if x != ei]
                ov = (colors[other[0]], colors[other[1]], colors[other[2]])
                if len(set(ov)) == 3:
                    safe_new = False
                    break

            if safe_new:
                c_new = len(used_colors)
                colors[ei] = c_new
                used_colors.add(c_new)
                color_edges[c_new] = [ei]
                color_vcount[c_new] = [0] * N
                color_vcount[c_new][u] += 1
                color_vcount[c_new][v] += 1
                placed = True
            else:
                # Must reuse an existing color. Try existing colors in random order.
                existing = list(used_colors)
                rng.shuffle(existing)
                for c in existing:
                    # Check safety of reusing c on (u,v)
                    # For every w not in {u,v}: w must not have 2 incident edges of
                    # color c currently. Because adding (u,v) doesn't touch w.
                    # Also need: no 4-cycle containing (u,v) becomes rainbow.
                    # The 4-cycle check: other 3 edges + c. Rainbow iff other 3
                    # are distinct and none equals c.
                    ok = True
                    # per-vertex count check
                    cv = color_vcount[c]
                    for w in range(N):
                        if w == u or w == v:
                            continue
                        if cv[w] >= 2:
                            ok = False
                            break
                    if not ok:
                        continue
                    # cycle check
                    for ci in EDGE_CYCLES[ei]:
                        cyc = FOUR_CYCLES[ci]
                        other = [x for x in cyc if x != ei]
                        ov = (colors[other[0]], colors[other[1]], colors[other[2]])
                        # rainbow if 3 distinct and none == c
                        if len(set(ov)) == 3 and c not in ov:
                            ok = False
                            break
                    if ok:
                        colors[ei] = c
                        color_edges[c].append(ei)
                        cv[u] += 1
                        cv[v] += 1
                        placed = True
                        break
                if not placed:
                    success = False
                    break

        if success:
            # Verify
            if is_valid(colors):
                sc = count_distinct(colors)
                if sc > best_score:
                    best_score = sc
                    best = colors[:]

        # Early stop if we hit a high score
        if best_score >= 60:
            break

    # --- Phase 2: Local search polish (color swaps / recoloring) ---
    if best is not None:
        best = best[:]
        improve = True
        while improve and time.time() < deadline - 3.0:
            improve = False
            # Try to recolor edges to introduce new colors
            # For each edge, try assigning a new color if safe
            for ei in range(E):
                if time.time() > deadline - 3.0:
                    break
                u, v = EDGE_LIST[ei]
                # Try new color
                new_c = max(best) + 1 if best else 0
                # Check safety of new color on (u,v)
                safe = True
                for ci in EDGE_CYCLES[ei]:
                    cyc = FOUR_CYCLES[ci]
                    other = [x for x in cyc if x != ei]
                    ov = (best[other[0]], best[other[1]], best[other[2]])
                    if len(set(ov)) == 3:
                        safe = False
                        break
                if safe:
                    best[ei] = new_c
                    improve = True

            # Try swapping colors to free up: for each edge, try changing to an
            # existing color to enable a new color elsewhere. (Simpler: just
            # recolor to reduce conflicts isn't needed; we only add colors.)
            # Also try: for each edge, if it currently shares a color that blocks
            # new colors, reassign it to another existing safe color.
            # (This is more complex; skip for time.)

    if best is None:
        # Fallback: all zeros is valid
        return [0] * E

    return best
