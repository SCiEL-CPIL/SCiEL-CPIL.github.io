import time
import random

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120


def edge_index(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)


# Precompute edge endpoints and all 4-cycles (3 per 4-vertex subset)
_EDGE_ENDS = []
for _i in range(N):
    for _j in range(_i + 1, N):
        _EDGE_ENDS.append((_i, _j))

_CYCLES = []
for _a in range(N):
    for _b in range(_a + 1, N):
        for _c in range(_b + 1, N):
            for _d in range(_c + 1, N):
                _CYCLES.append((
                    edge_index(_a, _b), edge_index(_b, _c),
                    edge_index(_c, _d), edge_index(_d, _a)))
                _CYCLES.append((
                    edge_index(_a, _b), edge_index(_b, _d),
                    edge_index(_d, _c), edge_index(_c, _a)))
                _CYCLES.append((
                    edge_index(_a, _c), edge_index(_c, _b),
                    edge_index(_b, _d), edge_index(_d, _a)))

# Per-edge list of cycles (each cycle is a 4-tuple of edge indices)
_CYCLES_BY_EDGE = [[] for _ in range(NUM_EDGES)]
for _cy in _CYCLES:
    for _e in _cy:
        _CYCLES_BY_EDGE[_e].append(_cy)


def _valid_recolor(colors, e, nc):
    """Check whether recoloring edge e to nc keeps every 4-cycle rainbow-free."""
    for cy in _CYCLES_BY_EDGE[e]:
        # count distinct colors among the other 3 edges
        o1 = colors[cy[0]] if cy[0] != e else -1
        o2 = colors[cy[1]] if cy[1] != e else -1
        o3 = colors[cy[2]] if cy[2] != e else -1
        o4 = colors[cy[3]] if cy[3] != e else -1
        s = set()
        for v in (o1, o2, o3, o4):
            if v != -1:
                s.add(v)
        if len(s) == 3 and nc not in s:
            return False
    return True


def construct():
    deadline = time.time() + 50  # safety margin under the 60s/90s limits
    rng = random.Random(20240517)

    # ---- Structured start: assign a few base colors by (i+j) mod K,
    #      then greedily introduce as many new colors as possible. ----
    base = 3
    colors = [0] * NUM_EDGES
    for e in range(NUM_EDGES):
        i, j = _EDGE_ENDS[e]
        colors[e] = (i + j) % base
    used = set(colors)

    # Greedy: for each edge (most constrained first) try to give it a fresh color.
    edge_order = sorted(range(NUM_EDGES),
                        key=lambda e: len(_CYCLES_BY_EDGE[e]), reverse=True)
    for e in edge_order:
        for c in range(base, NUM_EDGES):
            if c in used:
                continue
            if _valid_recolor(colors, e, c):
                colors[e] = c
                used.add(c)
                break

    best = colors[:]
    best_count = len(set(colors))

    def color_count():
        return len(set(colors))

    def try_single_new_color():
        """Give some edge a brand-new color. Return True if improved."""
        nonlocal used
        for e in edge_order:
            for nc in range(NUM_EDGES):
                if nc in used:
                    continue
                if _valid_recolor(colors, e, nc):
                    colors[e] = nc
                    used.add(nc)
                    return True
        return False

    def try_two_opt():
        """Recolor one edge to a shared color to free a unique color,
        then place a new color on a second edge. Return True if improved."""
        nonlocal used
        counts = {}
        for c in colors:
            counts[c] = counts.get(c, 0) + 1
        unique = [e for e in range(NUM_EDGES) if counts[colors[e]] == 1]
        for e1 in unique:
            c1 = colors[e1]
            # target: recolor e1 to an existing color c2 (so c1 becomes free)
            for e2 in edge_order:
                if e2 == e1:
                    continue
                c2 = colors[e2]
                if c2 == c1:
                    continue
                if not _valid_recolor(colors, e1, c2):
                    continue
                # tentatively recolor e1 -> c2
                colors[e1] = c2
                # now try to place a fresh color on e2 (or any edge)
                placed = False
                for e3 in edge_order:
                    for nc in range(NUM_EDGES):
                        if nc == c1 or nc == c2 or nc in used:
                            continue
                        if _valid_recolor(colors, e3, nc):
                            colors[e3] = nc
                            used.discard(c1)
                            used.add(nc)
                            placed = True
                            break
                    if placed:
                        break
                if placed:
                    return True
                colors[e1] = c1  # revert
        return False

    def swap_move():
        """Swap colors of two edges (preserves multiset of colors)."""
        e1 = rng.randrange(NUM_EDGES)
        e2 = rng.randrange(NUM_EDGES)
        if e1 == e2:
            return False
        c1, c2 = colors[e1], colors[e2]
        if c1 == c2:
            return False
        colors[e1], colors[e2] = c2, c1
        ok1 = _valid_recolor(colors, e1, c2)
        ok2 = _valid_recolor(colors, e2, c1)
        if not (ok1 and ok2):
            colors[e1], colors[e2] = c1, c2
            return False
        return True

    improved = True
    stall = 0
    while time.time() < deadline and improved:
        improved = False
        # 1) aggressive new-color placement
        if try_single_new_color():
            improved = True
        # 2) two-opt to free a color and place a new one
        if try_two_opt():
            improved = True
        # 3) color-preserving swaps to reach a better region
        for _ in range(200):
            if time.time() >= deadline:
                break
            if swap_move():
                improved = True
        cc = color_count()
        if cc > best_count:
            best_count = cc
            best = colors[:]
            stall = 0
        else:
            stall += 1
        if stall >= 3:
            # perturbation: randomly recolor a few edges (may drop a color)
            for _ in range(rng.randint(3, 8)):
                e = rng.randrange(NUM_EDGES)
                nc = rng.randrange(NUM_EDGES)
                if nc == colors[e]:
                    continue
                if _valid_recolor(colors, e, nc):
                    colors[e] = nc
            improved = True
            stall = 0

    # ---- Final high-precision polish: keep trying new colors until stuck. ----
    while time.time() < deadline:
        if not try_single_new_color():
            break
        cc = color_count()
        if cc > best_count:
            best_count = cc
            best = colors[:]

    if color_count() > best_count:
        best = colors[:]

    return best
