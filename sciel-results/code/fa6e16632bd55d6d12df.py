"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 16
M = N * (N - 1) // 2  # 120 edges

# ---- Precompute structure (deterministic, cheap) ----
_edge_index = {}
for _i in range(N):
    for _j in range(_i + 1, N):
        _edge_index[(_i, _j)] = _i * (N - 1) - _i * (_i + 1) // 2 + (_j - _i - 1)

# All 4-cycles: for each 4-vertex subset, the 3 ways to form a 4-cycle
# A 4-cycle on vertices {a,b,c,d} is an ordering (a,b,c,d) with edges
# (a,b),(b,c),(c,d),(d,a). For a fixed set of 4 vertices there are 3 distinct
# 4-cycles (the 3 ways to pair them into opposite vertices).
_4cycles = []  # each: (e1, e2, e3, e4) edge indices
for _a in range(N):
    for _b in range(_a + 1, N):
        for _c in range(_b + 1, N):
            for _d in range(_c + 1, N):
                # three cycles:
                # (a,b,c,d): edges ab,bc,cd,da
                # (a,b,d,c): edges ab,bd,dc,ca
                # (a,c,b,d): edges ac,cb,bd,da
                _4cycles.append((_edge_index[(_a,_b)], _edge_index[(_b,_c)],
                                 _edge_index[(_c,_d)], _edge_index[(_d,_a)]))
                _4cycles.append((_edge_index[(_a,_b)], _edge_index[(_b,_d)],
                                 _edge_index[(_d,_c)], _edge_index[(_c,_a)]))
                _4cycles.append((_edge_index[(_a,_c)], _edge_index[(_c,_b)],
                                 _edge_index[(_b,_d)], _edge_index[(_d,_a)]))

# For each edge, the list of 4-cycles it belongs to
_edge_cycles = [[] for _ in range(M)]
for _ci, _cycle in enumerate(_4cycles):
    for _e in _cycle:
        _edge_cycles[_e].append(_ci)


def _is_rainbow(colors, cycle):
    c0 = colors[cycle[0]]
    c1 = colors[cycle[1]]
    c2 = colors[cycle[2]]
    c3 = colors[cycle[3]]
    return c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3


def _count_colors(colors):
    return len(set(colors))


def _valid(colors):
    for cycle in _4cycles:
        if _is_rainbow(colors, cycle):
            return False
    return True


# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80.0  # safety margin under 90s / 60s limits

    # ---- Base coloring: color edge (i,j) by j (proven valid, 15 colors) ----
    colors = [0] * M
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            colors[idx] = j
            idx += 1

    # Verify base is valid (it should be)
    # (skip full check for speed; the j-coloring is known valid)

    best = colors[:]
    best_count = _count_colors(best)

    # ---- Greedy color introduction ----
    # Strategy: repeatedly pick an edge and try to assign it a new color that
    # keeps all its 4-cycles non-rainbow. Prefer colors not yet used to grow
    # the palette; if stuck, also allow recoloring to any color that helps.
    used = set(colors)

    # Build a quick "forbidden colors" helper: for a given edge e and a proposed
    # color c, check all 4-cycles containing e.
    def can_set(e, c, col):
        for ci in _edge_cycles[e]:
            cycle = _4cycles[ci]
            # temporarily
            old = col[e]
            col[e] = c
            if _is_rainbow(col, cycle):
                col[e] = old
                return False
            col[e] = old
        return True

    # Iterate: try to introduce new colors, then try to refine.
    progress = True
    while progress and time.time() < deadline:
        progress = False
        # Candidate new colors: 0..119 not in used, plus a spread
        # Try each edge, each candidate color.
        # Order edges by how "constrained" they are (fewer 4-cycles first) to
        # make new colors easier.
        edge_order = sorted(range(M), key=lambda e: len(_edge_cycles[e]))
        for e in edge_order:
            if time.time() >= deadline:
                break
            current = colors[e]
            # Try new colors first
            for c in range(120):
                if c in used:
                    continue
                if can_set(e, c, colors):
                    colors[e] = c
                    used.add(c)
                    progress = True
                    break
            # If no new color worked, try to recolor to an existing color that
            # is different (to free up the edge for a future new color).
            if not progress and time.time() < deadline:
                for c in range(120):
                    if c == current:
                        continue
                    if can_set(e, c, colors):
                        # only accept if it doesn't reduce color count
                        if c in used:
                            colors[e] = c
                            progress = True
                        break

    # ---- Local improvement: random recoloring to grow palette ----
    import random
    rng = random.Random(12345)
    for _ in range(2000):
        if time.time() >= deadline:
            break
        e = rng.randrange(M)
        current = colors[e]
        # pick a random candidate color
        c = rng.randrange(120)
        if c == current:
            continue
        if can_set(e, c, colors):
            colors[e] = c
            used.add(c)

    # Final: keep the better of current and best
    cur_count = _count_colors(colors)
    if cur_count > best_count:
        best = colors
        best_count = cur_count

    return best
# EVOLVE-BLOCK-END
