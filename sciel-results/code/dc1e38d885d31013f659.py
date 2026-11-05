"""Rewritten program: greedy + local search for max distinct colors on C4-free edge coloring."""
import time
import random
from itertools import combinations

N = 16
M = N * (N - 1) // 2  # 120 edges

# Precompute edge indices in lexicographic order
_EDGE_LIST = [(i, j) for i in range(N) for j in range(i + 1, N)]
_EDGE_IDX = {}
for _k, (_i, _j) in enumerate(_EDGE_LIST):
    _EDGE_IDX[(_i, _j)] = _k

def _get_edge_idx(u, v):
    if u > v:
        u, v = v, u
    return _EDGE_IDX[(u, v)]

# Precompute all 4-cycles (sets of 4 edge indices)
# A 4-cycle on vertices {a,b,c,d} has 3 possible cycle edge-sets:
#   (a,b), (b,c), (c,d), (d,a)
#   (a,b), (b,d), (d,c), (c,a)
#   (a,c), (c,b), (b,d), (d,a)
_FOUR_CYCLES = []
_seen_cycles = set()
for quad in combinations(range(N), 4):
    a, b, c, d = quad
    for cyc in [
        [(a, b), (b, c), (c, d), (d, a)],
        [(a, b), (b, d), (d, c), (c, a)],
        [(a, c), (c, b), (b, d), (d, a)],
    ]:
        idx_tuple = tuple(sorted(_get_edge_idx(u, v) for u, v in cyc))
        if idx_tuple not in _seen_cycles:
            _seen_cycles.add(idx_tuple)
            _FOUR_CYCLES.append(idx_tuple)

# For each edge, precompute the list of 4-cycles it participates in
_EDGE_CYCLES = [[] for _ in range(M)]
for _cyc in _FOUR_CYCLES:
    for _e in _cyc:
        _EDGE_CYCLES[_e].append(_cyc)

# For fast rainbow check: given a partial coloring, check if adding
# color c to edge e creates a rainbow 4-cycle.
# A 4-cycle (e1,e2,e3,e4) is rainbow if all four edges have distinct colors.
# When we assign color c to edge e, a cycle containing e becomes rainbow iff
#   - the other 3 edges all have colors already assigned
#   - those 3 colors are all distinct from each other and from c

def _is_rainbow_conflict(coloring, edge_idx, new_color):
    """Check if assigning new_color to edge_idx creates any rainbow 4-cycle."""
    for cyc in _EDGE_CYCLES[edge_idx]:
        other_colors = []
        all_assigned = True
        for e in cyc:
            if e == edge_idx:
                continue
            c = coloring[e]
            if c == -1:
                all_assigned = False
                break
            other_colors.append(c)
        if not all_assigned:
            continue
        # Check if {new_color} + other_colors has 4 distinct values
        if len(set(other_colors + [new_color])) == 4:
            return True
    return False


def _count_distinct(coloring):
    return len(set(c for c in coloring if c != -1))


def _validate(coloring):
    """Full validation: no 4-cycle has 4 distinct colors."""
    for cyc in _FOUR_CYCLES:
        colors = [coloring[e] for e in cyc]
        if -1 not in colors and len(set(colors)) == 4:
            return False
    return True


def _greedy_construct():
    """Greedy: process edges in order, assign smallest valid new color."""
    coloring = [-1] * M
    # We want to maximize distinct colors. Strategy: for each edge,
    # first try to use a color not yet used (to increase palette),
    # then fall back to any valid color.
    used_colors = set()
    for e in range(M):
        # Try to find a color that doesn't create a rainbow 4-cycle
        # Prefer colors not yet in used_colors to expand palette
        found = False
        # First pass: try new colors (not in used_colors)
        for c in range(120):
            if c in used_colors:
                continue
            if not _is_rainbow_conflict(coloring, e, c):
                coloring[e] = c
                used_colors.add(c)
                found = True
                break
        if not found:
            # Second pass: try already-used colors
            for c in range(120):
                if c not in used_colors:
                    continue
                if not _is_rainbow_conflict(coloring, e, c):
                    coloring[e] = c
                    found = True
                    break
        if not found:
            # Shouldn't happen with 120 colors, but fallback
            coloring[e] = 0
            used_colors.add(0)
    return coloring


def _local_search(coloring, deadline):
    """Try to improve by reassigning edges to new colors."""
    coloring = list(coloring)
    best_score = _count_distinct(coloring)
    start = time.time()
    
    # Try to change each edge to a new color
    for _iteration in range(50):
        if time.time() > deadline:
            break
        improved = False
        for e in range(M):
            if time.time() > deadline:
                return coloring, best_score
            old_c = coloring[e]
            # Try to assign a color not currently used by any other edge
            # that would increase the distinct count
            for new_c in range(120):
                if new_c == old_c:
                    continue
                # Check if new_c is used elsewhere
                # (we could optimize this, but for now just check validity)
                if _is_rainbow_conflict(coloring, e, new_c):
                    continue
                # Tentatively assign
                coloring[e] = new_c
                new_score = _count_distinct(coloring)
                if new_score > best_score:
                    best_score = new_score
                    improved = True
                    break
                else:
                    coloring[e] = old_c
        if not improved:
            break
    return coloring, best_score


# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin within 90s limit
    
    # Phase 1: Greedy construction
    coloring = _greedy_construct()
    score = _count_distinct(coloring)
    
    # Validate
    if not _validate(coloring):
        # Fallback: all zeros is always valid
        return [0] * M
    
    # Phase 2: Local search to try to improve
    if time.time() < deadline:
        coloring, score = _local_search(coloring, deadline)
    
    # Final validation
    if not _validate(coloring):
        return [0] * M
    
    return coloring
# EVOLVE-BLOCK-END
