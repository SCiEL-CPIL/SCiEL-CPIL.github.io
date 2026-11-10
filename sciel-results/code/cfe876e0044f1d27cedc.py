import time
import random
import numpy as np
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 119
NUM_COLORS = MAX_COLOR + 1  # 120

def _edge_index(i, j):
    """Map pair (i,j) with i<j to edge index 0..119 in lexicographic order."""
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def _build_cycles():
    """Return list of cycles, each cycle is a tuple of 4 edge indices."""
    cycles = []
    for a in range(N - 3):
        for b in range(a + 1, N - 2):
            for c in range(b + 1, N - 1):
                for d in range(c + 1, N):
                    # Cycle 1: a-b-c-d-a (edges: ab, bc, cd, da)
                    cycles.append((_edge_index(a, b), _edge_index(b, c), _edge_index(c, d), _edge_index(a, d)))
                    # Cycle 2: a-b-d-c-a (edges: ab, bd, dc, ca)
                    cycles.append((_edge_index(a, b), _edge_index(b, d), _edge_index(c, d), _edge_index(a, c)))
                    # Cycle 3: a-c-b-d-a (edges: ac, cb, bd, da)
                    cycles.append((_edge_index(a, c), _edge_index(b, c), _edge_index(b, d), _edge_index(a, d)))
    return cycles

CYCLES = _build_cycles()
NUM_CYCLES = len(CYCLES)

def _build_edge_to_cycles():
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cyc in enumerate(CYCLES):
        for e in cyc:
            edge_to_cycles[e].append(ci)
    return edge_to_cycles

EDGE_TO_CYCLES = _build_edge_to_cycles()

def _is_rainbow(cyc, colors):
    """Check if the cycle has 4 distinct colors."""
    c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
    return c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3

def _validate(colors):
    """Full validation of a coloring."""
    for cyc in CYCLES:
        if _is_rainbow(cyc, colors):
            return False
    return True

def _count_distinct(colors):
    return len(set(colors))

def _greedy_seed(seed):
    """Build a valid coloring using greedy color introduction."""
    rng = random.Random(seed)
    colors = np.zeros(NUM_EDGES, dtype=np.int32)
    color_counts = np.zeros(NUM_COLORS, dtype=np.int32)
    color_counts[0] = NUM_EDGES

    def _is_move_valid(e, new_color, colors_arr):
        old_color = colors_arr[e]
        if old_color == new_color:
            return True
        for ci, pos, others in EDGE_CYCLE_STRUCT[e]:
            nc0 = new_color if pos == 0 else colors_arr[CYCLES[ci][0]]
            nc1 = new_color if pos == 1 else colors_arr[CYCLES[ci][1]]
            nc2 = new_color if pos == 2 else colors_arr[CYCLES[ci][2]]
            nc3 = new_color if pos == 3 else colors_arr[CYCLES[ci][3]]
            if nc0 != nc1 and nc0 != nc2 and nc0 != nc3 and nc1 != nc2 and nc1 != nc3 and nc2 != nc3:
                return False
        return True

    # Introduce colors 1..119 one at a time
    for new_c in range(1, MAX_COLOR + 1):
        edge_order = list(range(NUM_EDGES))
        rng.shuffle(edge_order)
        for e in edge_order:
            if _is_move_valid(e, new_c, colors):
                old = colors[e]
                colors[e] = new_c
                color_counts[old] -= 1
                color_counts[new_c] += 1
                break
    return colors, color_counts, None

def _build_constraint_structures():
    """Precompute constraint structures for fast validity checking.
    
    For each edge e and each cycle c containing e at position pos:
    - other_edges: the 3 other edges in the cycle
    - We track for each (e, color) how many cycles would become rainbow if e had that color.
    """
    # For each edge, list of (cycle_idx, position, other_edge_indices)
    edge_cycle_struct = [[] for _ in range(NUM_EDGES)]
    for ci, cyc in enumerate(CYCLES):
        for pos in range(4):
            e = cyc[pos]
            others = [cyc[k] for k in range(4) if k != pos]
            edge_cycle_struct[e].append((ci, pos, others))
    return edge_cycle_struct

EDGE_CYCLE_STRUCT = _build_constraint_structures()

def _compute_forbidden_counts(colors):
    """Compute for each edge e and color c, the number of cycles that would be 
    rainbow if edge e had color c. A move is valid iff forbidden_count[e][c] == 0."""
    # forbidden[e][c] = number of cycles containing e where the other 3 edges have 
    # 3 distinct colors all different from c
    forbidden = np.zeros((NUM_EDGES, NUM_COLORS), dtype=np.int32)
    
    for e in range(NUM_EDGES):
        for ci, pos, others in EDGE_CYCLE_STRUCT[e]:
            c0, c1, c2 = colors[others[0]], colors[others[1]], colors[others[2]]
            if c0 != c1 and c0 != c2 and c1 != c2:
                # All three other colors are distinct
                # Any color c not in {c0, c1, c2} would make this cycle rainbow
                # So forbidden[e][c] += 1 for all c not in {c0, c1, c2}
                # Equivalently: forbidden[e][*] += 1, then forbidden[e][c0] -= 1, etc.
                # But that's expensive. Instead, we can note that the "allowed" colors
                # are exactly {c0, c1, c2} when the three others are distinct.
                # So we mark that this cycle constrains e to use one of {c0, c1, c2}.
                pass
    
    # Alternative approach: for each edge, compute the set of colors that are "safe"
    # A color c is safe for edge e if for every cycle containing e, the other 3 edges
    # don't all have distinct colors different from c.
    # Equivalently: c is safe iff for every cycle with e, at least two of the other 
    # 3 edges share a color with each other or with c.
    
    # Let's use a different formulation:
    # For edge e in cycle with others (o1, o2, o3), color c is forbidden by this cycle
    # iff colors[o1], colors[o2], colors[o3] are all distinct AND none equals c.
    # So c is forbidden by this cycle iff {colors[o1], colors[o2], colors[o3]} is a set 
    # of 3 distinct elements not containing c.
    
    # For each edge e, the set of safe colors is the intersection over all cycles 
    # containing e of: (if the 3 others are distinct, then {those 3 colors}, else all colors)
    
    safe_sets = [None] * NUM_EDGES  # safe_sets[e] = set of safe colors, or None if all colors safe
    
    for e in range(NUM_EDGES):
        safe = None
        for ci, pos, others in EDGE_CYCLE_STRUCT[e]:
            c0, c1, c2 = colors[others[0]], colors[others[1]], colors[others[2]]
            if c0 != c1 and c0 != c2 and c1 != c2:
                # This cycle constrains e to use one of {c0, c1, c2}
                constraint = {c0, c1, c2}
                if safe is None:
                    safe = constraint
                else:
                    safe = safe & constraint
                if not safe:
                    break
        safe_sets[e] = safe
    
    return safe_sets

def _local_search(colors, color_counts, edge_cycle_info, deadline, rng):
    """Local search maximizing distinct colors while keeping validity.
    
    Uses precomputed safe-set structure for O(1) candidate generation."""
    best_colors = colors.copy()
    best_distinct = _count_distinct(colors)
    current_distinct = best_distinct
    
    # Precompute initial safe sets
    safe_sets = _compute_forbidden_counts(colors)
    
    def _is_move_valid(e, new_color, colors_arr):
        """Check if assigning new_color to edge e keeps all cycles valid."""
        old_color = colors_arr[e]
        if old_color == new_color:
            return True
        for ci, pos, others in EDGE_CYCLE_STRUCT[e]:
            nc0 = new_color if pos == 0 else colors_arr[CYCLES[ci][0]]
            nc1 = new_color if pos == 1 else colors_arr[CYCLES[ci][1]]
            nc2 = new_color if pos == 2 else colors_arr[CYCLES[ci][2]]
            nc3 = new_color if pos == 3 else colors_arr[CYCLES[ci][3]]
            if nc0 != nc1 and nc0 != nc2 and nc0 != nc3 and nc1 != nc2 and nc1 != nc3 and nc2 != nc3:
                return False
        return True
    
    def _update_safe_sets(e_old, old_c, new_c):
        """Update safe sets after changing edge e_old from old_c to new_c."""
        # Only edges in cycles with e_old need updating
        affected = set()
        for ci, pos, others in EDGE_CYCLE_STRUCT[e_old]:
            for oe in others:
                affected.add(oe)
        
        for e in affected:
            # Recompute safe set for e
            safe = None
            for ci2, pos2, others2 in EDGE_CYCLE_STRUCT[e]:
                c0, c1, c2 = colors[others2[0]], colors[others2[1]], colors[others2[2]]
                if c0 != c1 and c0 != c2 and c1 != c2:
                    constraint = {c0, c1, c2}
                    if safe is None:
                        safe = constraint
                    else:
                        safe = safe & constraint
                    if not safe:
                        break
            safe_sets[e] = safe
    
    while time.time() < deadline:
        edge_order = list(range(NUM_EDGES))
        rng.shuffle(edge_order)
        improved = False

        for e in edge_order:
            if time.time() >= deadline:
                break
            old_color = colors[e]
            
            # Get safe colors for this edge
            safe = safe_sets[e]
            if safe is not None:
                candidates = [c for c in safe if c != old_color]
                if not candidates:
                    continue
            else:
                # No constraint; prefer unused then under-represented colors
                unused = [c for c in range(NUM_COLORS) if color_counts[c] == 0]
                if unused:
                    rng.shuffle(unused)
                    candidates = unused[:15]
                else:
                    present = [c for c in range(NUM_COLORS) if color_counts[c] > 0]
                    present.sort(key=lambda c: color_counts[c])
                    candidates = present[:15]
                if old_color in candidates:
                    candidates.remove(old_color)
                if not candidates:
                    continue

            new_color = rng.choice(candidates)

            # Quick validity check (should be fast due to safe sets)
            if not _is_move_valid(e, new_color, colors):
                continue

            # Apply
            color_counts[old_color] -= 1
            color_counts[new_color] += 1
            colors[e] = new_color
            _update_safe_sets(e, old_color, new_color)

            new_distinct = current_distinct
            if color_counts[old_color] == 0:
                new_distinct -= 1
            if color_counts[new_color] == 1:
                new_distinct += 1

            if new_distinct >= current_distinct:
                current_distinct = new_distinct
                if new_distinct > best_distinct:
                    best_distinct = new_distinct
                    best_colors = colors.copy()
                    improved = True
            else:
                # Revert
                colors[e] = old_color
                color_counts[old_color] += 1
                color_counts[new_color] -= 1
                _update_safe_sets(e, new_color, old_color)

        if not improved:
            # Perturbation to escape local minimum
            for _ in range(rng.randint(2, 6)):
                e = rng.randint(0, NUM_EDGES - 1)
                old = colors[e]
                new = rng.randint(0, MAX_COLOR)
                if new == old:
                    continue
                if _is_move_valid(e, new, colors):
                    color_counts[old] -= 1
                    color_counts[new] += 1
                    colors[e] = new
                    _update_safe_sets(e, old, new)

    return best_colors, best_distinct

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s hard kill

    best_overall_colors = None
    best_overall_distinct = 0

    num_restarts = 0
    while time.time() < deadline - 3.0:
        seed = 1000 + num_restarts
        colors, color_counts, _ = _greedy_seed(seed)

        local_deadline = min(time.time() + 10.0, deadline - 2.0)
        rng = random.Random(seed + 7777)

        # Small initial perturbation for diversity
        for _ in range(rng.randint(0, 5)):
            e = rng.randint(0, NUM_EDGES - 1)
            new_c = rng.randint(0, MAX_COLOR)
            old = colors[e]
            if new_c == old:
                continue
            # Quick validity check
            ok = True
            for ci, pos, others in EDGE_CYCLE_STRUCT[e]:
                nc0 = new_c if pos == 0 else colors[CYCLES[ci][0]]
                nc1 = new_c if pos == 1 else colors[CYCLES[ci][1]]
                nc2 = new_c if pos == 2 else colors[CYCLES[ci][2]]
                nc3 = new_c if pos == 3 else colors[CYCLES[ci][3]]
                if nc0 != nc1 and nc0 != nc2 and nc0 != nc3 and nc1 != nc2 and nc1 != nc3 and nc2 != nc3:
                    ok = False
                    break
            if ok:
                color_counts[old] -= 1
                color_counts[new_c] += 1
                colors[e] = new_c

        colors, distinct = _local_search(colors, color_counts, None, local_deadline, rng)

        if distinct > best_overall_distinct:
            best_overall_distinct = distinct
            best_overall_colors = colors.copy()

        num_restarts += 1

    if best_overall_colors is None:
        # Fallback: all zeros is always valid
        best_overall_colors = np.zeros(NUM_EDGES, dtype=np.int32)

    # Final validation
    if not _validate(best_overall_colors):
        return [0] * NUM_EDGES

    return best_overall_colors.tolist()
