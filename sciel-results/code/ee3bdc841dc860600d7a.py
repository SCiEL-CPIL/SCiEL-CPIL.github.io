import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

# Edge index for pair (i,j) with i<j in lexicographic order
def _edge_index(i, j):
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

# Build edge index lookup
_EDGE_IDX = {}
for i in range(N):
    for j in range(i + 1, N):
        _EDGE_IDX[(i, j)] = _edge_index(i, j)

# Precompute all 4-cycles (4-vertex subsets with their 3 possible cycles)
_CYCLES = []
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            for d in range(c + 1, N):
                # Cycle 1: a-b, b-c, c-d, d-a
                _CYCLES.append((_EDGE_IDX[(a, b)], _EDGE_IDX[(b, c)], _EDGE_IDX[(c, d)], _EDGE_IDX[(a, d)]))
                # Cycle 2: a-b, b-d, c-d, a-c
                _CYCLES.append((_EDGE_IDX[(a, b)], _EDGE_IDX[(b, d)], _EDGE_IDX[(c, d)], _EDGE_IDX[(a, c)]))
                # Cycle 3: a-c, b-c, b-d, a-d
                _CYCLES.append((_EDGE_IDX[(a, c)], _EDGE_IDX[(b, c)], _EDGE_IDX[(b, d)], _EDGE_IDX[(a, d)]))

# For each edge, which cycles does it appear in?
_EDGE_TO_CYCLES = [[] for _ in range(NUM_EDGES)]
for ci, cyc in enumerate(_CYCLES):
    for e in cyc:
        _EDGE_TO_CYCLES[e].append(ci)

# Precompute for each edge and each of its cycles, the other 3 edge indices
# This allows O(1) validity check per cycle
_EDGE_CYCLE_OTHERS = [[] for _ in range(NUM_EDGES)]
for e in range(NUM_EDGES):
    for ci in _EDGE_TO_CYCLES[e]:
        cyc = _CYCLES[ci]
        others = [cyc[i] for i in range(4) if cyc[i] != e]
        _EDGE_CYCLE_OTHERS[e].append(others)

def _count_violations(colors):
    """Count number of 4-cycles with 4 distinct colors."""
    violations = 0
    for cyc in _CYCLES:
        c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
        if len(set((c0, c1, c2, c3))) == 4:
            violations += 1
    return violations

def _count_distinct(colors):
    return len(set(colors))

def _is_valid_change(colors, e, new_color):
    """Check if recoloring edge e to new_color maintains validity."""
    for others in _EDGE_CYCLE_OTHERS[e]:
        c0 = colors[others[0]]
        c1 = colors[others[1]]
        c2 = colors[others[2]]
        # Check if {new_color, c0, c1, c2} has 4 distinct elements
        if new_color != c0 and new_color != c1 and new_color != c2:
            if c0 != c1 and c0 != c2 and c1 != c2:
                return False
    return True

def _make_valid_coloring():
    """
    Create a guaranteed valid coloring.
    Use the approach: color by (i mod 4, j mod 4).
    """
    colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i + 1, N):
            colors[_EDGE_IDX[(i, j)]] = (i % 4) * 4 + (j % 4)
    return colors

def _local_search(colors, deadline, seed=42):
    """
    Local search to increase distinct colors while maintaining validity.
    """
    colors = list(colors)
    best_colors = list(colors)
    best_distinct = _count_distinct(colors)
    
    rng = random.Random(seed)
    
    while time.time() < deadline:
        # Pick a random edge to recolor
        e = rng.randint(0, NUM_EDGES - 1)
        old_color = colors[e]
        
        # Pick a new color
        new_color = rng.randint(0, 119)
        
        if new_color == old_color:
            continue
        
        # Check if this change maintains validity
        if _is_valid_change(colors, e, new_color):
            colors[e] = new_color
            
            # Check if we improved distinct count
            cur_distinct = _count_distinct(colors)
            if cur_distinct > best_distinct:
                best_distinct = cur_distinct
                best_colors = list(colors)
    
    return best_colors, best_distinct

def _try_new_color_insertion(colors, deadline, seed=123):
    """
    Try to insert new colors by finding edges that can be recolored without
    creating violations, specifically targeting colors not yet in use.
    """
    colors = list(colors)
    used_colors = set(colors)
    unused_colors = [c for c in range(120) if c not in used_colors]
    
    rng = random.Random(seed)
    
    # Shuffle unused colors
    rng.shuffle(unused_colors)
    
    for new_color in unused_colors:
        if time.time() > deadline:
            break
        
        # Try to find an edge that can take this new color
        edges_to_try = list(range(NUM_EDGES))
        rng.shuffle(edges_to_try)
        
        for e in edges_to_try:
            if time.time() > deadline:
                break
            
            old_color = colors[e]
            if old_color == new_color:
                continue
            
            # Check if recoloring e to new_color maintains validity
            if _is_valid_change(colors, e, new_color):
                colors[e] = new_color
                used_colors.add(new_color)
                break  # Move to next unused color
    
    return colors

def _multi_start_local_search(deadline):
    """
    Run multiple local searches from different starting points and return the best.
    """
    best_colors = None
    best_distinct = 0
    
    # Time per start
    num_starts = 5
    time_per_start = (deadline - time.time()) / num_starts
    
    for start in range(num_starts):
        if time.time() > deadline - 1:
            break
        
        # Create a different starting coloring
        if start == 0:
            colors = _make_valid_coloring()
        else:
            # Start with random valid-ish coloring
            colors = [0] * NUM_EDGES
            rng = random.Random(start * 1000 + 7)
            for i in range(N):
                for j in range(i + 1, N):
                    # Use a structured pattern with some randomness
                    colors[_EDGE_IDX[(i, j)]] = rng.randint(0, 15)
            
            # Fix any violations by reverting to safe colors
            # Simple repair: if violation, set all edges in violating cycles to 0
            for _ in range(100):  # Max repair iterations
                violations = []
                for cyc in _CYCLES:
                    c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
                    if len(set((c0, c1, c2, c3))) == 4:
                        violations.append(cyc)
                if not violations:
                    break
                # Fix first violation by setting one edge to match another
                cyc = violations[0]
                # Set edge 3 to match edge 0
                colors[cyc[3]] = colors[cyc[0]]
        
        # Run local search
        start_deadline = time.time() + time_per_start
        improved_colors, improved_distinct = _local_search(colors, start_deadline, seed=start * 100 + 1)
        
        if improved_distinct > best_distinct:
            best_distinct = improved_distinct
            best_colors = improved_colors
    
    return best_colors, best_distinct

def construct():
    deadline = time.time() + 85  # Safety margin before 90s limit
    
    # Try multi-start local search
    colors, distinct = _multi_start_local_search(deadline)
    
    # If we still have time, try new color insertion on the best result
    if colors is not None and time.time() < deadline - 5:
        final_colors = _try_new_color_insertion(colors, deadline)
        final_distinct = _count_distinct(final_colors)
        if final_distinct > distinct and _count_violations(final_colors) == 0:
            colors = final_colors
            distinct = final_distinct
    
    # Final verification
    if colors is None or _count_violations(colors) > 0:
        # Fallback to a known valid coloring
        colors = _make_valid_coloring()
    
    return colors

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
