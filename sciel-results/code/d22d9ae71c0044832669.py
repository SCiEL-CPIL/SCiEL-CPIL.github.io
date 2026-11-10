import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

# Precompute all 4-cycles (4-vertex subsets with their 3 possible cycles)
def _precompute_cycles():
    """
    For each 4-vertex subset {a,b,c,d} with a<b<c<d, there are 3 four-cycles:
    Cycle 1: a-b-c-d-a  -> edges (a,b),(b,c),(c,d),(d,a)
    Cycle 2: a-b-d-c-a  -> edges (a,b),(b,d),(d,c),(c,a)
    Cycle 3: a-c-b-d-a  -> edges (a,c),(c,b),(b,d),(d,a)
    Each cycle has 4 edges. The constraint: these 4 edges must NOT have 4 distinct colors.
    """
    cycles = []  # list of tuples of 4 edge indices
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # Edge index for pair (i,j) with i<j: i*(N-1) - i*(i+1)//2 + (j-i-1)
                    # Actually simpler: precompute edge index map
                    e1 = (a, b)
                    e2 = (b, c)
                    e3 = (c, d)
                    e4 = (d, a)
                    # Cycle 1: a-b, b-c, c-d, d-a
                    cycles.append((e1, e2, e3, e4))
                    # Cycle 2: a-b, b-d, d-c, c-a  -> (a,b),(b,d),(c,d),(a,c)
                    cycles.append(((a, b), (b, d), (c, d), (a, c)))
                    # Cycle 3: a-c, c-b, b-d, d-a  -> (a,c),(b,c),(b,d),(a,d)
                    cycles.append(((a, c), (b, c), (b, d), (a, d)))
    return cycles

def _edge_index(i, j):
    """Index for edge (i,j) with i<j in the flat array."""
    # Number of edges before row i: sum_{k=0}^{i-1} (N-1-k) = i*(N-1) - i*(i-1)//2
    # Wait, let's just use a simple formula.
    # Edges are ordered lexicographically by (i,j).
    # For vertex i, edges go to i+1, i+2, ..., N-1. That's (N-1-i) edges.
    # Index = sum_{k=0}^{i-1} (N-1-k) + (j - i - 1)
    # sum_{k=0}^{i-1} (N-1-k) = i*(N-1) - (0+1+...+(i-1)) = i*(N-1) - i*(i-1)//2
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

# Build edge index lookup
_EDGE_IDX = {}
for i in range(N):
    for j in range(i + 1, N):
        _EDGE_IDX[(i, j)] = _edge_index(i, j)

# Precompute cycles as lists of edge indices
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

def _make_structured_baseline():
    """
    Create a valid coloring using a structured approach.
    Key insight: if we color edges based on (i mod k, j mod k) or similar,
    we can ensure that in any 4-cycle, at least two edges share a color.
    
    One approach: color edge (i,j) by (i + j) mod k. In a 4-cycle a-b-c-d-a,
    the sums are i+j, j+c, c+d, d+a. These might all be distinct.
    
    Better: Use a 2D grid structure. For N=16, think of vertices as (x,y) with x,y in 0..3.
    Edge (v1,v2) color based on direction/distance in a structured way.
    """
    colors = [0] * NUM_EDGES
    # Map vertex to (x, y) in 4x4 grid
    vertex_coords = [(i // 4, i % 4) for i in range(16)]
    
    for i in range(N):
        for j in range(i + 1, N):
            xi, yi = vertex_coords[i]
            xj, yj = vertex_coords[j]
            dx = abs(xi - xj)
            dy = abs(yi - yj)
            # Color based on (dx, dy) pair - gives at most 4*4=16 colors
            # But we need to ensure 4-cycle constraint
            # In a 4-cycle, the edge "types" should repeat
            colors[_EDGE_IDX[(i, j)]] = dx * 4 + dy
    return colors

def _make_valid_coloring():
    """
    Create a guaranteed valid coloring with reasonable color count.
    Use the approach: color by (i mod 4, j mod 4) difference or similar.
    """
    colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i + 1, N):
            # Use (i mod 4, j mod 4) as color - this gives 16 colors
            # Need to verify this is valid
            colors[_EDGE_IDX[(i, j)]] = (i % 4) * 4 + (j % 4)
    return colors

def _improve(colors, deadline):
    """
    Local search to increase distinct colors while maintaining validity.
    Strategy: try to recolor edges to new colors, accepting only if no new violations.
    Use simulated annealing with entropy-based penalty.
    """
    colors = list(colors)
    best_colors = list(colors)
    best_distinct = _count_distinct(colors)
    current_violations = _count_violations(colors)
    
    # Track color usage for entropy calculation
    def get_entropy_penalty(counts, num_colors=120):
        """Penalty for low entropy - encourages spreading colors."""
        total = sum(counts)
        if total == 0:
            return 0.0
        probs = [c / total for c in counts if c > 0]
        if not probs:
            return 0.0
        entropy = -sum(p * np.log2(p) for p in probs)
        max_entropy = np.log2(num_colors)
        # Penalty is higher when entropy is lower
        return (max_entropy - entropy) / max_entropy if max_entropy > 0 else 0.0
    
    # Build adjacency: for each edge, which other edges share a cycle with it
    # We'll use a simple approach: when changing edge e, check all cycles containing e
    
    rng = random.Random(42)
    t0 = time.time()
    
    # Temperature schedule
    T_start = 1.0
    T_end = 0.01
    total_steps = 50000
    step = 0
    
    while step < total_steps:
        if time.time() > deadline:
            break
        
        step += 1
        T = T_start * (T_end / T_start) ** (step / total_steps)
        
        # Pick a random edge to recolor
        e = rng.randint(0, NUM_EDGES - 1)
        old_color = colors[e]
        
        # Pick a new color: either a random color or try to introduce a new one
        # Bias toward colors that are underused or new
        if rng.random() < 0.3:
            # Try a new color
            new_color = rng.randint(0, 119)
        else:
            # Try a color from the current palette or nearby
            new_color = rng.randint(0, 119)
        
        if new_color == old_color:
            continue
        
        # Check if this change creates any new violations
        # Only need to check cycles containing edge e
        valid = True
        for ci in _EDGE_TO_CYCLES[e]:
            cyc = _CYCLES[ci]
            # Get the 4 colors in this cycle with the new color for edge e
            c0 = colors[cyc[0]] if cyc[0] != e else new_color
            c1 = colors[cyc[1]] if cyc[1] != e else new_color
            c2 = colors[cyc[2]] if cyc[2] != e else new_color
            c3 = colors[cyc[3]] if cyc[3] != e else new_color
            if len(set((c0, c1, c2, c3))) == 4:
                valid = False
                break
        
        if valid:
            # Accept the move
            colors[e] = new_color
            current_violations = 0  # Still valid
            
            # Check if we improved distinct count
            cur_distinct = _count_distinct(colors)
            if cur_distinct > best_distinct:
                best_distinct = cur_distinct
                best_colors = list(colors)
        else:
            # Would create violation - reject (we're keeping valid colorings only)
            pass
    
    return best_colors, best_distinct

def _try_new_color_insertion(colors, deadline):
    """
    Try to insert new colors by finding edges that can be recolored without
    creating violations, specifically targeting colors not yet in use.
    """
    colors = list(colors)
    used_colors = set(colors)
    unused_colors = [c for c in range(120) if c not in used_colors]
    
    rng = random.Random(123)
    t0 = time.time()
    
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
            valid = True
            for ci in _EDGE_TO_CYCLES[e]:
                cyc = _CYCLES[ci]
                c0 = colors[cyc[0]] if cyc[0] != e else new_color
                c1 = colors[cyc[1]] if cyc[1] != e else new_color
                c2 = colors[cyc[2]] if cyc[2] != e else new_color
                c3 = colors[cyc[3]] if cyc[3] != e else new_color
                if len(set((c0, c1, c2, c3))) == 4:
                    valid = False
                    break
            
            if valid:
                colors[e] = new_color
                used_colors.add(new_color)
                break  # Move to next unused color
    
    return colors

def construct():
    deadline = time.time() + 85  # Safety margin before 90s limit
    
    # Start with a valid structured coloring
    colors = _make_valid_coloring()
    distinct = _count_distinct(colors)
    
    # Verify it's valid
    if _count_violations(colors) > 0:
        # Fallback to all zeros (always valid)
        colors = [0] * NUM_EDGES
        distinct = 1
    
    # Try to improve with local search
    improved_colors, improved_distinct = _improve(colors, deadline)
    if improved_distinct > distinct:
        colors = improved_colors
        distinct = improved_distinct
    
    # Try to insert new colors
    if time.time() < deadline - 5:
        final_colors = _try_new_color_insertion(colors, deadline)
        final_distinct = _count_distinct(final_colors)
        if final_distinct > distinct and _count_violations(final_colors) == 0:
            colors = final_colors
            distinct = final_distinct
    
    # Final verification
    if _count_violations(colors) > 0:
        # Shouldn't happen, but safety net
        return [0] * NUM_EDGES
    
    return colors

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
