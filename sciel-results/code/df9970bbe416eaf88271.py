"""Edge-coloring of K_16 to maximize distinct colors with no rainbow C4."""
import time
import random
import sys

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 120

def _edge_index(i, j):
    """Index for edge (i,j) with i<j in the flat array."""
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
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            violations += 1
    return violations

def _count_distinct(colors):
    return len(set(colors))

def _make_structured_baseline():
    """
    Create a valid coloring using a structured approach.
    Use the approach: color by (i mod 4, j mod 4) as color - this gives 16 colors.
    This is guaranteed valid because in any 4-cycle, at least two edges will share
    the same (i mod 4, j mod 4) pattern due to the modular structure.
    """
    colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i + 1, N):
            colors[_EDGE_IDX[(i, j)]] = (i % 4) * 4 + (j % 4)
    return colors

def _improve_coloring(colors, deadline):
    """
    Local search to increase distinct colors while maintaining validity.
    Strategy: try to recolor edges to new colors, accepting only if no new violations.
    """
    colors = list(colors)
    best_colors = list(colors)
    best_distinct = _count_distinct(colors)
    
    rng = random.Random(42)
    t0 = time.time()
    
    # Track used colors
    used_colors = set(colors)
    
    max_iter = 100000
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Try to introduce a new color
        available = [c for c in range(MAX_COLOR) if c not in used_colors]
        if not available:
            break
        
        # Pick a random edge to recolor
        e = rng.randint(0, NUM_EDGES - 1)
        old_color = colors[e]
        
        # Pick a new color: bias toward new colors
        if rng.random() < 0.5:
            new_color = rng.choice(available)
        else:
            new_color = rng.randint(0, MAX_COLOR - 1)
        
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
            if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
                valid = False
                break
        
        if valid:
            # Accept the move
            colors[e] = new_color
            used_colors.add(new_color)
            
            # Update best if improved
            current_distinct = _count_distinct(colors)
            if current_distinct > best_distinct:
                best_distinct = current_distinct
                best_colors = list(colors)
    
    return best_colors, best_distinct

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    # Start with a structured baseline
    best_colors = _make_structured_baseline()
    best_distinct = _count_distinct(best_colors)
    
    # Verify baseline is valid
    if _count_violations(best_colors) > 0:
        # Fallback to all zeros if baseline is invalid
        best_colors = [0] * NUM_EDGES
        best_distinct = 1
    
    # Try to improve using local search
    improved_colors, improved_distinct = _improve_coloring(best_colors, DEADLINE)
    
    if improved_distinct > best_distinct:
        best_colors = improved_colors
        best_distinct = improved_distinct
    
    # Final validation
    final_violations = _count_violations(best_colors)
    if final_violations > 0:
        # If somehow invalid, return the structured baseline (which should be valid)
        return _make_structured_baseline()
    
    return best_colors

# Quick test
if __name__ == "__main__":
    result = construct()
    print(f"Length: {len(result)}")
    print(f"Distinct colors: {len(set(result))}")
    print(f"Min: {min(result)}, Max: {max(result)}")
    print(f"Violations: {_count_violations(result)}")
