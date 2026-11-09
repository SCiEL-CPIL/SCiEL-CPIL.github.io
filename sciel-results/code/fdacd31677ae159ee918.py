"""
Combined solver: hierarchical Gallai partition (structural) + fast local search (refinement).

The structural insight: a recursive partition of vertices where cross-edges between
parts get a single shared color, and each part is recursively colored, yields a valid
Gallai coloring. The number of colors used equals the number of internal nodes in the
partition tree.

The refinement machinery: fast incremental validity checking via precomputed
edge-to-cycle mappings, enabling rapid local search to potentially find colorings
with more distinct colors than the pure structural approach.

Strategy:
1. Generate the best hierarchical Gallai coloring (various binary splits, recursively).
2. Use fast local search to try to increase the number of distinct colors while
   maintaining validity.
"""
import time
import random

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 119

def _edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

# Precompute all 4-cycles as tuples of 4 edge indices
_CYCLES = []
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            for d in range(c + 1, N):
                _CYCLES.append((_edge_idx(a,b), _edge_idx(b,c), _edge_idx(c,d), _edge_idx(d,a)))
                _CYCLES.append((_edge_idx(a,b), _edge_idx(b,d), _edge_idx(d,c), _edge_idx(c,a)))
                _CYCLES.append((_edge_idx(a,c), _edge_idx(c,b), _edge_idx(b,d), _edge_idx(d,a)))

NUM_CYCLES = len(_CYCLES)

# For each edge, precompute which cycles contain it
_EDGE_TO_CYCLES = [[] for _ in range(NUM_EDGES)]
for ci, cycle in enumerate(_CYCLES):
    for e in cycle:
        _EDGE_TO_CYCLES[e].append(ci)

def _check_edge_change(colors, edge_idx, new_color):
    """Check if changing colors[edge_idx] to new_color maintains validity."""
    old_color = colors[edge_idx]
    if old_color == new_color:
        return True
    for ci in _EDGE_TO_CYCLES[edge_idx]:
        e0, e1, e2, e3 = _CYCLES[ci]
        c0 = colors[e0] if e0 != edge_idx else new_color
        c1 = colors[e1] if e1 != edge_idx else new_color
        c2 = colors[e2] if e2 != edge_idx else new_color
        c3 = colors[e3] if e3 != edge_idx else new_color
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            return False
    return True

def _count_colors(colors):
    return len(set(colors))

def _is_valid(colors):
    """Check if no 4-cycle has four distinct edge colors."""
    for cycle in _CYCLES:
        e0, e1, e2, e3 = cycle
        c0, c1, c2, c3 = colors[e0], colors[e1], colors[e2], colors[e3]
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            return False
    return True

def _build_hierarchical(vertices, start_color, splits, color_map):
    """
    Build a hierarchical Gallai coloring.
    vertices: list of vertex indices
    start_color: the next available color index
    splits: dict mapping len(vertices) -> split point k
    color_map: dict to store edge colors
    Returns the next available color index.
    """
    n = len(vertices)
    if n <= 1:
        return start_color
    if n == 2:
        u, v = vertices
        color_map[(min(u,v), max(u,v))] = start_color
        return start_color + 1
    if n == 3:
        u, v, w = vertices
        color_map[(min(u,v), max(u,v))] = start_color
        color_map[(min(v,w), max(v,w))] = start_color + 1
        color_map[(min(u,w), max(u,w))] = start_color + 2
        return start_color + 3
    
    k = splits.get(n, n // 2)
    v1 = vertices[:k]
    v2 = vertices[k:]
    
    # Cross edges get one shared color
    cross_color = start_color
    for u in v1:
        for v in v2:
            color_map[(min(u,v), max(u,v))] = cross_color
    
    next_color = start_color + 1
    next_color = _build_hierarchical(v1, next_color, splits, color_map)
    next_color = _build_hierarchical(v2, next_color, splits, color_map)
    return next_color

def _generate_structural_candidates():
    """Generate multiple hierarchical Gallai colorings with different split strategies."""
    candidates = []
    
    # Various split strategies
    split_configs = []
    
    # From inspiration 1 (proven optimal for small n)
    splits1 = {
        4: 3, 5: 3, 6: 3, 7: 3, 8: 4, 9: 3, 10: 4,
        11: 5, 12: 6, 13: 6, 14: 7, 15: 6, 16: 7
    }
    split_configs.append(splits1)
    
    # Balanced splits
    splits2 = {n: n // 2 for n in range(4, N+1)}
    split_configs.append(splits2)
    
    # Star-like (split off one vertex)
    splits3 = {n: 1 for n in range(4, N+1)}
    split_configs.append(splits3)
    
    # Split off two vertices
    splits4 = {n: 2 for n in range(4, N+1)}
    split_configs.append(splits4)
    
    # Mixed: small splits for small n, balanced for large
    splits5 = {}
    for n in range(4, N+1):
        if n <= 5:
            splits5[n] = 2
        elif n <= 8:
            splits5[n] = 3
        else:
            splits5[n] = n // 2
    split_configs.append(splits5)
    
    # Another mixed strategy
    splits6 = {}
    for n in range(4, N+1):
        if n <= 6:
            splits6[n] = 3
        elif n <= 10:
            splits6[n] = 4
        else:
            splits6[n] = n // 2
    split_configs.append(splits6)
    
    # Try all possible split values for each n (exhaustive for small n)
    # For n=4..8, try all valid splits
    for s4 in range(1, 4):
        for s5 in range(1, 5):
            for s6 in range(1, 6):
                splits7 = {4: s4, 5: s5, 6: s6}
                for s7 in range(1, 7):
                    splits7[7] = s7
                    for s8 in range(1, 8):
                        splits7[8] = s8
                        # For larger n, use balanced
                        for n in range(9, N+1):
                            splits7[n] = n // 2
                        split_configs.append(dict(splits7))
    
    vertices = list(range(N))
    
    for splits in split_configs:
        color_map = {}
        _build_hierarchical(vertices, 0, splits, color_map)
        
        # Flatten to list of 120 integers
        colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i+1, N):
                colors[_edge_idx(i, j)] = color_map.get((i, j), 0)
        
        if _is_valid(colors):
            candidates.append(colors)
    
    return candidates

def _local_search(colors, deadline, max_iter_per_round=200):
    """
    Perform local search to maximize the number of distinct colors.
    Returns the best coloring found.
    """
    current_score = _count_colors(colors)
    best_colors = colors[:]
    best_score = current_score
    
    no_improve_count = 0
    max_no_improve = 50000
    iterations = 0
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        iterations += 1
        if iterations % 5000 == 0 and time.time() > deadline - 5:
            break
        
        improved = False
        
        # Try random edge changes
        for _ in range(max_iter_per_round):
            if time.time() > deadline:
                break
            
            edge_idx = random.randint(0, NUM_EDGES - 1)
            old_color = colors[edge_idx]
            
            # Try a new random color
            new_color = random.randint(0, MAX_COLOR)
            if new_color == old_color:
                continue
            
            # Check if this change maintains validity
            if _check_edge_change(colors, edge_idx, new_color):
                colors[edge_idx] = new_color
                new_score = _count_colors(colors)
                if new_score > current_score:
                    current_score = new_score
                    no_improve_count = 0
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_colors = colors[:]
                else:
                    colors[edge_idx] = old_color
            else:
                pass  # Revert (already unchanged)
        
        if not improved:
            no_improve_count += max_iter_per_round
    
    return best_colors, best_score

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill
    
    random.seed(42)
    
    # Generate structural colorings
    candidates = _generate_structural_candidates()
    
    # Find the best structural coloring
    best_colors = None
    best_score = 0
    for cand in candidates:
        score = _count_colors(cand)
        if score > best_score:
            best_score = score
            best_colors = cand[:]
    
    # If no structural coloring found, start with all zeros
    if best_colors is None:
        best_colors = [0] * NUM_EDGES
        best_score = 1
    
    # Local search from the best structural coloring
    colors = best_colors[:]
    current_score = best_score
    
    # Run local search
    improved_colors, improved_score = _local_search(colors, deadline)
    
    if improved_score > best_score:
        best_score = improved_score
        best_colors = improved_colors[:]
    
    return best_colors
