"""Stronger solver: structured colorings + fast local search with incremental validity."""
import time
import random
import itertools

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

def _is_valid(colors):
    """Check if no 4-cycle has four distinct edge colors."""
    for cycle in _CYCLES:
        e0, e1, e2, e3 = cycle
        c0, c1, c2, c3 = colors[e0], colors[e1], colors[e2], colors[e3]
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            return False
    return True

def _check_edge_change(colors, edge_idx, new_color):
    """Check if changing colors[edge_idx] to new_color maintains validity.
    Only need to check cycles containing edge_idx."""
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

def _generate_structural_colorings():
    """Generate a set of structured colorings that may be valid."""
    candidates = []
    
    # Pattern 1: (i + j) % k
    for k in range(2, 120):
        colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i + 1, N):
                colors[_edge_idx(i, j)] = (i + j) % k
        if _is_valid(colors):
            candidates.append(colors[:])
    
    # Pattern 2: (i * j) % k
    for k in range(2, 120):
        colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i + 1, N):
                colors[_edge_idx(i, j)] = (i * j) % k
        if _is_valid(colors):
            candidates.append(colors[:])
    
    # Pattern 3: (i - j) % k
    for k in range(2, 120):
        colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i + 1, N):
                colors[_edge_idx(i, j)] = (i - j) % k
        if _is_valid(colors):
            candidates.append(colors[:])
    
    # Pattern 4: (i * 7 + j * 3) % k
    for k in range(2, 120):
        colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i + 1, N):
                colors[_edge_idx(i, j)] = (i * 7 + j * 3) % k
        if _is_valid(colors):
            candidates.append(colors[:])
    
    # Pattern 5: (i * 5 + j * 11) % k
    for k in range(2, 120):
        colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i + 1, N):
                colors[_edge_idx(i, j)] = (i * 5 + j * 11) % k
        if _is_valid(colors):
            candidates.append(colors[:])
    
    # Pattern 6: (i^2 + j) % k
    for k in range(2, 120):
        colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i + 1, N):
                colors[_edge_idx(i, j)] = (i * i + j) % k
        if _is_valid(colors):
            candidates.append(colors[:])
    
    # Pattern 7: (i + j*j) % k
    for k in range(2, 120):
        colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i + 1, N):
                colors[_edge_idx(i, j)] = (i + j * j) % k
        if _is_valid(colors):
            candidates.append(colors[:])
    
    # Pattern 8: (i * j + i + j) % k
    for k in range(2, 120):
        colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i + 1, N):
                colors[_edge_idx(i, j)] = (i * j + i + j) % k
        if _is_valid(colors):
            candidates.append(colors[:])
    
    # Pattern 9: (i * 13 + j * 17) % k
    for k in range(2, 120):
        colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i + 1, N):
                colors[_edge_idx(i, j)] = (i * 13 + j * 17) % k
        if _is_valid(colors):
            candidates.append(colors[:])
    
    # Pattern 10: (i * 3 + j * 5 + 1) % k
    for k in range(2, 120):
        colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i + 1, N):
                colors[_edge_idx(i, j)] = (i * 3 + j * 5 + 1) % k
        if _is_valid(colors):
            candidates.append(colors[:])
    
    # Pattern 11: (i * 11 + j * 7 + i*j) % k
    for k in range(2, 120):
        colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i + 1, N):
                colors[_edge_idx(i, j)] = (i * 11 + j * 7 + i * j) % k
        if _is_valid(colors):
            candidates.append(colors[:])
    
    # Pattern 12: (i * 17 + j * 13 + i*j*2) % k
    for k in range(2, 120):
        colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i + 1, N):
                colors[_edge_idx(i, j)] = (i * 17 + j * 13 + i * j * 2) % k
        if _is_valid(colors):
            candidates.append(colors[:])
    
    # Pattern 13: (i * 19 + j * 23) % k
    for k in range(2, 120):
        colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i + 1, N):
                colors[_edge_idx(i, j)] = (i * 19 + j * 23) % k
        if _is_valid(colors):
            candidates.append(colors[:])
    
    # Pattern 14: (i * 23 + j * 19 + 3) % k
    for k in range(2, 120):
        colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i + 1, N):
                colors[_edge_idx(i, j)] = (i * 23 + j * 19 + 3) % k
        if _is_valid(colors):
            candidates.append(colors[:])
    
    # Pattern 15: (i * 29 + j * 31) % k
    for k in range(2, 120):
        colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i + 1, N):
                colors[_edge_idx(i, j)] = (i * 29 + j * 31) % k
        if _is_valid(colors):
            candidates.append(colors[:])
    
    # Pattern 16: (i * 31 + j * 29 + 5) % k
    for k in range(2, 120):
        colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i + 1, N):
                colors[_edge_idx(i, j)] = (i * 31 + j * 29 + 5) % k
        if _is_valid(colors):
            candidates.append(colors[:])
    
    # Pattern 17: (i * 37 + j * 41) % k
    for k in range(2, 120):
        colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i + 1, N):
                colors[_edge_idx(i, j)] = (i * 37 + j * 41) % k
        if _is_valid(colors):
            candidates.append(colors[:])
    
    # Pattern 18: (i * 41 + j * 37 + 7) % k
    for k in range(2, 120):
        colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i + 1, N):
                colors[_edge_idx(i, j)] = (i * 41 + j * 37 + 7) % k
        if _is_valid(colors):
            candidates.append(colors[:])
    
    # Pattern 19: (i * 43 + j * 47) % k
    for k in range(2, 120):
        colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i + 1, N):
                colors[_edge_idx(i, j)] = (i * 43 + j * 47) % k
        if _is_valid(colors):
            candidates.append(colors[:])
    
    # Pattern 20: (i * 47 + j * 43 + 11) % k
    for k in range(2, 120):
        colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i + 1, N):
                colors[_edge_idx(i, j)] = (i * 47 + j * 43 + 11) % k
        if _is_valid(colors):
            candidates.append(colors[:])
    
    return candidates

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    random.seed(12345)
    
    # Generate structural colorings
    candidates = _generate_structural_colorings()
    
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
    
    # Precompute which colors are currently used
    used_colors = set(colors)
    
    no_improve_count = 0
    max_no_improve = 200000
    iterations = 0
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        iterations += 1
        if iterations % 10000 == 0:
            # Periodic check
            pass
        
        improved = False
        
        # Try random edge changes
        for _ in range(500):
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
                    used_colors = set(colors)
                else:
                    colors[edge_idx] = old_color
            else:
                pass  # Revert (already unchanged)
        
        if not improved:
            no_improve_count += 500
    
    if current_score > best_score:
        best_score = current_score
        best_colors = colors[:]
    
    return best_colors
