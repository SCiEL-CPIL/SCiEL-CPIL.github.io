"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2
NUM_K4 = 1820

# Pre-compute edge indices
EDGE_IDX = {}
idx = 0
for i in range(N):
    for j in range(i + 1, N):
        EDGE_IDX[(i, j)] = idx
        idx += 1

# Pre-compute K4s and their edge indices
K4_EDGES = []
for quad in combinations(range(N), 4):
    a, b, c, d = quad
    edges = [
        EDGE_IDX[(a, b)], EDGE_IDX[(a, c)], EDGE_IDX[(a, d)],
        EDGE_IDX[(b, c)], EDGE_IDX[(b, d)], EDGE_IDX[(c, d)]
    ]
    K4_EDGES.append(tuple(edges))

# Pre-compute for each edge, which K4s it belongs to
EDGE_TO_K4 = [[] for _ in range(NUM_EDGES)]
for k_idx, edges in enumerate(K4_EDGES):
    for e in edges:
        EDGE_TO_K4[e].append(k_idx)

def _count_distinct_6(c0, c1, c2, c3, c4, c5):
    return len({c0, c1, c2, c3, c4, c5})

def _initial_violations(colors):
    k4_status = [False] * NUM_K4
    total_v = 0
    for k_idx, edges in enumerate(K4_EDGES):
        c0, c1, c2, c3, c4, c5 = (colors[e] for e in edges)
        if _count_distinct_6(c0, c1, c2, c3, c4, c5) < 5:
            k4_status[k_idx] = True
            total_v += 1
    return k4_status, total_v

def _update_k4_status_for_edge(current, e_idx, k4_status):
    """Update K4 status for all K4s containing edge e_idx."""
    for k_idx in EDGE_TO_K4[e_idx]:
        edges = K4_EDGES[k_idx]
        c0, c1, c2, c3, c4, c5 = (current[e] for e in edges)
        k4_status[k_idx] = _count_distinct_6(c0, c1, c2, c3, c4, c5) < 5

def _check_move_valid(current, e_idx, new_c, k4_status, current_violations):
    """Check if changing edge e_idx to new_c keeps all K4s valid (or improves violations).
    Returns (is_valid, new_violations)."""
    new_violations = current_violations
    for k_idx in EDGE_TO_K4[e_idx]:
        edges = K4_EDGES[k_idx]
        e0, e1, e2, e3, e4, e5 = edges
        c0 = current[e0] if e0 != e_idx else new_c
        c1 = current[e1] if e1 != e_idx else new_c
        c2 = current[e2] if e2 != e_idx else new_c
        c3 = current[e3] if e3 != e_idx else new_c
        c4 = current[e4] if e4 != e_idx else new_c
        c5 = current[e5] if e5 != e_idx else new_c
        
        new_distinct = _count_distinct_6(c0, c1, c2, c3, c4, c5)
        new_status = new_distinct < 5
        
        if k4_status[k_idx] and not new_status:
            new_violations -= 1
        elif not k4_status[k_idx] and new_status:
            new_violations += 1
            return False, new_violations
    return True, new_violations

def _construct():
    deadline = time.time() + 110
    random.seed(42)
    
    # Start with (i+j) % 17
    best = [(i + j) % 17 for i in range(N) for j in range(i + 1, N)]
    k4_status, total_v = _initial_violations(best)
    
    if total_v > 0:
        found_valid = False
        for k in range(13, 21):
            if time.time() > deadline - 10: break
            cand = [(i + j) % k for i in range(N) for j in range(i + 1, N)]
            ks, tv = _initial_violations(cand)
            if tv == 0:
                best = cand
                k4_status = ks
                total_v = 0
                found_valid = True
                break
        if not found_valid:
            for _ in range(5000):
                if time.time() > deadline - 5: break
                cand = [random.randint(0, 19) for _ in range(NUM_EDGES)]
                ks, tv = _initial_violations(cand)
                if tv == 0:
                    best = cand
                    k4_status = ks
                    total_v = 0
                    found_valid = True
                    break
                elif tv < total_v:
                    best = cand
                    k4_status = ks
                    total_v = tv
            if not found_valid:
                return best

    current = list(best)
    current_violations = total_v
    current_k4_status = k4_status
    
    # Track color frequencies
    MAX_COLOR = 25
    color_freq = [0] * MAX_COLOR
    for c in current:
        if c < MAX_COLOR:
            color_freq[c] += 1
    current_colors = sum(1 for f in color_freq if f > 0)
    
    # Phase 1: Reduce violations if any
    while current_violations > 0:
        if time.time() > deadline - 5:
            break
        e_idx = random.randint(0, NUM_EDGES - 1)
        old_c = current[e_idx]
        new_c = random.randint(0, 19)
        if new_c == old_c:
            continue
        
        is_valid, new_violations = _check_move_valid(current, e_idx, new_c, current_k4_status, current_violations)
        if is_valid and new_violations < current_violations:
            current[e_idx] = new_c
            if old_c < MAX_COLOR:
                color_freq[old_c] -= 1
            if new_c < MAX_COLOR:
                color_freq[new_c] += 1
            current_violations = new_violations
            _update_k4_status_for_edge(current, e_idx, current_k4_status)
    
    # Phase 2: Reduce colors (valid coloring)
    if current_violations == 0:
        # Build list of edges by color frequency (edges with rare colors first)
        # Target: for each color with freq==1, try to merge it into another color
        for step in range(100000):
            if time.time() > deadline - 3:
                break
            
            # Find all edges whose color has frequency 1
            singleton_edges = []
            for e in range(NUM_EDGES):
                c = current[e]
                if c < MAX_COLOR and color_freq[c] == 1:
                    singleton_edges.append(e)
            
            if not singleton_edges:
                break
            
            # Pick a random singleton edge
            e_idx = random.choice(singleton_edges)
            old_c = current[e_idx]
            
            # Try to change to a color that already exists (to reduce color count)
            # Prefer colors with high frequency (more robust)
            existing_colors = [c for c in range(MAX_COLOR) if color_freq[c] > 0 and c != old_c]
            if not existing_colors:
                break
            
            # Try each existing color, pick one that doesn't create violations
            found_move = False
            # Shuffle for randomness
            random.shuffle(existing_colors)
            for new_c in existing_colors:
                is_valid, new_violations = _check_move_valid(current, e_idx, new_c, current_k4_status, current_violations)
                if is_valid and new_violations == 0:
                    # Apply the move
                    current[e_idx] = new_c
                    color_freq[old_c] -= 1
                    color_freq[new_c] += 1
                    current_colors -= 1  # old_c freq becomes 0, new_c was > 0
                    _update_k4_status_for_edge(current, e_idx, current_k4_status)
                    found_move = True
                    break
            
            if not found_move:
                # Try a few random colors as fallback
                for _ in range(5):
                    new_c = random.randint(0, 12)
                    if new_c == old_c:
                        continue
                    is_valid, new_violations = _check_move_valid(current, e_idx, new_c, current_k4_status, current_violations)
                    if is_valid and new_violations == 0:
                        # Check if this reduces color count
                        if color_freq[old_c] == 1 and color_freq[new_c] > 0:
                            current[e_idx] = new_c
                            color_freq[old_c] -= 1
                            color_freq[new_c] += 1
                            current_colors -= 1
                            _update_k4_status_for_edge(current, e_idx, current_k4_status)
                            found_move = True
                            break
        
        # Phase 3: Polish - try to reduce colors further with more aggressive search
        polish_deadline = deadline - 2
        for step in range(50000):
            if time.time() > polish_deadline:
                break
            
            # Pick random edge
            e_idx = random.randint(0, NUM_EDGES - 1)
            old_c = current[e_idx]
            
            # Try all colors from 0 to current_colors-1
            for new_c in range(current_colors):
                if new_c == old_c:
                    continue
                is_valid, new_violations = _check_move_valid(current, e_idx, new_c, current_k4_status, current_violations)
                if not is_valid or new_violations > 0:
                    continue
                
                # Calculate new color count
                new_color_count = current_colors
                if color_freq[old_c] == 1:
                    new_color_count -= 1
                if color_freq[new_c] == 0:
                    new_color_count += 1
                
                if new_color_count < current_colors:
                    current[e_idx] = new_c
                    color_freq[old_c] -= 1
                    color_freq[new_c] += 1
                    current_colors = new_color_count
                    _update_k4_status_for_edge(current, e_idx, current_k4_status)
                    break
    
    return current

def construct():
    return _construct()
# EVOLVE-BLOCK-END
