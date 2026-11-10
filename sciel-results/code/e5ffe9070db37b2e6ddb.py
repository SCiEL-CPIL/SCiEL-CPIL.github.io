import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 119

# Precompute edges in lexicographic order
EDGES = [(i, j) for i in range(N) for j in range(i + 1, N)]
EDGE_IDX = {}
for idx, (i, j) in enumerate(EDGES):
    EDGE_IDX[(i, j)] = idx
    EDGE_IDX[(j, i)] = idx

# Precompute all 4-cycles and their edge indices
CYCLES = []
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            for d in range(c + 1, N):
                # Cycle 1: a-b-c-d-a
                e1 = EDGE_IDX[(a, b)]
                e2 = EDGE_IDX[(b, c)]
                e3 = EDGE_IDX[(c, d)]
                e4 = EDGE_IDX[(d, a)]
                CYCLES.append((e1, e2, e3, e4))
                # Cycle 2: a-b-d-c-a
                e1 = EDGE_IDX[(a, b)]
                e2 = EDGE_IDX[(b, d)]
                e3 = EDGE_IDX[(d, c)]
                e4 = EDGE_IDX[(c, a)]
                CYCLES.append((e1, e2, e3, e4))
                # Cycle 3: a-c-b-d-a
                e1 = EDGE_IDX[(a, c)]
                e2 = EDGE_IDX[(c, b)]
                e3 = EDGE_IDX[(b, d)]
                e4 = EDGE_IDX[(d, a)]
                CYCLES.append((e1, e2, e3, e4))

NUM_CYCLES = len(CYCLES)  # 5460

# Convert cycles to numpy array for fast vectorized checks
CYCLES_ARR = np.array(CYCLES, dtype=np.int32)  # shape (5460, 4)

# For each edge, which cycles does it belong to?
OTHER_EDGES = [[] for _ in range(NUM_EDGES)]
EDGE_CYCLE_COUNT = [0] * NUM_EDGES
for cyc in CYCLES:
    e1, e2, e3, e4 = cyc
    OTHER_EDGES[e1].append((e2, e3, e4))
    OTHER_EDGES[e2].append((e1, e3, e4))
    OTHER_EDGES[e3].append((e1, e2, e4))
    OTHER_EDGES[e4].append((e1, e2, e3))
    EDGE_CYCLE_COUNT[e1] += 1
    EDGE_CYCLE_COUNT[e2] += 1
    EDGE_CYCLE_COUNT[e3] += 1
    EDGE_CYCLE_COUNT[e4] += 1

# Sort edges by cycle count (ascending) - edges in fewer cycles are more flexible
EDGES_BY_FLEXIBILITY = sorted(range(NUM_EDGES), key=lambda e: EDGE_CYCLE_COUNT[e])


def count_violations(colors):
    """Count number of 4-cycles with 4 distinct edge colors using numpy."""
    c = np.array(colors, dtype=np.int32)
    c1 = c[CYCLES_ARR[:, 0]]
    c2 = c[CYCLES_ARR[:, 1]]
    c3 = c[CYCLES_ARR[:, 2]]
    c4 = c[CYCLES_ARR[:, 3]]
    # Check if all four are distinct
    all_distinct = (c1 != c2) & (c1 != c3) & (c1 != c4) & (c2 != c3) & (c2 != c4) & (c3 != c4)
    return int(np.sum(all_distinct))


def check_edge_change(colors, edge_idx, new_color):
    """
    Check if changing colors[edge_idx] to new_color creates a rainbow cycle.
    Returns True if valid, False otherwise.
    """
    if colors[edge_idx] == new_color:
        return True
    
    for o1, o2, o3 in OTHER_EDGES[edge_idx]:
        c1 = colors[o1]
        c2 = colors[o2]
        c3 = colors[o3]
        
        if c1 == c2 or c2 == c3 or c1 == c3:
            continue
            
        if new_color == c1 or new_color == c2 or new_color == c3:
            continue
            
        return False
        
    return True


def make_structured_baseline():
    """Create a structured coloring with many colors from the start."""
    colors = [0] * NUM_EDGES
    # Use vertex-based scheme: color edge (i,j) with (i*16+j) % some_base
    # This naturally creates many distinct colors
    for e, (i, j) in enumerate(EDGES):
        colors[e] = (i * 16 + j) % 24
    
    # Verify and fix if needed
    if count_violations(colors) > 0:
        # Fallback to simpler scheme
        for e, (i, j) in enumerate(EDGES):
            colors[e] = (i + j) % 8
    
    return colors


def try_add_new_color(colors, rng, deadline):
    """Try to add a new color to an edge. Returns True if successful."""
    used = set(colors)
    if len(used) >= MAX_COLOR + 1:
        return False
    
    unused = [c for c in range(MAX_COLOR + 1) if c not in used]
    if not unused:
        return False
    
    # Try edges in order of flexibility (fewest cycles first)
    for e in EDGES_BY_FLEXIBILITY:
        if time.time() >= deadline:
            break
        
        # Try several unused colors on this edge
        for new_c in rng.sample(unused, min(5, len(unused))):
            if check_edge_change(colors, e, new_c):
                colors[e] = new_c
                return True
    
    # If systematic approach failed, try random
    for _ in range(20):
        e = rng.randint(0, NUM_EDGES - 1)
        new_c = rng.choice(unused)
        if check_edge_change(colors, e, new_c):
            colors[e] = new_c
            return True
    
    return False


def local_search(colors, rng, deadline):
    """Local search to maximize distinct colors."""
    current_score = len(set(colors))
    best_score = current_score
    best_colors = list(colors)
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        # Try to add a new color
        if try_add_new_color(colors, rng, deadline):
            new_score = len(set(colors))
            if new_score > best_score:
                best_score = new_score
                best_colors = list(colors)
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        # If stuck for too long, do perturbation
        if no_improve_count > max_no_improve:
            # Recolor some edges randomly
            num_perturb = rng.randint(5, 15)
            for _ in range(num_perturb):
                e = rng.randint(0, NUM_EDGES - 1)
                c = rng.randint(0, MAX_COLOR)
                if check_edge_change(colors, e, c):
                    colors[e] = c
            
            no_improve_count = 0
    
    return best_colors, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    rng = random.Random(42)
    
    best_colors = [0] * NUM_EDGES
    best_score = 1
    
    # Phase 1: Start with structured baseline
    colors = make_structured_baseline()
    initial_score = len(set(colors))
    
    if count_violations(colors) == 0 and initial_score > best_score:
        best_score = initial_score
        best_colors = list(colors)
    
    # Phase 2: Local search from multiple starting points
    num_starts = 6
    per_start_time = (deadline - start_time) / num_starts
    
    for start_i in range(num_starts):
        if time.time() >= deadline - 3.0:
            break
        
        if start_i == 0:
            # Continue from structured baseline
            colors = list(best_colors)
        elif start_i == 1:
            # Different structured baseline
            colors = [0] * NUM_EDGES
            for e, (i, j) in enumerate(EDGES):
                colors[e] = (i * 7 + j * 3) % 16
            if count_violations(colors) > 0:
                colors = make_structured_baseline()
        elif start_i == 2:
            # Greedy valid coloring
            colors = [0] * NUM_EDGES
            for e in EDGES_BY_FLEXIBILITY:
                for attempt in range(30):
                    c = rng.randint(0, MAX_COLOR)
                    if check_edge_change(colors, e, c):
                        colors[e] = c
                        break
        elif start_i == 3:
            # Perturb best
            colors = list(best_colors)
            for _ in range(rng.randint(10, 25)):
                e = rng.randint(0, NUM_EDGES - 1)
                c = rng.randint(0, MAX_COLOR)
                if check_edge_change(colors, e, c):
                    colors[e] = c
        elif start_i == 4:
            # Another structured variant
            colors = [0] * NUM_EDGES
            for e, (i, j) in enumerate(EDGES):
                colors[e] = (i * 11 + j * 5) % 20
            if count_violations(colors) > 0:
                colors = make_structured_baseline()
        else:
            # Random restart with greedy fill
            colors = [rng.randint(0, 10) for _ in range(NUM_EDGES)]
            if count_violations(colors) > 0:
                colors = [0] * NUM_EDGES
                for e in EDGES_BY_FLEXIBILITY:
                    for attempt in range(20):
                        c = rng.randint(0, MAX_COLOR)
                        if check_edge_change(colors, e, c):
                            colors[e] = c
                            break
        
        # Run local search
        local_deadline = min(start_time + per_start_time * (start_i + 1) + 2.0, deadline - 2.0)
        improved_colors, improved_score = local_search(colors, rng, local_deadline)
        
        if improved_score > best_score:
            best_score = improved_score
            best_colors = improved_colors
    
    # Final verification
    if count_violations(best_colors) != 0:
        best_colors = [0] * NUM_EDGES
        best_score = 1
    
    return list(best_colors)
