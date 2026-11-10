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

NUM_CYCLES = len(CYCLES)  # 3 * C(16,4) = 3 * 1820 = 5460

# For each edge, which cycles does it belong to?
EDGE_TO_CYCLES = [[] for _ in range(NUM_EDGES)]
for ci, cyc in enumerate(CYCLES):
    for ei in cyc:
        EDGE_TO_CYCLES[ei].append(ci)


def count_violations(colors):
    """Count number of 4-cycles with 4 distinct edge colors."""
    violations = 0
    for e1, e2, e3, e4 in CYCLES:
        c1, c2, c3, c4 = colors[e1], colors[e2], colors[e3], colors[e4]
        if len(set((c1, c2, c3, c4))) == 4:
            violations += 1
    return violations


def count_distinct_colors(colors):
    return len(set(colors))


def make_initial_coloring():
    """Create a valid coloring using sum-based coloring.
    Color of edge (i,j) = i+j (range 1..31).
    This guarantees the property because in any 4-cycle, diagonals share a color.
    """
    colors = [0] * NUM_EDGES
    for idx, (i, j) in enumerate(EDGES):
        colors[idx] = i + j
    return colors


def compute_delta_violations(colors, edge_idx, new_color):
    """Compute change in violation count when changing edge_idx to new_color."""
    old_color = colors[edge_idx]
    if old_color == new_color:
        return 0
    
    delta = 0
    for ci in EDGE_TO_CYCLES[edge_idx]:
        e1, e2, e3, e4 = CYCLES[ci]
        other_colors = []
        for ej in (e1, e2, e3, e4):
            if ej != edge_idx:
                other_colors.append(colors[ej])
        
        old_set = set([old_color] + other_colors)
        old_bad = 1 if len(old_set) == 4 else 0
        
        new_set = set([new_color] + other_colors)
        new_bad = 1 if len(new_set) == 4 else 0
        
        delta += (new_bad - old_bad)
    
    return delta


def compute_delta_distinct(colors, edge_idx, new_color):
    """Compute change in distinct color count when changing edge_idx to new_color."""
    old_color = colors[edge_idx]
    if old_color == new_color:
        return 0
    
    old_count = colors.count(old_color)
    new_count = colors.count(new_color)
    
    if old_count == 1 and new_count == 0:
        return -1
    elif old_count == 1 and new_count > 0:
        return 0  # old removed, new already exists
    elif old_count > 1 and new_count == 0:
        return 1  # new color added
    else:
        return 0


def simulated_annealing(initial_colors, deadline):
    """Simulated annealing to maximize distinct colors while keeping 0 violations."""
    colors = list(initial_colors)
    best_colors = list(colors)
    best_distinct = count_distinct_colors(colors)
    cur_violations = count_violations(colors)
    cur_distinct = count_distinct_colors(colors)
    
    if cur_violations == 0:
        best_distinct = cur_distinct
        best_colors = list(colors)
    
    T = 5.0
    T_min = 0.01
    cooling = 0.9995
    
    while time.time() < deadline - 2.0:
        # Pick a random edge to change
        ei = random.randint(0, NUM_EDGES - 1)
        old_color = colors[ei]
        
        # Choose new color strategy based on current state
        if cur_violations == 0:
            # Try to increase distinct colors
            used = set(colors)
            if len(used) < MAX_COLOR + 1:
                unused = [c for c in range(MAX_COLOR + 1) if c not in used]
                if unused:
                    new_color = random.choice(unused)
                else:
                    new_color = random.randint(0, MAX_COLOR)
            else:
                new_color = random.randint(0, MAX_COLOR)
        else:
            # Try to reduce violations
            new_color = random.randint(0, MAX_COLOR)
        
        if new_color == old_color:
            continue
        
        # Compute deltas
        delta_viol = compute_delta_violations(colors, ei, new_color)
        delta_distinct = compute_delta_distinct(colors, ei, new_color)
        
        new_violations = cur_violations + delta_viol
        new_distinct = cur_distinct + delta_distinct
        
        # Acceptance criterion
        if new_violations == 0 and new_distinct > cur_distinct:
            accept = True
        elif new_violations < cur_violations:
            accept = True
        elif new_violations == cur_violations and new_distinct >= cur_distinct:
            accept = True
        else:
            # Metropolis acceptance with combined energy
            delta_energy = (new_violations - cur_violations) * 10.0 - (new_distinct - cur_distinct)
            if delta_energy < 0:
                accept = True
            else:
                accept = random.random() < np.exp(-delta_energy / max(T, 1e-10))
        
        if accept:
            colors[ei] = new_color
            cur_violations = new_violations
            cur_distinct = new_distinct
            
            if cur_violations == 0 and cur_distinct > best_distinct:
                best_distinct = cur_distinct
                best_colors = list(colors)
        
        # Cool down
        T = max(T * cooling, T_min)
    
    return best_colors, best_distinct


def construct():
    """Return 120 integer colors in 0..119 for lexicographically ordered pairs (i,j), 0<=i<j<16.
    No four-cycle may have four distinct edge colors.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    random.seed(42)
    np.random.seed(42)
    
    # Start with the known valid sum-based coloring
    initial = make_initial_coloring()
    best_colors = initial
    best_score = count_distinct_colors(initial)
    
    # Run simulated annealing
    sa_colors, sa_distinct = simulated_annealing(initial, deadline)
    if sa_distinct > best_score:
        best_score = sa_distinct
        best_colors = list(sa_colors)
    
    # Final safety check
    if count_violations(best_colors) != 0:
        best_colors = make_initial_coloring()
    
    return list(best_colors)
