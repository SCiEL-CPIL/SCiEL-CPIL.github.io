"""Stronger search for a 16-vertex edge coloring with no rainbow 4-cycle."""
import time
import random
import itertools
from collections import defaultdict

N = 16
M = N * (N - 1) // 2  # 120 edges

# Precompute edge index for (i, j) with i < j
edge_idx = {}
for i in range(N):
    for j in range(i + 1, N):
        edge_idx[(i, j)] = i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)
        # Actually let's just use a simple flat index
        # Better: enumerate all pairs in lex order
pairs = []
for i in range(N):
    for j in range(i + 1, N):
        pairs.append((i, j))
edge_idx = {}
for idx, (i, j) in enumerate(pairs):
    edge_idx[(i, j)] = idx

# Precompute all 4-vertex subsets and their 3 cycles (each cycle is 3 edges)
# A 4-cycle on vertices {a,b,c,d} has 3 distinct cycles:
#   (a,b,c,d), (a,b,d,c), (a,c,b,d)  -> edges:
#   (a,b),(b,c),(c,d),(d,a)
#   (a,b),(b,d),(d,c),(c,a)
#   (a,c),(c,b),(b,d),(d,a)
# Each cycle uses 4 edges. Rainbow means all 4 edges have distinct colors.

four_sets = []
for quad in itertools.combinations(range(N), 4):
    a, b, c, d = quad
    # Three 4-cycles:
    # Cycle 1: a-b-c-d-a  -> edges (a,b),(b,c),(c,d),(d,a)
    # Cycle 2: a-b-d-c-a  -> edges (a,b),(b,d),(d,c),(c,a)
    # Cycle 3: a-c-b-d-a  -> edges (a,c),(c,b),(b,d),(d,a)
    e1 = [edge_idx[(a,b)], edge_idx[(b,c)], edge_idx[(c,d)], edge_idx[(d,a)]]
    e2 = [edge_idx[(a,b)], edge_idx[(b,d)], edge_idx[(d,c)], edge_idx[(c,a)]]
    e3 = [edge_idx[(a,c)], edge_idx[(c,b)], edge_idx[(b,d)], edge_idx[(d,a)]]
    four_sets.append((e1, e2, e3))

# For fast constraint checking: for each edge, which 4-cycles contain it?
# We'll use a different approach: maintain a count of "rainbow violations" per 4-set.
# Actually, let's just check all 4-sets after each move. With C(16,4)=1820 4-sets,
# and 3 cycles each, that's ~5460 cycle checks per move. Each check is 4 lookups + set of 4.
# That might be slow. Let's optimize.

# Alternative: For each 4-set, we can track the multiset of colors on each of its 3 cycles.
# A cycle is rainbow iff all 4 colors are distinct.
# When we change an edge's color, we need to update all 4-sets that contain that edge.

# Let's precompute: for each edge e, list of (four_set_idx, cycle_idx) where e is in that cycle.
edge_to_cycles = defaultdict(list)  # edge_idx -> list of (fs_idx, cycle_idx, pos_in_cycle)
for fs_idx, (c1, c2, c3) in enumerate(four_sets):
    for cyc_idx, cyc in enumerate((c1, c2, c3)):
        for pos, e in enumerate(cyc):
            edge_to_cycles[e].append((fs_idx, cyc_idx, pos))

# For each 4-set and cycle, store the 4 edge indices
# four_sets[fs_idx] = (c1, c2, c3) where each is a tuple of 4 edge indices

def is_rainbow(colors, edge_list):
    """Check if the 4 edges in edge_list have 4 distinct colors."""
    c0 = colors[edge_list[0]]
    c1 = colors[edge_list[1]]
    c2 = colors[edge_list[2]]
    c3 = colors[edge_list[3]]
    return c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3

def count_violations(colors):
    """Count total rainbow 4-cycles."""
    violations = 0
    for fs in four_sets:
        for cyc in fs:
            if is_rainbow(colors, cyc):
                violations += 1
    return violations

def local_search(initial_colors, max_time=55.0):
    """Simulated annealing / local search to minimize violations, then maximize colors."""
    start_time = time.time()
    deadline = start_time + max_time
    
    # Start with a random coloring
    colors = list(initial_colors)
    
    # Count distinct colors used
    def num_colors(c):
        return len(set(c))
    
    best_colors = colors[:]
    best_num_colors = num_colors(colors)
    best_violations = count_violations(colors)
    
    # Phase 1: Eliminate all violations using local search
    # Phase 2: Maximize number of colors while keeping 0 violations
    
    temperature = 5.0
    cooling = 0.9995
    max_iters = 500000
    
    iters = 0
    while iters < max_iters:
        if time.time() > deadline:
            break
        iters += 1
        
        # Pick a random edge to recolor
        e = random.randrange(M)
        old_color = colors[e]
        # Pick a new color: try to use a color that reduces violations
        # Strategy: pick a random color from 0..119, or bias toward existing colors
        new_color = random.randrange(120)
        if new_color == old_color:
            continue
        
        # Compute violation delta
        # Only cycles containing edge e are affected
        delta_v = 0
        for (fs_idx, cyc_idx, pos) in edge_to_cycles[e]:
            cyc = four_sets[fs_idx][cyc_idx]
            # Old state
            old_rainbow = is_rainbow(colors, cyc)
            # Temporarily set new color
            colors[e] = new_color
            new_rainbow = is_rainbow(colors, cyc)
            # Revert
            colors[e] = old_color
            if old_rainbow and not new_rainbow:
                delta_v -= 1
            elif not old_rainbow and new_rainbow:
                delta_v += 1
        
        # Acceptance
        if delta_v < 0:
            accept = True
        elif delta_v == 0:
            accept = True  # Neutral moves help explore
        else:
            # Accept with probability exp(-delta_v / temperature)
            import math
            accept = random.random() < math.exp(-delta_v / temperature)
        
        if accept:
            colors[e] = new_color
            # Update best
            nv = best_violations + delta_v
            if nv < best_violations or (nv == best_violations and num_colors(colors) > best_num_colors):
                best_colors = colors[:]
                best_violations = nv
                best_num_colors = num_colors(colors)
        
        temperature *= cooling
        if temperature < 0.1:
            temperature = 0.1
    
    return best_colors, best_violations, best_num_colors

def greedy_max_colors(colors, max_time=55.0):
    """Given a valid coloring (0 violations), try to increase the number of distinct colors."""
    start_time = time.time()
    deadline = start_time + max_time
    
    colors = colors[:]
    current_violations = count_violations(colors)
    if current_violations > 0:
        return colors
    
    def num_colors(c):
        return len(set(c))
    
    best = colors[:]
    best_nc = num_colors(best)
    
    # Try to change edges to new colors, accepting only if violations stay 0
    for _ in range(200000):
        if time.time() > deadline:
            break
        e = random.randrange(M)
        old_color = colors[e]
        # Try a random new color
        new_color = random.randrange(120)
        if new_color == old_color:
            continue
        
        # Check if this move keeps violations at 0
        delta_v = 0
        for (fs_idx, cyc_idx, pos) in edge_to_cycles[e]:
            cyc = four_sets[fs_idx][cyc_idx]
            old_rainbow = is_rainbow(colors, cyc)
            colors[e] = new_color
            new_rainbow = is_rainbow(colors, cyc)
            colors[e] = old_color
            if old_rainbow and not new_rainbow:
                delta_v -= 1
            elif not old_rainbow and new_rainbow:
                delta_v += 1
        
        if delta_v == 0:
            colors[e] = new_color
            nc = num_colors(colors)
            if nc > best_nc:
                best = colors[:]
                best_nc = nc
        elif delta_v < 0:
            # Shouldn't happen if we start with 0 violations, but just in case
            colors[e] = new_color
            nc = num_colors(colors)
            if nc > best_nc:
                best = colors[:]
                best_nc = nc
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin
    
    random.seed(42)
    
    # Start with all zeros (valid, 1 color)
    base = [0] * M
    
    # Try multiple random initializations
    best_result = base
    best_nc = 1
    best_viol = 0
    
    num_restarts = 10
    per_restart_time = 5.0
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random initial coloring
        init = [random.randrange(120) for _ in range(M)]
        
        # Run local search to eliminate violations
        colors, viol, nc = local_search(init, max_time=per_restart_time)
        
        if viol == 0:
            if nc > best_nc:
                best_result = colors
                best_nc = nc
        
        # If we have time, try to maximize colors on the best valid coloring
        if time.time() < deadline - 15.0 and viol == 0:
            improved = greedy_max_colors(colors, max_time=deadline - time.time() - 5.0)
            nc2 = len(set(improved))
            if nc2 > best_nc:
                best_result = improved
                best_nc = nc2
    
    # Final greedy improvement on the best result
    if time.time() < deadline - 2.0:
        final = greedy_max_colors(best_result, max_time=deadline - time.time() - 1.0)
        if len(set(final)) > best_nc:
            best_result = final
    
    return best_result

# Quick self-test
if __name__ == "__main__":
    result = construct()
    print(f"Colors used: {len(set(result))}")
    print(f"Violations: {count_violations(result)}")
