import time
import random
import math
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLOR = 120

# Precompute edge index mapping
def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

# Precompute all 4-cycles (3 per 4-vertex subset)
CYCLES = []
for a, b, c, d in combinations(range(N), 4):
    # Three distinct 4-cycles on {a,b,c,d}
    CYCLES.append((edge_idx(a, b), edge_idx(b, c), edge_idx(c, d), edge_idx(d, a)))
    CYCLES.append((edge_idx(a, b), edge_idx(b, d), edge_idx(d, c), edge_idx(c, a)))
    CYCLES.append((edge_idx(a, c), edge_idx(c, b), edge_idx(b, d), edge_idx(d, a)))

NUM_CYCLES = len(CYCLES)

def count_distinct_colors(colors):
    return len(set(colors))

def count_violations(colors):
    """Count 4-cycles with 4 distinct edge colors."""
    v = 0
    for cycle in CYCLES:
        c0 = colors[cycle[0]]
        c1 = colors[cycle[1]]
        c2 = colors[cycle[2]]
        c3 = colors[cycle[3]]
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            v += 1
    return v

def initial_coloring():
    """Create initial coloring with many distinct colors but likely violations."""
    colors = [0] * NUM_EDGES
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            # Use a pattern that creates many distinct colors
            colors[idx] = (i * N + j) % MAX_COLOR
            idx += 1
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    random.seed(42)
    
    best_colors = initial_coloring()
    best_score = -10**9
    best_distinct = 0
    best_violations = count_violations(best_colors)
    
    # Evaluate initial
    if best_violations == 0:
        best_distinct = count_distinct_colors(best_colors)
        best_score = best_distinct
    
    # Simulated annealing
    T = 100.0
    T_min = 0.01
    cooling = 0.9995
    
    # Precompute for speed: for each edge, which cycles contain it
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cycle in enumerate(CYCLES):
        for edge in cycle:
            edge_to_cycles[edge].append(ci)
    
    # Current state
    cur_colors = best_colors[:]
    cur_violations = best_violations
    cur_distinct = count_distinct_colors(cur_colors)
    cur_score = -10**9 if cur_violations > 0 else cur_distinct
    
    step = 0
    no_improve = 0
    max_no_improve = 50000
    
    while T > T_min:
        # Check deadline
        if time.time() > deadline:
            break
        
        # Pick a random edge
        edge = random.randrange(NUM_EDGES)
        old_color = cur_colors[edge]
        
        # Try a new color
        new_color = random.randrange(MAX_COLOR)
        if new_color == old_color:
            continue
        
        # Compute delta in violations for affected cycles
        delta_viol = 0
        affected = edge_to_cycles[edge]
        
        # Check before
        for ci in affected:
            cycle = CYCLES[ci]
            c0 = cur_colors[cycle[0]]
            c1 = cur_colors[cycle[1]]
            c2 = cur_colors[cycle[2]]
            c3 = cur_colors[cycle[3]]
            if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
                delta_viol -= 1  # This violation will be removed if new color matches one of the others
        
        # Temporarily change
        cur_colors[edge] = new_color
        
        # Check after
        for ci in affected:
            cycle = CYCLES[ci]
            c0 = cur_colors[cycle[0]]
            c1 = cur_colors[cycle[1]]
            c2 = cur_colors[cycle[2]]
            c3 = cur_colors[cycle[3]]
            if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
                delta_viol += 1
        
        new_violations = cur_violations + delta_viol
        
        # Compute new distinct count (approximate - track set membership)
        # We'll recompute only if accepting
        if new_violations <= 0:
            # No violations, check distinct colors
            new_distinct = count_distinct_colors(cur_colors)
            new_score = new_distinct
        else:
            new_score = -10**9
            new_distinct = 0
        
        # Accept/reject
        if new_score >= cur_score:
            accept = True
        else:
            # Accept with probability exp((new_score - cur_score) / T)
            # But scores can be negative, so handle carefully
            delta = new_score - cur_score
            if T > 0:
                accept_prob = math.exp(delta / T)
            else:
                accept_prob = 0.0
            accept = random.random() < accept_prob
        
        if accept:
            cur_violations = new_violations
            cur_distinct = new_distinct if new_violations == 0 else count_distinct_colors(cur_colors)
            cur_score = cur_distinct if cur_violations == 0 else -10**9
            
            # Update best
            if cur_score > best_score:
                best_score = cur_score
                best_colors = cur_colors[:]
                best_violations = cur_violations
                best_distinct = cur_distinct
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve > max_no_improve and T < 1.0:
                break
        else:
            cur_colors[edge] = old_color
        
        T *= cooling
        step += 1
        
        if step % 10000 == 0 and time.time() > deadline:
            break
    
    # Final check: return best valid coloring found, or fallback
    if best_violations == 0:
        return best_colors
    
    # If no valid coloring found, try a simple valid one
    # All edges same color is trivially valid
    return [0] * NUM_EDGES
