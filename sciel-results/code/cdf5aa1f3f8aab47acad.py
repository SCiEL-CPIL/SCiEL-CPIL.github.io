"""Optimized program: simulated annealing with diversity pressure for 4-cycle constraint."""
import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

# Precompute edge indices
def _edge_idx(i, j):
    """Map (i,j) with i<j to linear edge index."""
    # edges for vertex 0: (0,1)..(0,15) -> 0..14
    # edges for vertex 1: (1,2)..(1,15) -> 15..28
    # ...
    # offset for vertex i: i*N - i*(i+1)//2
    return i * N - i * (i + 1) // 2 + (j - i - 1)

# Precompute all 4-cycles: for each set of 4 vertices {a,b,c,d} with a<b<c<d,
# there are 3 distinct 4-cycles (3 ways to pair them into a cycle).
# Cycle 1: a-b-c-d-a (edges: ab, bc, cd, da)
# Cycle 2: a-b-d-c-a (edges: ab, bd, dc, ca)
# Cycle 3: a-c-b-d-a (edges: ac, cb, bd, da)

def _build_cycles():
    """Build list of 4-cycles, each as a tuple of 4 edge indices."""
    cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # Cycle 1: a-b-c-d-a
                    e1 = (_edge_idx(a, b), _edge_idx(b, c), _edge_idx(c, d), _edge_idx(a, d))
                    # Cycle 2: a-b-d-c-a
                    e2 = (_edge_idx(a, b), _edge_idx(b, d), _edge_idx(c, d), _edge_idx(a, c))
                    # Cycle 3: a-c-b-d-a
                    e3 = (_edge_idx(a, c), _edge_idx(b, c), _edge_idx(b, d), _edge_idx(a, d))
                    cycles.append(e1)
                    cycles.append(e2)
                    cycles.append(e3)
    return cycles

CYCLES = _build_cycles()
NUM_CYCLES = len(CYCLES)  # C(16,4)*3 = 1820*3 = 5460

# For each edge, precompute which cycles it belongs to
def _build_edge_to_cycles():
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cyc in enumerate(CYCLES):
        for e in cyc:
            edge_to_cycles[e].append(ci)
    return edge_to_cycles

EDGE_TO_CYCLES = _build_edge_to_cycles()

# For each edge, precompute the other 3 edges in each cycle it belongs to
def _build_cycle_neighbors():
    """For each edge e and each cycle c it's in, store the other 3 edges."""
    cycle_neighbors = [{} for _ in range(NUM_EDGES)]
    for ci, cyc in enumerate(CYCLES):
        for i, e in enumerate(cyc):
            others = [cyc[j] for j in range(4) if j != i]
            cycle_neighbors[e].setdefault(ci, others)
    return cycle_neighbors

CYCLE_NEIGHBORS = _build_cycle_neighbors()


def _count_violations(colors):
    """Count number of 4-cycles with 4 distinct colors."""
    violations = 0
    for cyc in CYCLES:
        c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            violations += 1
    return violations


def _count_distinct(colors):
    """Count number of distinct colors used."""
    return len(set(colors))


def construct():
    """
    Use simulated annealing to find a valid coloring (no 4-cycle with 4 distinct colors)
    maximizing the number of distinct colors.
    """
    deadline = time.time() + 80.0  # 10s safety margin
    
    # Initialize with random colors from a large pool to encourage diversity
    # Use colors 0..119
    colors = np.random.randint(0, 120, size=NUM_EDGES, dtype=np.int32)
    
    # Precompute for fast violation checking
    # Store colors as a list for easier access
    col_list = colors.tolist()
    
    # Compute initial violations
    cur_viol = 0
    for cyc in CYCLES:
        c0, c1, c2, c3 = col_list[cyc[0]], col_list[cyc[1]], col_list[cyc[2]], col_list[cyc[3]]
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            cur_viol += 1
    
    best_colors = col_list[:]
    best_viol = cur_viol
    best_distinct = _count_distinct(col_list)
    
    # Simulated annealing parameters
    T = 50.0  # initial temperature
    cooling = 0.999
    min_T = 0.01
    
    # Track best valid solution
    best_valid_colors = None
    best_valid_distinct = 0
    
    if cur_viol == 0:
        best_valid_colors = col_list[:]
        best_valid_distinct = best_distinct
    
    step = 0
    while time.time() < deadline:
        step += 1
        
        # Pick a random edge to change
        e = np.random.randint(0, NUM_EDGES)
        old_color = col_list[e]
        
        # Try a new color - bias towards unused or less-used colors to encourage diversity
        # With some probability, try a completely new color
        if np.random.random() < 0.3:
            new_color = np.random.randint(0, 120)
        else:
            new_color = np.random.randint(0, 120)
        
        if new_color == old_color:
            continue
        
        # Compute change in violations
        delta_viol = 0
        for ci in EDGE_TO_CYCLES[e]:
            others = CYCLE_NEIGHBORS[e][ci]
            o0, o1, o2 = others[0], others[1], others[2]
            
            # Old cycle colors
            oc0, oc1, oc2, oc3 = old_color, col_list[o0], col_list[o1], col_list[o2]
            old_distinct = len(set([oc0, oc1, oc2, oc3]))
            old_viol = 1 if old_distinct == 4 else 0
            
            # New cycle colors
            nc0, nc1, nc2, nc3 = new_color, col_list[o0], col_list[o1], col_list[o2]
            new_distinct = len(set([nc0, nc1, nc2, nc3]))
            new_viol = 1 if new_distinct == 4 else 0
            
            delta_viol += new_viol - old_viol
        
        # Compute change in distinct count
        # Check if old_color is still used elsewhere
        old_count = col_list.count(old_color)
        new_count = col_list.count(new_color)
        
        # Remove old, add new
        delta_distinct = 0
        if old_count == 1:  # old color disappears
            delta_distinct -= 1
        if new_count == 0:  # new color appears for first time
            delta_distinct += 1
        
        # Acceptance criterion
        # We want to minimize violations and maximize distinct colors
        # Energy = violations * large_penalty - distinct * small_bonus
        # Or use a combined score
        
        # Primary goal: reduce violations. Secondary: increase distinct.
        if delta_viol < 0:
            # Better violations - always accept
            accept = True
        elif delta_viol == 0:
            # Same violations - accept if it increases distinct, or with some probability
            if delta_distinct > 0:
                accept = True
            else:
                # Accept with probability based on temperature
                # Small chance to accept even if distinct doesn't improve
                accept = np.random.random() < 0.1
        else:
            # Worse violations - accept with probability exp(-delta_viol / T)
            # But only if it significantly improves distinct
            if delta_distinct > 0 and delta_viol <= 3:
                prob = np.exp(-delta_viol / max(T, 0.01)) * (1 + delta_distinct * 0.5)
                accept = np.random.random() < min(prob, 1.0)
            else:
                accept = np.random.random() < np.exp(-delta_viol / max(T, 0.01))
        
        if accept:
            col_list[e] = new_color
            cur_viol += delta_viol
            best_distinct = _count_distinct(col_list)  # Recompute occasionally
            # Actually, let's track distinct more carefully
            # For now, recompute when we have a valid solution
            
            if cur_viol == 0:
                d = _count_distinct(col_list)
                if d > best_valid_distinct:
                    best_valid_distinct = d
                    best_valid_colors = col_list[:]
            
            # Update best overall (prefer low violations, then high distinct)
            if cur_viol < best_viol or (cur_viol == best_viol and best_distinct > best_distinct):
                best_viol = cur_viol
                best_distinct = best_distinct
                best_colors = col_list[:]
        
        # Cool down
        T *= cooling
        if T < min_T:
            T = min_T
        
        # Periodic restart if stuck
        if step % 5000 == 0 and cur_viol > 0:
            # Try a local repair: for each violating cycle, change one edge
            for _ in range(100):
                if time.time() >= deadline:
                    break
                # Find a random violating cycle
                for _ in range(50):
                    ci = np.random.randint(0, NUM_CYCLES)
                    cyc = CYCLES[ci]
                    c0, c1, c2, c3 = col_list[cyc[0]], col_list[cyc[1]], col_list[cyc[2]], col_list[cyc[3]]
                    if len(set([c0, c1, c2, c3])) == 4:
                        # Pick a random edge in this cycle and change its color to match another
                        ei = np.random.randint(0, 4)
                        e_idx = cyc[ei]
                        # Change to match one of the other edges
                        match_ei = np.random.choice([x for x in range(4) if x != ei])
                        col_list[e_idx] = col_list[cyc[match_ei]]
                        break
        
        # If we have a valid solution with high distinct, try to improve
        if best_valid_colors is not None and best_valid_distinct >= 50:
            # Do some targeted moves to try to add more colors
            if step % 1000 == 0:
                for _ in range(50):
                    if time.time() >= deadline:
                        break
                    e = np.random.randint(0, NUM_EDGES)
                    old_c = col_list[e]
                    # Try to change to a color not currently used
                    used_colors = set(col_list)
                    if len(used_colors) < 120:
                        unused = [c for c in range(120) if c not in used_colors]
                        if unused:
                            new_c = random.choice(unused)
                        else:
                            new_c = np.random.randint(0, 120)
                    else:
                        new_c = np.random.randint(0, 120)
                    
                    if new_c == old_c:
                        continue
                    
                    # Check if this change keeps validity
                    delta_v = 0
                    for ci in EDGE_TO_CYCLES[e]:
                        others = CYCLE_NEIGHBORS[e][ci]
                        o0, o1, o2 = others[0], others[1], others[2]
                        oc0, oc1, oc2, oc3 = old_c, col_list[o0], col_list[o1], col_list[o2]
                        old_v = 1 if len(set([oc0, oc1, oc2, oc3])) == 4 else 0
                        nc0, nc1, nc2, nc3 = new_c, col_list[o0], col_list[o1], col_list[o2]
                        new_v = 1 if len(set([nc0, nc1, nc2, nc3])) == 4 else 0
                        delta_v += new_v - old_v
                    
                    if delta_v <= 0:
                        col_list[e] = new_c
                        cur_viol += delta_v
                        if cur_viol == 0:
                            d = _count_distinct(col_list)
                            if d > best_valid_distinct:
                                best_valid_distinct = d
                                best_valid_colors = col_list[:]
    
    # Return the best valid solution if found, otherwise best overall
    if best_valid_colors is not None:
        return best_valid_colors
    else:
        # Return best_colors (may have violations, but that's the best we found)
        return best_colors
