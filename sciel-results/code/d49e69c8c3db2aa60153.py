"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import itertools
import numpy as np

N = 16
E = N*(N-1)//2  # 120 edges

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i*(2*N-1-i)//2 + j - i - 1

def precompute_cycles():
    """
    Returns:
    - cycles: list of tuples of 4 edge indices forming a 4-cycle
    - edge_to_cycles: for each edge, list of (cycle_idx, position) pairs
    - cycle_other_edges: for each cycle and position, the other 3 edge indices
    """
    cycles = []
    edge_to_cycles = [[] for _ in range(E)]
    cycle_other_edges = []  # cycle_other_edges[ci][pos] = tuple of 3 other edge indices
    
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b, b-c, c-d, d-a
        e1 = (edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a))
        # Cycle 2: a-b, b-d, d-c, c-a
        e2 = (edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a))
        # Cycle 3: a-c, c-b, b-d, d-a
        e3 = (edge_idx(a,c), edge_idx(c,b), edge_idx(b,d), edge_idx(d,a))
        
        for cyc in (e1, e2, e3):
            ci = len(cycles)
            cycles.append(cyc)
            other_edges_for_cycle = []
            for pos in range(4):
                others = tuple(cyc[p] for p in range(4) if p != pos)
                other_edges_for_cycle.append(others)
            cycle_other_edges.append(tuple(other_edges_for_cycle))
            for pos, e in enumerate(cyc):
                edge_to_cycles[e].append((ci, pos))
    return cycles, edge_to_cycles, cycle_other_edges

def count_colors(colors):
    return len(set(colors))

def construct():
    deadline = time.time() + 55.0
    
    cycles, edge_to_cycles, cycle_other_edges = precompute_cycles()
    num_cycles = len(cycles)
    
    # Start with a valid construction: color(i,j) = i (min vertex)
    # This uses 15 colors (0..14).
    colors = np.zeros(E, dtype=int)
    for i in range(N):
        for j in range(i+1, N):
            e = edge_idx(i, j)
            colors[e] = i
    
    best = colors.copy()
    best_count = count_colors(best)
    
    random.seed(42)
    max_color = 119
    
    # Track frequency of each color
    freq = np.zeros(120, dtype=int)
    for c in colors:
        freq[c] += 1
    
    distinct_count = int(np.sum(freq > 0))
    
    # Phase 1: Greedy constraint-aware assignment to maximize diversity
    # Sort edges by number of cycles (fewer first - easier to reassign)
    edge_order_easy = sorted(range(E), key=lambda e: len(edge_to_cycles[e]))
    
    greedy_deadline = time.time() + 10.0
    while time.time() < greedy_deadline:
        e = random.choice(edge_order_easy)
        old_color = int(colors[e])
        
        # Find valid new colors for this edge
        valid_new_colors = []
        for c in range(max_color + 1):
            if c == old_color:
                continue
            valid = True
            for ci, pos in edge_to_cycles[e]:
                others = cycle_other_edges[ci][pos]
                oc0, oc1, oc2 = colors[others[0]], colors[others[1]], colors[others[2]]
                if oc0 != oc1 and oc1 != oc2 and oc0 != oc2:
                    if c != oc0 and c != oc1 and c != oc2:
                        valid = False
                        break
            if valid:
                valid_new_colors.append(c)
        
        if not valid_new_colors:
            continue
        
        # Prefer colors not in used_colors (to maximize diversity)
        new_unused = [c for c in valid_new_colors if freq[c] == 0]
        if new_unused:
            new_color = random.choice(new_unused)
        else:
            min_freq = min(freq[c] for c in valid_new_colors)
            candidates = [c for c in valid_new_colors if freq[c] == min_freq]
            new_color = random.choice(candidates)
        
        # Calculate delta_bad
        delta_bad = 0
        for ci, pos in edge_to_cycles[e]:
            others = cycle_other_edges[ci][pos]
            c1, c2, c3 = colors[others[0]], colors[others[1]], colors[others[2]]
            
            old_set = {old_color, c1, c2, c3}
            old_badness = 1 if len(old_set) == 4 else 0
            
            new_set = {new_color, c1, c2, c3}
            new_badness = 1 if len(new_set) == 4 else 0
            
            if old_badness != new_badness:
                delta_bad += new_badness - old_badness
        
        # Only accept if it doesn't increase badness
        if delta_bad > 0:
            continue
        
        # Apply the move
        colors[e] = new_color
        freq[old_color] -= 1
        freq[new_color] += 1
        
        if freq[old_color] == 0:
            distinct_count -= 1
        if freq[new_color] == 1:
            distinct_count += 1
        
        nc = distinct_count
        if nc > best_count:
            best_count = nc
            best = colors.copy()
    
    # Phase 2: Simulated Annealing with precomputed lookups
    T = 5.0
    cooling = 0.99995
    min_T = 0.005
    total_bad = 0
    
    # Count initial bad cycles
    for ci in range(num_cycles):
        cyc = cycles[ci]
        c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            total_bad += 1
    
    while time.time() < deadline - 5.0:
        if T < min_T:
            T = min_T
            
        # Pick random edge
        e = random.randint(0, E-1)
        old_color = int(colors[e])
        new_color = random.randint(0, max_color)
        
        if old_color == new_color:
            continue
            
        # Calculate delta_bad efficiently
        delta_bad = 0
        for ci, pos in edge_to_cycles[e]:
            others = cycle_other_edges[ci][pos]
            c1, c2, c3 = colors[others[0]], colors[others[1]], colors[others[2]]
            
            # Check if old creates rainbow
            old_rainbow = (old_color != c1 and old_color != c2 and old_color != c3 and 
                          c1 != c2 and c1 != c3 and c2 != c3)
            
            # Check if new creates rainbow
            new_rainbow = (new_color != c1 and new_color != c2 and new_color != c3 and 
                          c1 != c2 and c1 != c3 and c2 != c3)
            
            if old_rainbow != new_rainbow:
                delta_bad += 1 if new_rainbow else -1
                
        # Calculate delta_distinct
        delta_distinct = 0
        if freq[old_color] == 1:
            delta_distinct -= 1
        if freq[new_color] == 0:
            delta_distinct += 1
            
        new_total_bad = total_bad + delta_bad
        new_distinct = distinct_count + delta_distinct
        
        # Acceptance criteria
        accept = False
        
        if new_total_bad == 0 and total_bad == 0:
            if new_distinct > distinct_count:
                accept = True
        elif new_total_bad == 0 and total_bad > 0:
            accept = True
        elif new_total_bad > 0 and total_bad == 0:
            if random.random() < np.exp(-delta_bad / T) * 0.3:
                accept = True
        else:
            if delta_bad < 0:
                accept = True
            elif delta_bad > 0:
                if random.random() < np.exp(-delta_bad / T):
                    accept = True
            else:
                if random.random() < 0.05:
                    accept = True

        if accept:
            colors[e] = new_color
            freq[old_color] -= 1
            freq[new_color] += 1
            distinct_count = new_distinct
            total_bad = new_total_bad
            
            if total_bad == 0:
                if distinct_count > best_count:
                    best_count = distinct_count
                    best = colors.copy()
                    
        T *= cooling
    
    # Phase 3: Final polish - try to add one more color if valid
    if time.time() < deadline - 1.0:
        polish_deadline = min(deadline - 0.5, time.time() + 3.0)
        while time.time() < polish_deadline:
            e = random.randint(0, E-1)
            old_color = int(colors[e])
            # Try each unused color
            unused = [c for c in range(120) if freq[c] == 0]
            if not unused:
                break
            new_color = random.choice(unused)
            if new_color == old_color:
                continue
            
            # Check validity
            valid = True
            for ci, pos in edge_to_cycles[e]:
                others = cycle_other_edges[ci][pos]
                c1, c2, c3 = colors[others[0]], colors[others[1]], colors[others[2]]
                if new_color != c1 and new_color != c2 and new_color != c3 and \
                   c1 != c2 and c1 != c3 and c2 != c3:
                    valid = False
                    break
            
            if valid:
                colors[e] = new_color
                freq[old_color] -= 1
                freq[new_color] += 1
                if freq[old_color] == 0:
                    distinct_count -= 1
                if freq[new_color] == 1:
                    distinct_count += 1
                if distinct_count > best_count:
                    best_count = distinct_count
                    best = colors.copy()
    
    return best.tolist()
# EVOLVE-BLOCK-END
