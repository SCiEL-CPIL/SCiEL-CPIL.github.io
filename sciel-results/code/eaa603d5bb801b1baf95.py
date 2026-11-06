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

def precompute_cycles_for_edges():
    """
    Returns a list of lists. cycles_for_edge[e] is a list of lists of 3 edge indices (the other edges in the cycle).
    """
    cycles_for_edge = [[] for _ in range(E)]
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b, b-c, c-d, d-a
        e1 = [edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a)]
        # Cycle 2: a-b, b-d, d-c, c-a
        e2 = [edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a)]
        # Cycle 3: a-c, c-b, b-d, d-a
        e3 = [edge_idx(a,c), edge_idx(c,b), edge_idx(b,d), edge_idx(d,a)]
        
        for cyc in (e1, e2, e3):
            for i in range(4):
                e = cyc[i]
                others = [cyc[j] for j in range(4) if j != i]
                cycles_for_edge[e].append(others)
    return cycles_for_edge

def construct():
    # Safety margin: 60s limit, kill at 90s. Use 55s for computation.
    start_time = time.time()
    deadline = start_time + 55.0
    
    cycles_for_edge = precompute_cycles_for_edges()
    
    random.seed(42)
    
    # Start with valid coloring: color(i,j) = i
    colors = np.zeros(E, dtype=int)
    for i in range(N):
        for j in range(i+1, N):
            e = edge_idx(i, j)
            colors[e] = i
            
    # Frequency of colors
    freq = np.zeros(120, dtype=int)
    for c in colors:
        freq[c] += 1
    distinct_count = np.sum(freq > 0)
    
    # Precompute unique cycles for penalty calculation if needed, 
    # but we can just use the incremental update.
    # We need initial penalty.
    # Since start is valid, initial penalty is 0.
    current_penalty = 0
    
    best_colors = colors.copy()
    best_distinct = distinct_count
    
    # Simulated Annealing
    T = 50.0
    cooling = 0.9999
    min_T = 0.5
    
    while time.time() < deadline:
        if T < min_T:
            T = min_T
            
        # Pick random edge
        e = random.randint(0, E-1)
        old_color = colors[e]
        new_color = random.randint(0, 119)
        
        if old_color == new_color:
            continue
            
        # Calculate delta penalty
        delta_penalty = 0
        for others in cycles_for_edge[e]:
            c0 = colors[others[0]]
            c1 = colors[others[1]]
            c2 = colors[others[2]]
            
            old_set = {old_color, c0, c1, c2}
            new_set = {new_color, c0, c1, c2}
            
            old_bad = (len(old_set) == 4)
            new_bad = (len(new_set) == 4)
            
            if old_bad and not new_bad:
                delta_penalty -= 1
            elif not old_bad and new_bad:
                delta_penalty += 1
                
        # Calculate delta distinct
        delta_distinct = 0
        if freq[old_color] == 1:
            delta_distinct -= 1
        if freq[new_color] == 0:
            delta_distinct += 1
            
        # Energy: We want penalty 0 and max distinct.
        # If penalty > 0, we prioritize reducing penalty.
        # Energy = penalty * 10000 - distinct
        old_energy = current_penalty * 10000 - distinct_count
        new_energy = (current_penalty + delta_penalty) * 10000 - (distinct_count + delta_distinct)
        delta_energy = new_energy - old_energy
        
        accept = False
        if delta_energy < 0:
            accept = True
        else:
            if T > 0:
                prob = np.exp(-delta_energy / T)
            else:
                prob = 0
            if random.random() < prob:
                accept = True
                
        if accept:
            colors[e] = new_color
            freq[old_color] -= 1
            freq[new_color] += 1
            distinct_count += delta_distinct
            current_penalty += delta_penalty
            
            # Update best if valid (penalty 0) and distinct is higher
            if current_penalty == 0:
                if distinct_count > best_distinct:
                    best_distinct = distinct_count
                    best_colors = colors.copy()
                    
    return best_colors.tolist()
# EVOLVE-BLOCK-END
