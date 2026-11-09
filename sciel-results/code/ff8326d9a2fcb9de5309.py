import time
import random
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2
ALL_COLORS = set(range(120))

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Precompute edge indices
    edge_to_idx = {}
    idx_to_edge = []
    for i in range(N):
        for j in range(i + 1, N):
            edge_to_idx[(i, j)] = len(idx_to_edge)
            idx_to_edge.append((i, j))
    
    # Precompute 4-cycles
    cycles = []
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    
    for v1, v2, v3, v4 in itertools.combinations(range(N), 4):
        e12 = edge_to_idx[(v1, v2)]
        e13 = edge_to_idx[(v1, v3)]
        e14 = edge_to_idx[(v1, v4)]
        e23 = edge_to_idx[(v2, v3)]
        e24 = edge_to_idx[(v2, v4)]
        e34 = edge_to_idx[(v3, v4)]
        
        c1 = [e12, e23, e34, e14]
        cycles.append(c1)
        for e in c1:
            edge_to_cycles[e].append(len(cycles) - 1)
            
        c2 = [e12, e24, e34, e13]
        cycles.append(c2)
        for e in c2:
            edge_to_cycles[e].append(len(cycles) - 1)
            
        c3 = [e13, e23, e24, e14]
        cycles.append(c3)
        for e in c3:
            edge_to_cycles[e].append(len(cycles) - 1)

    def count_violations(colors):
        """Count number of 4-cycles with 4 distinct colors."""
        count = 0
        for cyc in cycles:
            c0 = colors[cyc[0]]
            c1 = colors[cyc[1]]
            c2 = colors[cyc[2]]
            c3 = colors[cyc[3]]
            if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
                count += 1
        return count

    def is_valid(colors):
        return count_violations(colors) == 0

    def count_colors(colors):
        return len(set(colors))

    def energy(colors):
        """Energy = violations * 1000 - colors_used. Lower is better."""
        v = count_violations(colors)
        c = count_colors(colors)
        return v * 1000 - c

    random.seed(42)
    
    # Smart initial construction: partition vertices into groups, use group-based coloring
    def smart_initial():
        colors = [0] * NUM_EDGES
        # Partition 16 vertices into 4 groups of 4
        groups = [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15]]
        vertex_group = [0]*N
        for gi, g in enumerate(groups):
            for v in g:
                vertex_group[v] = gi
        
        # Color edges: within-group edges get colors 0-3, between-group edges get colors 4+
        color_counter = 4
        for i in range(N):
            for j in range(i+1, N):
                eidx = edge_to_idx[(i,j)]
                gi = vertex_group[i]
                gj = vertex_group[j]
                if gi == gj:
                    # Within group: use a few colors
                    colors[eidx] = (i + j) % 4
                else:
                    # Between groups: assign unique-ish colors
                    colors[eidx] = color_counter
                    color_counter += 1
                    if color_counter >= 120:
                        color_counter = 4
        return colors

    # Try multiple initial constructions
    best_score = -1
    best_colors = None
    best_energy = float('inf')
    
    # Initial: all same color (valid, 1 color)
    init0 = [0] * NUM_EDGES
    e0 = energy(init0)
    best_energy = e0
    best_colors = init0[:]
    best_score = 1
    
    # Try smart initial
    for attempt in range(10):
        if time.time() > deadline - 30:
            break
        colors = smart_initial()
        # Shuffle the colors to break symmetry
        perm = list(range(120))
        random.shuffle(perm)
        colors = [perm[c] for c in colors]
        e = energy(colors)
        if e < best_energy:
            best_energy = e
            best_colors = colors[:]
            best_score = count_colors(colors)
    
    # Simulated annealing
    current = best_colors[:]
    current_energy = best_energy
    current_score = best_score
    
    T = 100.0  # Initial temperature
    T_min = 0.01
    cooling = 0.995
    
    # Precompute for fast violation checking: for each edge, which cycles it's in
    # and for each cycle, its edges
    cycle_edges = cycles  # list of lists of edge indices
    
    def delta_violation_for_edge_change(colors, e_idx, old_c, new_c):
        """Compute change in violation count when changing edge e_idx from old_c to new_c."""
        delta = 0
        for cyc_idx in edge_to_cycles[e_idx]:
            cyc = cycle_edges[cyc_idx]
            # Get the other 3 edges
            other_edges = [cyc[i] for i in range(4) if cyc[i] != e_idx]
            c1 = colors[other_edges[0]]
            c2 = colors[other_edges[1]]
            c3 = colors[other_edges[2]]
            
            # Check if old coloring was a violation
            old_colors = [old_c, c1, c2, c3]
            old_is_violation = len(set(old_colors)) == 4
            
            # Check if new coloring is a violation
            new_colors = [new_c, c1, c2, c3]
            new_is_violation = len(set(new_colors)) == 4
            
            if new_is_violation and not old_is_violation:
                delta += 1
            elif old_is_violation and not new_is_violation:
                delta -= 1
        
        return delta

    max_iterations = 50000
    iteration = 0
    
    while time.time() < deadline - 5 and iteration < max_iterations:
        iteration += 1
        
        # Pick a random edge
        e_idx = random.randint(0, NUM_EDGES - 1)
        old_c = current[e_idx]
        
        # Pick a new color
        if random.random() < 0.7:
            # Try to use a color that's already in use or a new one
            if random.random() < 0.5:
                new_c = random.randint(0, 119)
            else:
                new_c = random.choice(current)
        else:
            new_c = random.randint(0, 119)
        
        if new_c == old_c:
            continue
        
        # Compute delta in violations
        delta_v = delta_violation_for_edge_change(current, e_idx, old_c, new_c)
        
        # Compute delta in colors used
        old_used = set(current)
        current[e_idx] = new_c
        new_used = set(current)
        delta_c = len(new_used) - len(old_used)
        current[e_idx] = old_c  # Revert for now
        
        delta_e = delta_v * 1000 - delta_c
        
        if delta_e <= 0:
            # Accept
            current[e_idx] = new_c
            current_energy += delta_e
            new_score = count_colors(current)
            if new_score > best_score:
                best_score = new_score
                best_colors = current[:]
        else:
            # Accept with probability exp(-delta_e / T)
            import math
            prob = math.exp(-delta_e / T)
            if random.random() < prob:
                current[e_idx] = new_c
                current_energy += delta_e
                new_score = count_colors(current)
                if new_score > best_score:
                    best_score = new_score
                    best_colors = current[:]
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_colors
