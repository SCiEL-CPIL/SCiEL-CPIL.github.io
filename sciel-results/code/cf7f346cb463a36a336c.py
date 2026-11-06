"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
E = N * (N - 1) // 2  # 120 edges

# EVOLVE-BLOCK-START
import random
import time
import numpy as np

# Precompute edge index mapping
_edge_idx = {}
_eid = 0
for i in range(N):
    for j in range(i + 1, N):
        _edge_idx[(i, j)] = _eid
        _eid += 1

# Precompute cycles for each edge
# cycles_of_edge[e] = list of tuples (e1, e2, e3) such that {e, e1, e2, e3} is a 4-cycle
cycles_of_edge = [[] for _ in range(E)]

# Also precompute: for each edge, the set of other edges in its cycles (for fast violation counting)
# And for each cycle, store it as a tuple of 4 edge indices
all_cycles = []
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            for d in range(c + 1, N):
                # Cycle 1: a-b, b-c, c-d, d-a
                e1 = _edge_idx[(a, b)]
                e2 = _edge_idx[(b, c)]
                e3 = _edge_idx[(c, d)]
                e4 = _edge_idx[(a, d)]
                cyc = (e1, e2, e3, e4)
                all_cycles.append(cyc)
                for e in (e1, e2, e3, e4):
                    others = [x for x in (e1, e2, e3, e4) if x != e]
                    cycles_of_edge[e].append(tuple(others))
                
                # Cycle 2: a-b, b-d, c-d, a-c
                e1 = _edge_idx[(a, b)]
                e2 = _edge_idx[(b, d)]
                e3 = _edge_idx[(c, d)]
                e4 = _edge_idx[(a, c)]
                cyc = (e1, e2, e3, e4)
                all_cycles.append(cyc)
                for e in (e1, e2, e3, e4):
                    others = [x for x in (e1, e2, e3, e4) if x != e]
                    cycles_of_edge[e].append(tuple(others))
                
                # Cycle 3: a-c, b-c, b-d, a-d
                e1 = _edge_idx[(a, c)]
                e2 = _edge_idx[(b, c)]
                e3 = _edge_idx[(b, d)]
                e4 = _edge_idx[(a, d)]
                cyc = (e1, e2, e3, e4)
                all_cycles.append(cyc)
                for e in (e1, e2, e3, e4):
                    others = [x for x in (e1, e2, e3, e4) if x != e]
                    cycles_of_edge[e].append(tuple(others))

def count_violations(colors):
    """Count the number of rainbow 4-cycles."""
    v = 0
    for (e1, e2, e3, e4) in all_cycles:
        if len({colors[e1], colors[e2], colors[e3], colors[e4]}) == 4:
            v += 1
    return v

def count_distinct(colors):
    """Count the number of distinct colors used."""
    return len(set(colors))

def is_valid_change(colors, edge, new_color):
    """
    Check if setting colors[edge] = new_color keeps all 4-cycles valid.
    Only checks cycles containing 'edge'.
    """
    for (o1, o2, o3) in cycles_of_edge[edge]:
        c1, c2, c3 = colors[o1], colors[o2], colors[o3]
        # The cycle has colors: new_color, c1, c2, c3
        # Must NOT have 4 distinct colors
        if len({new_color, c1, c2, c3}) == 4:
            return False
    return True

def greedy_add_colors(colors, rng, deadline):
    """Greedy phase: try to add new colors one by one."""
    current_count = count_distinct(colors)
    for target_color in range(15, 120):
        if time.time() > deadline - 5:
            break
        if target_color < current_count:
            continue
            
        found = False
        edges_to_try = list(range(E))
        rng.shuffle(edges_to_try)
        
        for e in edges_to_try:
            if colors[e] == target_color:
                continue
            if is_valid_change(colors, e, target_color):
                colors[e] = target_color
                current_count = count_distinct(colors)
                found = True
                break
        if not found:
            pass
    return colors, current_count

def simulated_annealing(colors, rng, deadline):
    """
    Simulated annealing with violation-aware acceptance.
    Energy = -alpha * distinct_count + beta * violations
    We want to minimize energy, i.e., maximize distinct_count and minimize violations.
    """
    current = colors[:]
    current_distinct = count_distinct(current)
    current_violations = count_violations(current)
    
    best = current[:]
    best_distinct = current_distinct
    best_violations = current_violations
    
    # Temperature schedule
    t_start = 50.0
    t_end = 0.1
    n_steps = 100000
    alpha = 1.0  # Weight for distinct colors
    beta = 10.0  # Weight for violations (penalty)
    
    # Precompute set of used colors for efficiency
    used_colors = set(current)
    
    step = 0
    while time.time() < deadline - 2:
        # Temperature (geometric cooling)
        if step < n_steps:
            temp = t_start * (t_end / t_start) ** (step / n_steps)
        else:
            temp = t_end
        
        # Pick an edge
        e = rng.randint(0, E - 1)
        old_color = current[e]
        
        # Choose new color with bias towards unused colors
        r = rng.random()
        if r < 0.4:
            # Pick an unused color
            unused = [c for c in range(120) if c not in used_colors]
            if unused:
                new_color = rng.choice(unused)
            else:
                new_color = rng.randint(0, 119)
        elif r < 0.7:
            # Pick a random color from 0..119
            new_color = rng.randint(0, 119)
        else:
            # Pick a color that appears exactly once (to potentially free it up)
            from collections import Counter
            color_counts = Counter(current)
            singleton_colors = [c for c, cnt in color_counts.items() if cnt == 1]
            if singleton_colors:
                new_color = rng.choice(singleton_colors)
            else:
                new_color = rng.randint(0, 119)
        
        if new_color == old_color:
            step += 1
            continue
        
        # Check validity of the change
        valid = is_valid_change(current, e, new_color)
        
        # Calculate energy change
        old_energy = -alpha * current_distinct + beta * current_violations
        
        if valid:
            # Apply change
            current[e] = new_color
            new_distinct = count_distinct(current)
            new_violations = count_violations(current)
            new_energy = -alpha * new_distinct + beta * new_violations
            
            delta_e = new_energy - old_energy
            
            # Accept if better, or with probability exp(-delta_e / temp)
            if delta_e <= 0 or rng.random() < np.exp(-delta_e / max(temp, 1e-10)):
                # Accepted
                current_distinct = new_distinct
                current_violations = new_violations
                # Update used_colors
                if old_color not in current:
                    used_colors.discard(old_color)
                used_colors.add(new_color)
                
                # Update best if valid and better
                if current_violations == 0 and current_distinct > best_distinct:
                    best_distinct = current_distinct
                    best_violations = 0
                    best = current[:]
            else:
                # Revert
                current[e] = old_color
        else:
            # Invalid change: calculate how many new violations would be introduced
            # For now, just reject invalid moves unless temperature is very high
            # In the future, we could accept invalid moves at high temperature to escape local minima
            # But for safety, we only accept valid moves
            pass
        
        step += 1
    
    return best, best_distinct, best_violations

def construct():
    deadline = time.time() + 55  # Safety margin for 60s limit
    
    # Initial coloring: color by min(i, j)
    # This creates a valid coloring with 15 colors (0 to 14)
    colors = [0] * E
    for i in range(N):
        for j in range(i + 1, N):
            colors[_edge_idx[(i, j)]] = min(i, j)
    
    best_colors = colors[:]
    best_count = count_distinct(best_colors)
    
    rng = random.Random(42)
    
    # Phase 1: Greedy addition of new colors
    colors, current_count = greedy_add_colors(colors, rng, deadline)
    if current_count > best_count:
        best_count = current_count
        best_colors = colors[:]
    
    # Phase 2: Simulated Annealing with multiple restarts
    # Use remaining time for SA
    remaining = deadline - time.time() - 5
    if remaining > 10:
        # Run SA with the current best as starting point
        sa_result, sa_distinct, sa_violations = simulated_annealing(best_colors, rng, deadline)
        if sa_violations == 0 and sa_distinct > best_count:
            best_count = sa_distinct
            best_colors = sa_result
    
    # Phase 3: More restarts with different initializations if time permits
    remaining = deadline - time.time() - 3
    restarts = 0
    while remaining > 15 and restarts < 5:
        restarts += 1
        # Create a new random valid coloring
        new_colors = [0] * E
        # Start with min(i,j) coloring
        for i in range(N):
            for j in range(i + 1, N):
                new_colors[_edge_idx[(i, j)]] = min(i, j)
        
        # Do some random valid moves to diversify
        for _ in range(200):
            if time.time() > deadline - 10:
                break
            e = rng.randint(0, E - 1)
            old_c = new_colors[e]
            new_c = rng.randint(0, 119)
            if new_c != old_c and is_valid_change(new_colors, e, new_c):
                new_colors[e] = new_c
        
        # Greedy add
        new_colors, nc = greedy_add_colors(new_colors, rng, deadline)
        if nc > best_count:
            best_count = nc
            best_colors = new_colors[:]
        
        # SA on this
        remaining = deadline - time.time() - 5
        if remaining > 10:
            sa_result, sa_distinct, sa_violations = simulated_annealing(new_colors, rng, deadline)
            if sa_violations == 0 and sa_distinct > best_count:
                best_count = sa_distinct
                best_colors = sa_result
        
        remaining = deadline - time.time() - 3
    
    return best_colors
# EVOLVE-BLOCK-END
