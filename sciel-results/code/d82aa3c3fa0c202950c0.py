import time
import random
import numpy as np
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLOR = 119

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def precompute_cycles():
    """Precompute all 4-cycles and their edge indices."""
    cycles = []
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        # Three possible 4-cycles on 4 vertices
        cycles.append((edge_idx(a, b), edge_idx(b, c), edge_idx(c, d), edge_idx(d, a)))
        cycles.append((edge_idx(a, b), edge_idx(b, d), edge_idx(d, c), edge_idx(c, a)))
        cycles.append((edge_idx(a, c), edge_idx(c, b), edge_idx(b, d), edge_idx(d, a)))
    return cycles

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    cycles = precompute_cycles()
    num_cycles = len(cycles)
    
    # Reverse index
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cycle in enumerate(cycles):
        for ei in cycle:
            edge_to_cycles[ei].append(ci)
    
    def is_rainbow(colors, cycle):
        return len({colors[cycle[0]], colors[cycle[1]], colors[cycle[2]], colors[cycle[3]]}) == 4
    
    # Initialize
    colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i + 1, N):
            colors[edge_idx(i, j)] = i
    colors[edge_idx(13, 15)] = 15
    colors[edge_idx(14, 15)] = 16
    
    best_colors = colors[:]
    best_div = len(set(best_colors))
    
    # Track violation count per cycle
    cycle_violations = [0] * num_cycles
    total_violations = 0
    for ci, cycle in enumerate(cycles):
        if is_rainbow(colors, cycle):
            cycle_violations[ci] = 1
            total_violations += 1
    
    random.seed(42)
    T = 50.0
    alpha = 0.9995
    max_iter = 2000000
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        edge = random.randint(0, NUM_EDGES - 1)
        old_color = colors[edge]
        
        used = set(colors)
        unused = [c for c in range(MAX_COLOR + 1) if c not in used]
        
        if random.random() < 0.6 and unused:
            new_color = random.choice(unused)
        else:
            new_color = random.randint(0, MAX_COLOR)
        
        if new_color == old_color:
            continue
        
        # Calculate delta in violations for affected cycles
        delta_viol = 0
        for ci in edge_to_cycles[edge]:
            cycle = cycles[ci]
            old_rainbow = is_rainbow(colors, cycle)
            colors[edge] = new_color
            new_rainbow = is_rainbow(colors, cycle)
            colors[edge] = old_color  # revert temporarily
            
            if not old_rainbow and new_rainbow:
                delta_viol += 1
            elif old_rainbow and not new_rainbow:
                delta_viol -= 1
        
        new_total_viol = total_violations + delta_viol
        new_div = len(set(colors)) + (1 if new_color not in used else 0)
        
        # Accept if no violations and more diversity, or with SA probability
        if new_total_viol == 0 and new_div > best_div:
            # Apply
            colors[edge] = new_color
            for ci in edge_to_cycles[edge]:
                if is_rainbow(colors, cycles[ci]):
                    if cycle_violations[ci] == 0:
                        cycle_violations[ci] = 1
                        total_violations += 1
                else:
                    if cycle_violations[ci] == 1:
                        cycle_violations[ci] = 0
                        total_violations -= 1
            best_colors = colors[:]
            best_div = new_div
        elif new_total_viol == 0:
            # Accept valid move
            colors[edge] = new_color
            for ci in edge_to_cycles[edge]:
                if is_rainbow(colors, cycles[ci]):
                    if cycle_violations[ci] == 0:
                        cycle_violations[ci] = 1
                        total_violations += 1
                else:
                    if cycle_violations[ci] == 1:
                        cycle_violations[ci] = 0
                        total_violations -= 1
        elif random.random() < np.exp(-new_total_viol / T):
            # Accept invalid move with SA probability
            colors[edge] = new_color
            for ci in edge_to_cycles[edge]:
                if is_rainbow(colors, cycles[ci]):
                    if cycle_violations[ci] == 0:
                        cycle_violations[ci] = 1
                        total_violations += 1
                else:
                    if cycle_violations[ci] == 1:
                        cycle_violations[ci] = 0
                        total_violations -= 1
        else:
            # Reject
            pass
        
        T = max(T * alpha, 0.01)
    
    return best_colors
