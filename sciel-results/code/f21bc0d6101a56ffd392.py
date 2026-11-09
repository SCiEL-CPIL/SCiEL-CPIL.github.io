"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 119

def _edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

def _precompute_cycles():
    cycles = []
    from itertools import combinations
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        e1 = (_edge_idx(a,b), _edge_idx(b,c), _edge_idx(c,d), _edge_idx(d,a))
        e2 = (_edge_idx(a,b), _edge_idx(b,d), _edge_idx(d,c), _edge_idx(c,a))
        e3 = (_edge_idx(a,c), _edge_idx(c,b), _edge_idx(b,d), _edge_idx(d,a))
        cycles.append(e1)
        cycles.append(e2)
        cycles.append(e3)
    return cycles

_CYCLES = _precompute_cycles()
NUM_CYCLES = len(_CYCLES)

_EDGE_CYCLES = [[] for _ in range(NUM_EDGES)]
for cycle in _CYCLES:
    for e in cycle:
        _EDGE_CYCLES[e].append(cycle)

def _check_single_change(colors, edge, new_color):
    old_color = colors[edge]
    if old_color == new_color:
        return True
    colors[edge] = new_color
    for (e1, e2, e3, e4) in _EDGE_CYCLES[edge]:
        c1, c2, c3, c4 = colors[e1], colors[e2], colors[e3], colors[e4]
        if c1 != c2 and c1 != c3 and c1 != c4 and c2 != c3 and c2 != c4 and c3 != c4:
            colors[edge] = old_color
            return False
    colors[edge] = old_color
    return True

def _count_colors(colors):
    return len(set(colors))

def _initial_coloring():
    colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i + 1, N):
            gi, gj = i // 4, j // 4
            if gi == gj:
                colors[_edge_idx(i, j)] = 4 * gi
            else:
                k, l = min(gi, gj), max(gi, gj)
                colors[_edge_idx(i, j)] = 4 * k + (l - 1)
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    colors = _initial_coloring()
    best_colors = colors[:]
    best_count = _count_colors(best_colors)
    
    random.seed(42)
    
    # Precompute used colors set for efficiency
    used = set(colors)
    
    # Local search with restarts
    max_iter_per_restart = 50000
    num_restarts = 5
    
    for restart in range(num_restarts):
        if restart > 0:
            # Perturb: randomly change some edges
            colors = best_colors[:]
            for _ in range(20):
                e = random.randint(0, NUM_EDGES - 1)
                c = random.randint(0, MAX_COLOR)
                if _check_single_change(colors, e, c):
                    colors[e] = c
            used = set(colors)
        
        for iteration in range(max_iter_per_restart):
            if time.time() > deadline:
                return best_colors
            
            edge = random.randint(0, NUM_EDGES - 1)
            old_color = colors[edge]
            
            # Try to introduce a new color
            found_new = False
            for _ in range(10):
                new_color = random.randint(0, MAX_COLOR)
                if new_color in used:
                    continue
                if _check_single_change(colors, edge, new_color):
                    colors[edge] = new_color
                    used.add(new_color)
                    new_count = len(used)
                    if new_count > best_count:
                        best_count = new_count
                        best_colors = colors[:]
                    found_new = True
                    break
            
            if not found_new:
                # Try changing to a different existing color for exploration
                if random.random() < 0.3:
                    new_color = random.randint(0, MAX_COLOR)
                    if new_color != old_color:
                        if _check_single_change(colors, edge, new_color):
                            if new_color not in used:
                                # Shouldn't happen
                                pass
                            colors[edge] = new_color
                            # Update used set
                            # Check if old_color is still used
                            still_used = False
                            for e2 in range(NUM_EDGES):
                                if e2 != edge and colors[e2] == old_color:
                                    still_used = True
                                    break
                            if not still_used:
                                used.discard(old_color)
    
    return best_colors
