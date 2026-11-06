import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Target n=26 is the known maximum (Lazebnik-Woldar).
    # Score for n=26 is 26/11 ~ 2.36.
    # We must find n=26.
    
    for n in [26, 25, 24, 23, 22, 21, 20]:
        if time.time() > deadline - 5:
            break
            
        result = solve_n(n, time_limit=100)
        if result is not None:
            return result
            
    # Fallback to seed n=11
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

def solve_n(n, time_limit):
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Precompute edge index
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx
        
    # neighbors[c][v] is a bitmask of neighbors of v in color c
    neighbors = [[0] * n for _ in range(5)]
    colors = [0] * num_edges
    
    def update_neighbors(idx, old_color, new_color):
        i, j = edges[idx]
        neighbors[old_color][i] &= ~(1 << j)
        neighbors[old_color][j] &= ~(1 << i)
        neighbors[new_color][i] |= (1 << j)
        neighbors[new_color][j] |= (1 << i)
        
    def calc_violations():
        violations = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = neighbors[c][u] & neighbors[c][v]
                    k = common.bit_count()
                    if k >= 2:
                        violations += k * (k - 1) // 2
        return violations

    def get_violating_edge_single():
        # Try to find an edge (u,v) such that u and v have >= 2 common neighbors in some color
        for _ in range(10):
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v: continue
            if u > v: u, v = v, u
            for c in range(5):
                if (neighbors[c][u] & neighbors[c][v]).bit_count() >= 2:
                    return edge_idx[(u,v)]
        return random.randint(0, num_edges - 1)

    # 1. Greedy Initialization
    edge_order = list(range(num_edges))
    random.shuffle(edge_order)
    
    for idx in edge_order:
        i, j = edges[idx]
        best_color = -1
        best_score = float('inf')
        
        for c in range(5):
            common = neighbors[c][i] & neighbors[c][j]
            count = common.bit_count()
            if count < best_score:
                best_score = count
                best_color = c
                if best_score == 0:
                    break
        
        colors[idx] = best_color
        neighbors[best_color][i] |= (1 << j)
        neighbors[best_color][j] |= (1 << i)
        
    current_violations = calc_violations()
    if current_violations == 0:
        return {'n': n, 'colors': colors}
        
    # 2. Local Search (Simulated Annealing)
    start_time = time.time()
    temperature = 1.0
    cooling_rate = 0.9995
    min_temp = 0.01
    
    best_violations = current_violations
    best_colors = colors[:]
    
    while time.time() < start_time + time_limit:
        if temperature < min_temp:
            temperature = min_temp
            
        # Pick an edge to change
        if random.random() < 0.7:
            e_idx = get_violating_edge_single()
        else:
            e_idx = random.randint(0, num_edges - 1)
            
        old_color = colors[e_idx]
        
        new_color = random.randint(0, 4)
        if new_color == old_color:
            continue
            
        # Apply change
        colors[e_idx] = new_color
        update_neighbors(e_idx, old_color, new_color)
        
        new_violations = calc_violations()
        delta = new_violations - current_violations
        
        if delta < 0 or random.random() < math.exp(-delta / temperature):
            current_violations = new_violations
            if new_violations < best_violations:
                best_violations = new_violations
                best_colors = colors[:]
                if best_violations == 0:
                    return {'n': n, 'colors': colors}
        else:
            # Revert
            colors[e_idx] = old_color
            update_neighbors(e_idx, new_color, old_color)
            
        temperature *= cooling_rate
        
    return None
