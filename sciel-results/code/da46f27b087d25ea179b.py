import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Try to find a valid coloring for n from 26 down to 1
    for n in range(26, 0, -1):
        if time.time() > deadline - 10.0:
            break
            
        # Allocate time for this n
        time_budget = 10.0
        if n >= 26:
            time_budget = 20.0
            
        result = _find_coloring(n, time_budget)
        if result is not None:
            return result
            
    # Fallback: trivial solution
    return {'n': 1, 'colors': []}

def _find_coloring(n, time_limit):
    num_edges = n * (n - 1) // 2
    if num_edges == 0:
        return {'n': n, 'colors': []}
        
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    # Map edge tuple to index for fast lookup
    idx_map = {}
    for k, (i, j) in enumerate(edges):
        idx_map[(i, j)] = k
        idx_map[(j, i)] = k
        
    # Fast Validation: Returns True if valid, else False
    def is_valid(colors):
        # Build adjacency bitmasks for each color
        adj = [[0] * n for _ in range(5)]
        for k, (i, j) in enumerate(edges):
            c = colors[k]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            
        for c in range(5):
            for i in range(n):
                for j in range(i + 1, n):
                    common = adj[c][i] & adj[c][j]
                    # Check if common has more than 1 bit set
                    if common & (common - 1):
                        return False
        return True

    # Initialize with random colors
    colors = [random.randint(0, 4) for _ in range(num_edges)]
    if is_valid(colors):
        return {'n': n, 'colors': colors}
        
    start = time.time()
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    # Current cost: number of C4s (or just use validity check for simplicity in SA acceptance)
    # Since is_valid is fast, we can use a simpler metric: count of conflicting pairs.
    
    def count_conflicts(colors):
        adj = [[0] * n for _ in range(5)]
        for k, (i, j) in enumerate(edges):
            c = colors[k]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            
        total = 0
        for c in range(5):
            for i in range(n):
                for j in range(i + 1, n):
                    common = adj[c][i] & adj[c][j]
                    if common & (common - 1):
                        # Count pairs of common neighbors
                        cnt = bin(common).count('1')
                        total += cnt * (cnt - 1) // 2
        return total

    current_cost = count_conflicts(colors)
    best_colors = colors[:]
    best_cost = current_cost
    
    while time.time() - start < time_limit:
        # Find a conflict edge
        adj = [[0] * n for _ in range(5)]
        for k, (i, j) in enumerate(edges):
            c = colors[k]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            
        conflict_edge_idx = None
        for c in range(5):
            for i in range(n):
                for j in range(i + 1, n):
                    common = adj[c][i] & adj[c][j]
                    if common & (common - 1):
                        # Found a C4. Pick an edge from it.
                        # Extract two common neighbors u, v
                        lsb = common & -common
                        u = lsb.bit_length() - 1
                        common ^= lsb
                        lsb2 = common & -common
                        v = lsb2.bit_length() - 1
                        
                        # Edges in C4: (i,u), (u,j), (j,v), (v,i)
                        # Pick (i, u)
                        conflict_edge_idx = idx_map.get((i, u))
                        if conflict_edge_idx is None:
                            conflict_edge_idx = idx_map.get((u, i))
                        break
                if conflict_edge_idx is not None:
                    break
            if conflict_edge_idx is not None:
                break
                
        if conflict_edge_idx is None:
            # No conflicts
            return {'n': n, 'colors': colors}
            
        # Recolor
        old_color = colors[conflict_edge_idx]
        new_color = random.choice([c for c in range(5) if c != old_color])
        colors[conflict_edge_idx] = new_color
        
        new_cost = count_conflicts(colors)
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < pow(2.718, -delta / T):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_colors = colors[:]
            if new_cost == 0:
                return {'n': n, 'colors': colors}
        else:
            colors[conflict_edge_idx] = old_color
            
        T *= cooling
        if T < T_min:
            T = T_min
            
    if best_cost == 0:
        return {'n': n, 'colors': best_colors}
        
    return None
