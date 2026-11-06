import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Target n values in descending order
    target_ns = [26, 25, 24, 23, 22, 21, 20]
    
    # Allocate time budgets
    # We prioritize n=26 heavily.
    # Cyclic search is fast, so we give it a chunk.
    # General search is slower but more powerful.
    
    last_valid = {'n': 1, 'colors': []}
    
    for n in target_ns:
        if time.time() > deadline - 2.0:
            break
            
        # Time allocation strategy:
        # For n=26: 40s cyclic, 60s general
        # For n=25: 20s cyclic, 20s general
        # For n<25: 5s each
        
        if n == 26:
            t_cyclic = 40.0
            t_general = 60.0
        elif n == 25:
            t_cyclic = 20.0
            t_general = 20.0
        else:
            t_cyclic = 5.0
            t_general = 5.0
            
        # 1. Try Cyclic Search
        if time.time() < deadline - t_general - 1.0:
            result = search_cyclic(n, time_limit=t_cyclic)
            if result is not None:
                return result
        
        # 2. Try General Local Search
        if time.time() < deadline - 1.0:
            t_remaining = deadline - time.time() - 1.0
            if t_remaining > 0:
                result = local_search_general(n, time_limit=t_remaining)
                if result is not None:
                    return result
                    
    # Fallback: return a small valid solution
    # n=11 is easily constructible or just return n=1
    return {'n': 1, 'colors': []}

def search_cyclic(n, time_limit=10):
    """
    Search for a cyclic coloring where color(i, j) depends only on (j - i) % n.
    For n even, we need to be careful with the n/2 difference.
    """
    # Parameters: color for difference d, where d in 1..n//2
    # If n is odd, d in 1..(n-1)//2
    # If n is even, d in 1..n//2. Note that diff and n-diff are the same edge in undirected graph?
    # In a cyclic graph, edge (i, i+d) has diff d. Edge (i, i-d) has diff n-d.
    # If we define color by diff, we must ensure color(d) == color(n-d) for consistency?
    # Actually, in an undirected graph, the edge {i, j} is the same as {j, i}.
    # If we assign a color to the difference d, and the difference n-d, they correspond to the same set of edges.
    # So we only need parameters for d = 1, ..., n//2.
    
    max_diff = n // 2
    params = [random.randint(0, 4) for _ in range(max_diff)]
    
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    def gen_colors(params):
        colors = [0] * num_edges
        for idx, (i, j) in enumerate(edges):
            diff = (j - i) % n
            if diff > max_diff:
                diff = n - diff
            # diff is now in 1..max_diff
            colors[idx] = params[diff - 1]
        return colors

    def calc_violations(colors):
        # Bitmask representation for speed
        neighbors = [[0] * n for _ in range(5)]
        for idx, (i, j) in enumerate(edges):
            c = colors[idx]
            neighbors[c][i] |= (1 << j)
            neighbors[c][j] |= (1 << i)
            
        violations = 0
        for c in range(5):
            for u in range(n):
                nu = neighbors[c][u]
                for w in range(u + 1, n):
                    common = nu & neighbors[c][w]
                    k = common.bit_count()
                    if k >= 2:
                        violations += k * (k - 1) // 2
        return violations
        
    current_colors = gen_colors(params)
    current_violations = calc_violations(current_colors)
    best_violations = current_violations
    best_params = params[:]
    
    if current_violations == 0:
        return {'n': n, 'colors': current_colors}
        
    start_time = time.time()
    temperature = 1.0
    cooling_rate = 0.99
    
    while time.time() < start_time + time_limit:
        idx = random.randint(0, max_diff - 1)
        old_val = params[idx]
        new_val = random.randint(0, 4)
        if new_val == old_val:
            continue
            
        params[idx] = new_val
        new_colors = gen_colors(params)
        new_violations = calc_violations(new_colors)
        
        delta = new_violations - current_violations
        
        # Simulated Annealing acceptance
        if delta < 0 or random.random() < math.exp(-delta / temperature):
            current_violations = new_violations
            current_colors = new_colors
            if new_violations < best_violations:
                best_violations = new_violations
                best_params = params[:]
                if best_violations == 0:
                    return {'n': n, 'colors': current_colors}
        else:
            params[idx] = old_val
            
        temperature *= cooling_rate
        if temperature < 0.01:
            temperature = 0.01
            
    # If we didn't find 0, we might still return the best if it's 0 (handled above)
    # Otherwise, return None to let general search try
    if best_violations == 0:
         return {'n': n, 'colors': gen_colors(best_params)}
    return None

def local_search_general(n, time_limit=10):
    """
    General local search with incremental violation updates.
    """
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Precompute edge index for O(1) lookup
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx
        
    # Initialize colors randomly
    colors = [random.randint(0, 4) for _ in range(num_edges)]
    
    # Neighbor bitmasks
    neighbors = [[0] * n for _ in range(5)]
    
    def build_neighbors():
        for c in range(5):
            for v in range(n):
                neighbors[c][v] = 0
        for idx, (i, j) in enumerate(edges):
            c = colors[idx]
            neighbors[c][i] |= (1 << j)
            neighbors[c][j] |= (1 << i)
            
    build_neighbors()
    
    def get_violations():
        violations = 0
        for c in range(5):
            for u in range(n):
                nu = neighbors[c][u]
                for w in range(u + 1, n):
                    common = nu & neighbors[c][w]
                    k = common.bit_count()
                    if k >= 2:
                        violations += k * (k - 1) // 2
        return violations
        
    current_violations = get_violations()
    best_violations = current_violations
    best_colors = colors[:]
    
    if current_violations == 0:
        return {'n': n, 'colors': colors}
        
    start_time = time.time()
    temperature = 1.0
    cooling_rate = 0.999
    
    while time.time() < start_time + time_limit:
        e_idx = random.randint(0, num_edges - 1)
        i, j = edges[e_idx]
        old_color = colors[e_idx]
        
        # Pick a new color
        new_color = random.randint(0, 4)
        if new_color == old_color:
            continue
            
        # Calculate delta in violations efficiently
        # Violations are C4s. Changing edge (i,j) affects C4s that use this edge.
        # A C4 is u-v-w-x-u. The edge (i,j) can be part of a C4 in several ways.
        # Instead of complex delta calc, we just recalculate if we accept, 
        # but to decide acceptance we need delta.
        # For speed, we can just recalculate full violations if we accept a move? 
        # No, we need delta to decide.
        # Let's use a simpler heuristic: accept if it looks better, or use full recalc for decision.
        # Given n <= 26, full recalc is O(5 * n^2 * bit_count) ~ 5 * 676 * 1 = 3380 ops. Very fast.
        
        # Tentatively apply
        neighbors[old_color][i] &= ~(1 << j)
        neighbors[old_color][j] &= ~(1 << i)
        neighbors[new_color][i] |= (1 << j)
        neighbors[new_color][j] |= (1 << i)
        
        new_violations = get_violations()
        delta = new_violations - current_violations
        
        if delta < 0 or random.random() < math.exp(-delta / temperature):
            current_violations = new_violations
            colors[e_idx] = new_color
            if new_violations < best_violations:
                best_violations = new_violations
                best_colors = colors[:]
                if best_violations == 0:
                    return {'n': n, 'colors': colors}
        else:
            # Revert
            neighbors[new_color][i] &= ~(1 << j)
            neighbors[new_color][j] &= ~(1 << i)
            neighbors[old_color][i] |= (1 << j)
            neighbors[old_color][j] |= (1 << i)
            
        temperature *= cooling_rate
        if temperature < 0.01:
            temperature = 0.01
            
    if best_violations == 0:
        return {'n': n, 'colors': best_colors}
    return None

# Helper to ensure we have a valid fallback if everything fails
def get_fallback_solution():
    # n=1 is always valid
    return {'n': 1, 'colors': []}
