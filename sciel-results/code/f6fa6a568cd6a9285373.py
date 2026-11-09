"""Program for R_5(C4) maximization via optimized local search with bitsets."""
import random
import time

def construct():
    """
    Finds a valid 5-coloring of K_n without monochromatic C4.
    Aims for n=26, falls back to smaller n if not found in time.
    """
    
    def count_violations_fast(adj, n):
        """
        adj: list of 5 lists of integers (bitmasks).
        adj[c][i] is bitmask of neighbors of i in color c.
        Returns total number of 'excess' common neighbors (violations).
        """
        total = 0
        for c in range(5):
            adj_c = adj[c]
            for i in range(n):
                mask_i = adj_c[i]
                if mask_i == 0:
                    continue
                for j in range(i + 1, n):
                    common = (mask_i & adj_c[j]).bit_count()
                    if common > 1:
                        total += common - 1
        return total

    def greedy_init(n):
        """
        Initializes coloring using a greedy approach to minimize initial violations.
        """
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
        
        num_edges = len(edges)
        colors = [0] * num_edges
        adj = [[0] * n for _ in range(5)]
        
        # Process edges in random order to avoid bias
        order = list(range(num_edges))
        random.shuffle(order)
        
        for idx in order:
            i, j = edges[idx]
            best_color = 0
            min_violations = float('inf')
            
            # Try each color
            for c in range(5):
                # Temporarily add edge
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                
                v = count_violations_fast(adj, n)
                
                # Remove edge
                adj[c][i] &= ~(1 << j)
                adj[c][j] &= ~(1 << i)
                
                if v < min_violations:
                    min_violations = v
                    best_color = c
            
            colors[idx] = best_color
            adj[best_color][i] |= (1 << j)
            adj[best_color][j] |= (1 << i)
            
        return colors, adj

    def local_search(n, time_limit=50):
        """
        Runs local search for n vertices.
        Returns (n, colors) if valid, else None.
        """
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
        
        num_edges = len(edges)
        
        # Use greedy init
        colors, adj = greedy_init(n)
        current_violations = count_violations_fast(adj, n)
        
        if current_violations == 0:
            return n, colors
        
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            # Pick random edge
            idx = random.randint(0, num_edges - 1)
            i, j = edges[idx]
            old_color = colors[idx]
            
            # Try a new color
            new_color = random.randint(0, 4)
            if new_color == old_color:
                continue
            
            # Update adj
            adj[old_color][i] &= ~(1 << j)
            adj[old_color][j] &= ~(1 << i)
            
            adj[new_color][i] |= (1 << j)
            adj[new_color][j] |= (1 << i)
            
            # Evaluate
            new_violations = count_violations_fast(adj, n)
            
            if new_violations <= current_violations:
                current_violations = new_violations
                colors[idx] = new_color
                
                if current_violations == 0:
                    return n, colors
            else:
                # Revert
                adj[old_color][i] |= (1 << j)
                adj[old_color][j] |= (1 << i)
                
                adj[new_color][i] &= ~(1 << j)
                adj[new_color][j] &= ~(1 << i)
        
        return None

    # Try n=26 first
    time_budgets = {
        26: 40,
        25: 30,
        24: 20,
        23: 15,
        22: 10,
        21: 5,
        20: 5,
        19: 5,
        18: 5,
        17: 5,
        16: 5,
        15: 5,
        14: 5,
        13: 5,
        12: 5,
        11: 5
    }
    
    for n in sorted(time_budgets.keys(), reverse=True):
        t_limit = time_budgets[n]
        result = local_search(n, t_limit)
        if result:
            n_val, colors_val = result
            return {'n': n_val, 'colors': colors_val}
            
    # Fallback: Return a small valid graph
    # n=11 cyclic construction
    n = 11
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            d = min(d, n - d)
            colors.append(d - 1)
    return {'n': n, 'colors': colors}
