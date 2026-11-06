"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    hard_deadline = start_time + 115.0
    
    def verify(n, colors):
        """Verify that no color class contains a C6."""
        adj = [[0] * n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        
        for c in range(4):
            adj_list = [[] for _ in range(n)]
            for i in range(n):
                mask = adj[c][i]
                while mask:
                    lsb = mask & -mask
                    j = lsb.bit_length() - 1
                    adj_list[i].append(j)
                    mask -= lsb
            
            for u in range(n):
                for v in adj_list[u]:
                    if v <= u: continue
                    for w in adj_list[v]:
                        if w == u: continue
                        for x in adj_list[w]:
                            if x == u or x == v: continue
                            for y in adj_list[x]:
                                if y == u or y == v or y == w: continue
                                for z in adj_list[y]:
                                    if z == u or z == v or z == w or z == x: continue
                                    if (adj[c][z] >> u) & 1:
                                        return False
        return True

    def find_c6(c, adj, n):
        """Find a C6 in color class c, return the 6 vertices or None."""
        adj_list = [[] for _ in range(n)]
        for i in range(n):
            mask = adj[c][i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                adj_list[i].append(j)
                mask -= lsb
        
        for u in range(n):
            for v in adj_list[u]:
                if v <= u: continue
                for w in adj_list[v]:
                    if w == u: continue
                    for x in adj_list[w]:
                        if x == u or x == v: continue
                        for y in adj_list[x]:
                            if y == u or y == v or y == w: continue
                            for z in adj_list[y]:
                                if z == u or z == v or z == w or z == x: continue
                                if (adj[c][z] >> u) & 1:
                                    return [u, v, w, x, y, z]
        return None

    def random_greedy(n, deadline):
        """Randomized greedy construction with C6 avoidance."""
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        colors = [-1] * len(edges)
        adj = [[0] * n for _ in range(4)]
        
        # Process edges in random order
        order = list(range(len(edges)))
        random.shuffle(order)
        
        for idx in order:
            if time.time() > deadline:
                return None
            u, v = edges[idx]
            valid_colors = []
            for c in range(4):
                # Check if adding (u,v) in color c creates a C6
                adj[c][u] |= (1 << v)
                adj[c][v] |= (1 << u)
                c6 = find_c6(c, adj, n)
                adj[c][u] &= ~(1 << v)
                adj[c][v] &= ~(1 << u)
                if c6 is None:
                    valid_colors.append(c)
            
            if not valid_colors:
                return None
            
            # Choose randomly among valid colors
            c = random.choice(valid_colors)
            colors[idx] = c
            adj[c][u] |= (1 << v)
            adj[c][v] |= (1 << u)
        
        return colors

    def local_search(n, colors, deadline, max_iter=1000):
        """Local search to fix C6 violations by recoloring edges."""
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        adj = [[0] * n for _ in range(4)]
        for idx, (u, v) in enumerate(edges):
            c = colors[idx]
            adj[c][u] |= (1 << v)
            adj[c][v] |= (1 << u)
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            # Find a C6
            c6_found = None
            for c in range(4):
                c6 = find_c6(c, adj, n)
                if c6:
                    c6_found = (c, c6)
                    break
            
            if c6_found is None:
                return colors
            
            c, cycle = c6_found
            # Get the 6 edges of the cycle
            cycle_edges = []
            for k in range(6):
                u, v = cycle[k], cycle[(k + 1) % 6]
                if u > v:
                    u, v = v, u
                # Find index of this edge
                idx = u * (n - 1) - u * (u + 1) // 2 + (v - u - 1)
                cycle_edges.append(idx)
            
            # Try recoloring one of the cycle edges
            best_move = None
            best_violations = len(cycle_edges)
            for idx in cycle_edges:
                u, v = edges[idx]
                for new_c in range(4):
                    if new_c == c:
                        continue
                    # Remove edge from old color
                    adj[c][u] &= ~(1 << v)
                    adj[c][v] &= ~(1 << u)
                    # Add to new color
                    adj[new_c][u] |= (1 << v)
                    adj[new_c][v] |= (1 << u)
                    
                    # Count remaining C6s
                    violations = 0
                    for cc in range(4):
                        if find_c6(cc, adj, n):
                            violations += 1
                    
                    # Revert
                    adj[new_c][u] &= ~(1 << v)
                    adj[new_c][v] &= ~(1 << u)
                    adj[c][u] |= (1 << v)
                    adj[c][v] |= (1 << u)
                    
                    if violations < best_violations:
                        best_violations = violations
                        best_move = (idx, new_c)
            
            if best_move is None:
                # Try random recoloring
                idx = random.choice(cycle_edges)
                new_c = random.randint(0, 3)
                best_move = (idx, new_c)
            
            # Apply the move
            idx, new_c = best_move
            u, v = edges[idx]
            old_c = colors[idx]
            adj[old_c][u] &= ~(1 << v)
            adj[old_c][v] &= ~(1 << u)
            adj[new_c][u] |= (1 << v)
            adj[new_c][v] |= (1 << u)
            colors[idx] = new_c
        
        return colors

    def try_n(n, time_budget):
        """Try to find a valid coloring for n vertices."""
        deadline = time.time() + time_budget
        best_result = None
        
        # Multiple random restarts
        while time.time() < deadline - 2:
            # Random greedy construction
            colors = random_greedy(n, deadline - 5)
            if colors is not None:
                if verify(n, colors):
                    return colors
            
            # Try local search from random initial coloring
            colors = [random.randint(0, 3) for _ in range(n * (n - 1) // 2)]
            colors = local_search(n, colors, deadline - 5, max_iter=500)
            if colors is not None:
                if verify(n, colors):
                    return colors
        
        return best_result

    # Try descending from n=18
    for n in [18, 17, 16, 15, 14, 13, 12, 11, 10]:
        remaining = hard_deadline - time.time()
        if remaining < 10:
            break
        budget = min(remaining * 0.8, 40) if n >= 16 else min(remaining * 0.7, 20)
        result = try_n(n, budget)
        if result is not None:
            if verify(n, result):
                return {'n': n, 'colors': result}
    
    # Fallback: seed solution n=9
    n = 9
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    if verify(n, colors):
        return {'n': n, 'colors': colors}
    
    return {'n': 1, 'colors': []}
# EVOLVE-BLOCK-END
