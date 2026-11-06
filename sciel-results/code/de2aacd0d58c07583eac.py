import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Try from largest n down
    for n in range(20, 8, -1):
        if time.time() > deadline - 5:
            break
        remaining = deadline - time.time()
        if remaining < 3:
            break
        
        budget = min(remaining - 2, 25.0)
        if budget < 1:
            break
        
        result = _solve(n, budget)
        if result is not None:
            return result
    
    # Fallback: n=5 is always valid
    n = 5
    return {'n': n, 'colors': [0] * (n * (n - 1) // 2)}


def _solve(n, budget):
    """Solve via local search: start with random coloring, repair C6 violations."""
    deadline = time.time() + budget
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Precompute edge index
    edge_idx = {}
    for k, (u, v) in enumerate(edges):
        edge_idx[(u, v)] = k
        edge_idx[(v, u)] = k
    
    # Fast C6 check using bitmasks
    # adj[c][v] = bitmask of neighbors of v in color c
    
    def find_c6(c, adj):
        """Find one C6 in color c, return list of 6 vertices or None."""
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[c][u] & (1 << v)):
                    continue
                # Check 4-paths u-x1-x2-x3-x4-v
                nu = adj[c][u] & ~(1 << v)
                nv = adj[c][v] & ~(1 << u)
                m1 = nu
                while m1:
                    b1 = m1 & (-m1)
                    x1 = b1.bit_length() - 1
                    m1 ^= b1
                    m4 = nv & ~(1 << x1)
                    while m4:
                        b4 = m4 & (-m4)
                        x4 = b4.bit_length() - 1
                        m4 ^= b4
                        if x4 == u:
                            continue
                        forb = (1 << u) | (1 << v) | (1 << x1) | (1 << x4)
                        nx1 = adj[c][x1] & ~forb
                        nx4 = adj[c][x4] & ~forb
                        if nx1 == 0 or nx4 == 0:
                            continue
                        m2 = nx1
                        while m2:
                            b2 = m2 & (-m2)
                            x2 = b2.bit_length() - 1
                            m2 ^= b2
                            if adj[c][x2] & nx4:
                                # Found C6: u-x1-x2-x3-x4-v
                                # Need to find x3
                                nx2 = adj[c][x2] & nx4
                                b3 = nx2 & (-nx2)
                                x3 = b3.bit_length() - 1
                                return [u, x1, x2, x3, x4, v]
        return None
    
    def count_c6s(adj):
        """Count total C6s across all colors."""
        total = 0
        for c in range(4):
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[c][u] & (1 << v):
                        nu = adj[c][u] & ~(1 << v)
                        nv = adj[c][v] & ~(1 << u)
                        m1 = nu
                        while m1:
                            b1 = m1 & (-m1)
                            x1 = b1.bit_length() - 1
                            m1 ^= b1
                            m4 = nv & ~(1 << x1)
                            while m4:
                                b4 = m4 & (-m4)
                                x4 = b4.bit_length() - 1
                                m4 ^= b4
                                if x4 == u:
                                    continue
                                forb = (1 << u) | (1 << v) | (1 << x1) | (1 << x4)
                                nx1 = adj[c][x1] & ~forb
                                nx4 = adj[c][x4] & ~forb
                                if nx1 == 0 or nx4 == 0:
                                    continue
                                m2 = nx1
                                while m2:
                                    b2 = m2 & (-m2)
                                    x2 = b2.bit_length() - 1
                                    m2 ^= b2
                                    total += (adj[c][x2] & nx4).bit_count()
        return total // 6
    
    # Greedy initialization
    def greedy_init(rng):
        colors = [-1] * num_edges
        adj = [[0] * n for _ in range(4)]
        for k in range(num_edges):
            u, v = edges[k]
            # Try colors in random order
            c_order = [0, 1, 2, 3]
            rng.shuffle(c_order)
            placed = False
            for c in c_order:
                # Check if adding (u,v) in color c creates a C6
                # Need: no 4-path from u to v in G_c
                nu = adj[c][u]
                nv = adj[c][v]
                ok = True
                m1 = nu
                while m1 and ok:
                    b1 = m1 & (-m1)
                    x1 = b1.bit_length() - 1
                    m1 ^= b1
                    m4 = nv & ~(1 << x1)
                    while m4 and ok:
                        b4 = m4 & (-m4)
                        x4 = b4.bit_length() - 1
                        m4 ^= b4
                        if x4 == u:
                            continue
                        forb = (1 << u) | (1 << v) | (1 << x1) | (1 << x4)
                        nx1 = adj[c][x1] & ~forb
                        nx4 = adj[c][x4] & ~forb
                        if nx1 == 0 or nx4 == 0:
                            continue
                        m2 = nx1
                        while m2:
                            b2 = m2 & (-m2)
                            x2 = b2.bit_length() - 1
                            m2 ^= b2
                            if adj[c][x2] & nx4:
                                ok = False
                                break
                if ok:
                    colors[k] = c
                    adj[c][u] |= (1 << v)
                    adj[c][v] |= (1 << u)
                    placed = True
                    break
            if not placed:
                # Force random
                c = rng.randint(0, 3)
                colors[k] = c
                adj[c][u] |= (1 << v)
                adj[c][v] |= (1 << u)
        return colors, adj
    
    # Local search: repair C6 violations
    def local_search(colors, adj, deadline):
        rng = random.Random(int(time.time() * 1000) % 100000)
        
        # Count initial C6s
        total = count_c6s(adj)
        if total == 0:
            return colors[:]
        
        best_total = total
        best_colors = colors[:]
        stuck = 0
        
        while time.time() < deadline:
            # Find a C6
            c6 = None
            for c in range(4):
                c6 = find_c6(c, adj)
                if c6:
                    break
            
            if c6 is None:
                return colors[:]
            
            # The C6 is [v0, v1, v2, v3, v4, v5] with edges
            # (v0,v1), (v1,v2), (v2,v3), (v3,v4), (v4,v5), (v5,v0) in color c
            # Pick a random edge of this C6 and recolor it
            edge_positions = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,0)]
            pos = rng.randint(0, 5)
            a, b = c6[edge_positions[pos][0]], c6[edge_positions[pos][1]]
            
            k = edge_idx[(a, b)]
            old_c = colors[k]
            
            # Try all other colors
            new_c = rng.randint(0, 3)
            if new_c == old_c:
                new_c = (new_c + 1) % 4
            
            # Remove from old color
            adj[old_c][a] &= ~(1 << b)
            adj[old_c][b] &= ~(1 << a)
            
            # Add to new color
            adj[new_c][a] |= (1 << b)
            adj[new_c][b] |= (1 << a)
            
            colors[k] = new_c
            
            new_total = count_c6s(adj)
            
            if new_total < total:
                total = new_total
                stuck = 0
                if total < best_total:
                    best_total = total
                    best_colors = colors[:]
                    if total == 0:
                        return colors[:]
            else:
                # Revert
                adj[new_c][a] &= ~(1 << b)
                adj[new_c][b] &= ~(1 << a)
                adj[old_c][a] |= (1 << b)
                adj[old_c][b] |= (1 << a)
                colors[k] = old_c
                stuck += 1
            
            if stuck > 200:
                # Restart from best
                colors = best_colors[:]
                adj = [[0] * n for _ in range(4)]
                for k2 in range(num_edges):
                    u2, v2 = edges[k2]
                    c2 = colors[k2]
                    adj[c2][u2] |= (1 << v2)
                    adj[c2][v2] |= (1 << u2)
                total = count_c6s(adj)
                stuck = 0
        
        if best_total == 0:
            return best_colors[:]
        return None
    
    # Multiple restarts
    max_restarts = 30
    for r in range(max_restarts):
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        if remaining < 0.5:
            break
        
        rng = random.Random(r * 7919 + int(time.time()) % 1000)
        colors, adj = greedy_init(rng)
        
        # Quick check
        if count_c6s(adj) == 0:
            return colors[:]
        
        # Local search with time budget
        ls_budget = min(remaining - 0.3, 4.0)
        result = local_search(colors, adj, time.time() + ls_budget)
        if result is not None:
            return result
    
    return None
