"""Cage construction: 4-regular graph of girth >= 10 on as few vertices as possible."""

import time
import random
import sys

def construct(k, g):
    # For the (4,10) cage problem, k=4 and g=10.
    # The known (4,10)-cage has 320 vertices.
    # We try to find a 4-regular graph with girth >= 10 on fewer vertices.
    # Strategy:
    #   1. Try several candidate sizes from the Moore bound up to 320.
    #   2. For each size, generate random 4-regular graphs and check girth.
    #   3. If none found in time, fall back to the Lazebnik-Ustimenko-Woldar (LUW) construction
    #      which gives a valid (but large) 4-regular graph with girth >= 10.
    
    deadline = time.time() + 110  # Leave 10s safety margin before 120s limit
    
    # Moore bound for (4,10): 1 + 4*sum_{i=0}^{3} 3^i = 1 + 4*(1+3+9+27) = 1 + 4*40 = 161
    # So n >= 161. Since the cage is bipartite, n is even.
    # Try sizes: 162, 164, ..., up to 320. But that's a lot. Let's try a subset.
    
    # Actually, for girth 10 and degree 4, the graph must be bipartite (no odd cycles).
    # So n must be even.
    # Moore bound for bipartite (4,5): 2 * sum_{i=0}^{4} 3^i = 2*(1+3+9+27+81) = 2*121 = 242
    # So n >= 242 for bipartite (4,10). Since n is even, n >= 242.
    
    # Let's try sizes from 242 to 320, even only.
    # But checking girth for each random graph is expensive.
    # Better approach: start with n=320 (known cage size) and try to verify a known construction,
    # or try smaller sizes with limited attempts.
    
    # For now, let's implement the LUW construction as fallback, and try a few random searches.
    
    def luw_construct():
        """Lazebnik-Ustimenko-Woldar graph CD(4, 4) for (4,10) cage."""
        q = 4
        d = 5  # girth = d + 5 = 10
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)
        
        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges
    
    def check_girth(n, edges, min_girth):
        """Check if graph has girth >= min_girth using BFS from each vertex (non-backtracking)."""
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # For each vertex, do BFS to find shortest cycle
        # Actually, girth = shortest cycle. Use BFS from each vertex, but avoid backtracking.
        # More efficient: for each edge, check if removing it disconnects, or use BFS from each vertex
        # tracking parent to avoid immediate backtracking.
        
        # Simpler: BFS from each vertex, track visited with parent to detect cycles.
        for start in range(n):
            visited = [-1] * n  # -1 = unvisited, else = distance
            parent = [-1] * n
            visited[start] = 0
            queue = [start]
            found = False
            qi = 0
            while qi < len(queue) and not found:
                u = queue[qi]
                qi += 1
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if visited[v] == -1:
                        visited[v] = visited[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        # Cycle found
                        cycle_len = visited[u] + visited[v] + 1
                        if cycle_len < min_girth:
                            return False
                        found = True  # No need to continue this BFS
                        break
        return True
    
    def generate_random_4reg_bipartite(n):
        """Generate a random 4-regular bipartite graph on n vertices (n even).
        Left: 0..n/2-1, Right: n/2..n-1.
        Returns list of edges."""
        n_half = n // 2
        # Each left vertex connects to 4 distinct right vertices, and vice versa.
        # Use configuration model: create multiset of stubs, then random perfect matching.
        left_stubs = []
        for i in range(n_half):
            left_stubs.extend([i] * 4)
        right_stubs = []
        for i in range(n_half):
            right_stubs.extend([(n_half + i)] * 4)
        
        random.shuffle(right_stubs)
        edges = []
        for l, r in zip(left_stubs, right_stubs):
            edges.append((l, r))
        
        # Remove duplicates (keep only one of each parallel edge)
        # But we need exactly 4-regular, so we need to handle this.
        # Simple approach: retry if duplicates exist.
        edge_set = set()
        for u, v in edges:
            key = (min(u, v), max(u, v))
            if key in edge_set:
                return None  # Retry
            edge_set.add(key)
        
        # Verify degree
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if any(d != 4 for d in deg):
            return None
        
        return edges
    
    # Try random search for smaller cages
    random.seed(42)
    
    # Try sizes from 242 to 320, even only, with limited attempts per size
    best_n = None
    best_edges = None
    
    for n in range(242, 321, 2):
        if time.time() > deadline:
            break
        # Try up to 50 random graphs per size
        for attempt in range(50):
            if time.time() > deadline:
                break
            edges = generate_random_4reg_bipartite(n)
            if edges is None:
                continue
            if check_girth(n, edges, g):
                best_n = n
                best_edges = edges
                break  # Found valid graph for this size, try smaller? No, we go ascending.
                # Actually, we want the smallest n, so if we found one, we can break and return.
                # But we're going ascending, so first found is smallest.
                break
        if best_n is not None:
            break
    
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback to LUW construction
    return luw_construct()
