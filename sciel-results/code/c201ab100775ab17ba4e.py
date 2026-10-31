"""Construct a 3-regular graph of girth 16 via optimized random search with early girth check."""
# EVOLVE-BLOCK-START
import random
from collections import deque
import sys

def construct(k, g):
    # For k=3, g=16
    # Strategy: Random 3-regular graphs with fast girth verification.
    # The previous program was slow due to full BFS.
    # This version uses BFS with early termination for cycles < g.
    
    if k != 3:
        # Fallback for other k (keep LUW for safety, though task is k=3)
        return _luw_construct(k, g)
        
    # Moore bound for k=3, g=16 is 510, so n >= 512 (even).
    # Known best is 936.
    # We search from 512 upwards.
    
    def generate_random_3reg(n):
        """Generate a random 3-regular graph on n vertices using Configuration Model."""
        if n < 4 or n % 2 == 1:
            return None
        
        # Configuration model
        # To avoid self-loops and multi-edges, we use a rejection sampling or
        # a specific pairing algorithm.
        # Simple method: shuffle stubs, pair them.
        # If self-loop or multi-edge, retry or fix.
        # For speed, we just generate and check.
        
        for _ in range(100): # Max retries for a single graph
            stubs = []
            for i in range(n):
                for _ in range(3):
                    stubs.append(i)
            random.shuffle(stubs)
            
            edges = set()
            valid = True
            for i in range(0, n * 3, 2):
                u = stubs[i]
                v = stubs[i+1]
                if u == v:
                    valid = False
                    break
                key = (min(u, v), max(u, v))
                if key in edges:
                    valid = False
                    break
                edges.add(key)
            
            if valid:
                # Verify degrees (should be 3 by construction if no breaks)
                deg = [0] * n
                for u, v in edges:
                    deg[u] += 1
                    deg[v] += 1
                if all(d == 3 for d in deg):
                    return list(edges)
        return None

    def has_short_cycle(n, edges, max_len):
        """
        Check if the graph has any cycle of length <= max_len.
        Returns True if such a cycle exists.
        Uses BFS from each vertex, stopping early if a short cycle is found.
        """
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # For each start node, do BFS up to depth max_len//2
        # If we find a back edge to a node at depth d < current_depth,
        # cycle length is current_depth + d + 1.
        # We need to ensure we don't miss odd cycles?
        # BFS finds shortest path. 
        # A cycle of length L is found if there is an edge (u,v) such that 
        # dist[u] + dist[v] + 1 = L.
        # In BFS from s, if we encounter an edge (u,v) where both u and v are visited,
        # and v is not parent of u, then cycle length is dist[u] + dist[v] + 1.
        # We must check all such edges.
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                du = dist[u]
                
                # Optimization: if du > max_len // 2, we might still find a cycle
                # connecting to a shallow node?
                # Actually, if dist[u] + dist[v] + 1 <= max_len, and dist[v] >= 0,
                # then dist[u] <= max_len - 1.
                # So we can stop BFS at depth max_len - 1.
                
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        if dist[v] <= max_len: # Only enqueue if within range
                            queue.append(v)
                    else:
                        # v is already visited
                        # Check if this edge forms a short cycle
                        # We must avoid the edge (u, parent[u])
                        if v != parent[u] and u != parent[v]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len <= max_len:
                                return True
        return False

    # Search parameters
    min_n = 512
    max_n = 1000
    attempts_per_n = 20 # Increased attempts
    
    # Try sizes in increasing order
    for n in range(min_n, max_n + 1, 2):
        if n % 2 != 0:
            continue
            
        # Check for short cycles up to g-1
        # We can check incrementally: 3, 4, ... g-1
        # Or just check up to g-1 directly.
        # has_short_cycle with max_len = g-1 covers all.
        
        for _ in range(attempts_per_n):
            edges = generate_random_3reg(n)
            if edges is None:
                continue
            
            # Fast check: does it have a cycle < g?
            if not has_short_cycle(n, edges, g - 1):
                return n, edges
                
    # Fallback if not found
    return _luw_construct(k, g)

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction (kept as fallback)."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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
# EVOLVE-BLOCK-END
