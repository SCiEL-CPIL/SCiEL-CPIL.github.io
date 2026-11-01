import time
import random
import sys
import math
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Fallback: LUW Construction
    def luw_construct():
        q = k
        d_candidates = []
        for d in range(3, 20):
            if d % 2 == 1:
                if d + 5 >= g:
                    d_candidates.append(d)
            else:
                if d + 4 >= g:
                    d_candidates.append(d)
        if not d_candidates:
            d = max(3, g - 5)
            if d % 2 == 0:
                d += 1
        else:
            d = min(d_candidates)
            
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
            
        rule = {}
        rule[2] = (1, 1)
        rule[3] = (2, 1)
        rule[4] = (1, 2)
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
            rules_cycle = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
            rule[c] = rules_cycle[r]
            
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield (1 - side), tuple(y)
                
        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
                    
        return len(order), edges

    def is_valid_graph(n, edges, k, g):
        # Check degree
        degree = [0] * n
        for u, v in edges:
            if u == v: return False
            degree[u] += 1
            degree[v] += 1
        if any(d != k for d in degree):
            return False
            
        # Check simple
        edge_set = set()
        for u, v in edges:
            key = (min(u, v), max(u, v))
            if key in edge_set: return False
            edge_set.add(key)
            
        # Check girth
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        max_depth = (g - 1) // 2
        for start_v in range(n):
            visited = [[-1] * n for _ in range(max_depth + 1)]
            visited[0][start_v] = -2
            queue = deque([(start_v, 0)])
            while queue:
                v, depth = queue.popleft()
                if depth == max_depth:
                    continue
                for w in adj[v]:
                    if visited[depth + 1][w] == -1:
                        visited[depth + 1][w] = v
                        queue.append((w, depth + 1))
                    elif w != v and visited[depth + 1][w] != -2:
                        if depth + 1 < g:
                            return False
        return True

    def generate_random_4_reg(n):
        # Walecki or similar, but for 4-reg
        # Simple method: random permutation matching
        # 1. Create list of vertices
        # 2. Randomly pair them up for first 2 degrees? No.
        # Use configuration model:
        # 2n stubs. Randomly pair stubs.
        # If self-loop or multi-edge, retry.
        
        while True:
            stubs = []
            for i in range(n):
                stubs.extend([i, i, i, i])
            random.shuffle(stubs)
            
            edges = []
            used = [False] * n # Not quite, need to track pairs
            # Better:
            # List of (u, v)
            # Check for duplicates and loops
            
            # Efficient check:
            # Use a set of edges
            edge_set = set()
            valid = True
            for i in range(0, 2*n, 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                key = (min(u, v), max(u, v))
                if key in edge_set:
                    valid = False
                    break
                edge_set.add(key)
                edges.append((u, v))
            
            if valid:
                return edges

    def local_search(n, edges, k, g, deadline):
        # Convert to adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Edge set for quick lookup
        edge_set = set()
        for u, v in edges:
            edge_set.add((min(u, v), max(u, v)))
            
        def find_short_cycle():
            # BFS from random vertex
            # Return cycle as list of vertices
            for _ in range(10): # Try 10 random starts
                start = random.randint(0, n-1)
                parent = [-1] * n
                depth = [-1] * n
                parent[start] = -2
                depth[start] = 0
                queue = deque([start])
                
                while queue:
                    v = queue.popleft()
                    if depth[v] >= (g-1)//2:
                        continue
                    for w in adj[v]:
                        if depth[w] == -1:
                            depth[w] = depth[v] + 1
                            parent[w] = v
                            queue.append(w)
                        elif w != parent[v]:
                            # Found a cycle
                            # Reconstruct
                            if depth[v] + 1 < g:
                                # Cycle length is depth[v] + depth[w] + 1
                                # Wait, w is already visited.
                                # If w is in current path, it's a cycle.
                                # This simple BFS might find a cycle that is not the shortest.
                                # But any cycle < g is bad.
                                if depth[w] <= depth[v]:
                                    # Reconstruct path from w to v?
                                    # Actually, just return a small cycle if found
                                    pass
                # If BFS didn't find a cycle < g, assume good (heuristic)
            return None # This is a placeholder, need better cycle detection

        # Better: Just check girth periodically.
        # If girth < g, try to fix.
        
        # This implementation is getting complex.
        # Let's just try random restarts with a quick girth check.
        
        best_n = float('inf')
        best_edges = None
        
        # Try n values
        for n_try in [320, 330, 350, 400]:
            if time.time() > deadline:
                break
            for attempt in range(100):
                if time.time() > deadline:
                    break
                edges_try = generate_random_4_reg(n_try)
                # Quick girth check
                if is_valid_graph(n_try, edges_try, k, g):
                    if n_try < best_n:
                        best_n = n_try
                        best_edges = edges_try
                    return best_n, best_edges
                    
        if best_edges:
            return best_n, best_edges
        return None

    # Strategy:
    # 1. Try to find a small graph via random search (fast, low success rate for optimal, but might find suboptimal).
    # 2. If fails, return LUW.
    
    # The random search above is likely too slow or ineffective for n=320.
    # Let's just return LUW but ensure it's the smallest LUW possible.
    
    return luw_construct()
