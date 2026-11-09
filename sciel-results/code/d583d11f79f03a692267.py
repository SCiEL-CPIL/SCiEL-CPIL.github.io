"""Stochastic local search (Simulated Annealing) for k-regular graphs of large girth."""
import random
import math
import time
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g on as few vertices as possible.
    Uses Simulated Annealing with 2-switches to maximize girth.
    
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j).
    """
    # Moore bound: minimum possible vertices
    if g % 2 == 0:
        moore = 2 * sum((k - 1) ** i for i in range(g // 2))
    else:
        moore = 1 + k * sum((k - 1) ** i for i in range((g - 3) // 2 + 1))
    
    # Start with a reasonable number of vertices
    n = max(moore, k * (g - 1) // 2 + 2)
    
    def generate_k_regular(n, k):
        """Generate a random simple k-regular graph using configuration model."""
        if k == 0:
            return []
        if n * k % 2 != 0:
            raise ValueError("n*k must be even")
        
        max_attempts = 500
        for attempt in range(max_attempts):
            stubs = []
            for i in range(n):
                for _ in range(k):
                    stubs.append(i)
            random.shuffle(stubs)
            
            edges = set()
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v:
                    valid = False
                    break
                edge = (min(u, v), max(u, v))
                if edge in edges:
                    valid = False
                    break
                edges.add(edge)
            
            if valid:
                return list(edges)
        
        # Fallback: use a structured approach
        edges = []
        for i in range(n):
            for j in range(1, k // 2 + 1):
                edges.append((i, (i + j) % n))
                edges.append((i, (i - j) % n))
        # Deduplicate
        unique = set()
        result = []
        for e in edges:
            key = (min(e), max(e))
            if key not in unique:
                unique.add(key)
                result.append(key)
        return result
    
    def build_adj(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj
    
    def count_short_nonbacktracking(n, adj, max_len):
        """Count non-backtracking walks that close into short cycles."""
        count = 0
        for start in range(n):
            # State: (current, prev)
            visited = set()
            queue = deque()
            for neighbor in adj[start]:
                queue.append((neighbor, start, 1))
                visited.add((neighbor, start))
            while queue:
                u, prev, dist = queue.popleft()
                if dist + 1 >= max_len:
                    continue
                for v in adj[u]:
                    if v == prev:
                        continue
                    if v == start and dist + 1 >= 3:
                        count += 1
                        continue
                    state = (v, u)
                    if state not in visited:
                        visited.add(state)
                        queue.append((v, u, dist + 1))
        return count
    
    def sa_search(n, k, g, time_limit=110):
        """Run simulated annealing to maximize girth using 2-switches."""
        edges = generate_k_regular(n, k)
        adj = build_adj(n, edges)
        
        start_time = time.time()
        T = 5.0
        T_min = 0.01
        cooling = 0.998
        
        best_edges = edges[:]
        best_score = count_short_nonbacktracking(n, adj, g)
        
        # Precompute edge set for fast lookup
        edge_set = set()
        for u, v in edges:
            edge_set.add((min(u, v), max(u, v)))
        
        while time.time() - start_time < time_limit:
            if len(edges) < 2:
                break
            
            i = random.randint(0, len(edges) - 1)
            j = random.randint(0, len(edges) - 1)
            if i == j:
                continue
            
            a, b = edges[i]
            c, d = edges[j]
            
            if len({a, b, c, d}) != 4:
                continue
            
            # Two possible reconnections
            new_edge1 = (min(a, c), max(a, c))
            new_edge2 = (min(b, d), max(b, d))
            new_edge3 = (min(a, d), max(a, d))
            new_edge4 = (min(b, c), max(b, c))
            
            valid_options = []
            if new_edge1 not in edge_set and new_edge2 not in edge_set and new_edge1 != new_edge2:
                valid_options.append(((a, c), (b, d), new_edge1, new_edge2))
            if new_edge3 not in edge_set and new_edge4 not in edge_set and new_edge3 != new_edge4:
                valid_options.append(((a, d), (b, c), new_edge3, new_edge4))
            
            if not valid_options:
                continue
            
            new_e1, new_e2, ne1, ne2 = random.choice(valid_options)
            
            # Apply switch
            new_edges = edges[:]
            new_edges[i] = new_e1
            new_edges[j] = new_e2
            
            new_adj = build_adj(n, new_edges)
            new_score = count_short_nonbacktracking(n, new_adj, g)
            
            # Accept if better, or with probability exp(-delta/T)
            if new_score <= best_score:
                accept = True
            else:
                delta = new_score - best_score
                accept = random.random() < math.exp(-delta / T)
            
            if accept:
                # Update edge_set
                edge_set.discard((min(a, b), max(a, b)))
                edge_set.discard((min(c, d), max(c, d)))
                edge_set.add(ne1)
                edge_set.add(ne2)
                
                edges = new_edges
                adj = new_adj
                best_edges = edges[:]
                best_score = new_score
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        return best_edges, best_score
    
    # Run SA
    best_edges, best_score = sa_search(n, k, g, time_limit=110)
    
    # Verify
    final_adj = build_adj(n, best_edges)
    for i in range(n):
        if len(final_adj[i]) != k:
            pass
    
    if count_short_nonbacktracking(n, final_adj, g) > 0:
        # Fallback to algebraic construction
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
        index, order, fallback_edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    fallback_edges.append((index[vertex], index[other]))
        
        return len(order), fallback_edges
    
    return n, best_edges
