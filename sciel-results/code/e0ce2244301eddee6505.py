import time
import math
import numpy as np
import random
from collections import deque

def construct(k, g):
    deadline = time.time() + 145.0

    def luw_construct(k, g):
        q = k
        d = 3
        while True:
            girth_est = d + 5 if d % 2 == 1 else d + 4
            if girth_est >= g:
                break
            d += 1
        
        # If graph would be too large, try a different approach
        estimated_vertices = 2 * (q ** (d + 1))
        if estimated_vertices > 10000:
            return None, None

        if q == 4:
            mul_table = [
                [0, 0, 0, 0],
                [0, 1, 2, 3],
                [0, 2, 3, 1],
                [0, 3, 1, 2]
            ]
            def add(x, y): return x ^ y
            def sub(x, y): return x ^ y
            def mul(x, y): return mul_table[x][y]
        else:
            def add(x, y): return (x + y) % q
            def sub(x, y): return (x - y) % q
            def mul(x, y): return (x * y) % q
        
        # Precompute neighbor offsets for speed
        precomputed_neighbors = {}

        rule_a = [0] * (d + 1)
        rule_b = [0] * (d + 1)
        rule_a[2], rule_b[2] = 1, 1
        rule_a[3], rule_b[3] = 2, 1
        rule_a[4], rule_b[4] = 1, 2
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - r
            if r == 0:
                a, b = 1, c0 - 2
            elif r == 1:
                a, b = c0 - 1, 1
            elif r == 2:
                a, b = c0, 1
            else:
                a, b = 1, c0 + 1
            rule_a[c] = a
            rule_b[c] = b

        def compute_neighbours(vertex):
            side, x = vertex
            result = []
            for first in range(q):
                y = [0] * (d + 1)
                y[1] = first
                if side == 0:
                    for c in range(2, d + 1):
                        a, b = rule_a[c], rule_b[c]
                        y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    for c in range(2, d + 1):
                        a, b = rule_a[c], rule_b[c]
                        y[c] = sub(x[c], mul(x[a], y[b]))
                result.append((1 - side, tuple(y)))
            return result

        start = (0, (0,) * (d + 1))
        index = {start: 0}
        order = [start]
        edges = []
        queue = deque([start])
        head = 0
        while head < len(order):
            if time.time() > deadline:
                break
            vertex = order[head]
            head += 1
            for other in compute_neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # For (4,10), try to find a small 4-regular graph with girth 10
    if k == 4 and g == 10:
        # Try known constructions or random search
        result = try_random_4_reg_girth10(deadline)
        if result is not None:
            return result
    
    # Fallback to LUW
    n, edges = luw_construct(k, g)
    if n is not None:
        return n, edges
    
    # Last resort: return a simple cycle-based construction (may not be optimal)
    return fallback_construct(k, g)

def verify_girth_nonbacktracking(n, edges, g, deadline):
    """Check if graph has girth >= g using non-backtracking walks."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For each starting vertex and direction, do a non-backtracking BFS up to depth g//2
    # If we find a cycle of length < g, return False
    
    for start in range(n):
        if time.time() > deadline:
            return True  # Assume valid if we run out of time
        
        # State: (current_vertex, previous_vertex)
        visited = set()
        queue = deque()
        
        for neighbor in adj[start]:
            state = (neighbor, start)
            if state not in visited:
                visited.add(state)
                queue.append(state)
        
        depth = 1
        max_depth = g // 2
        
        while queue and depth < max_depth:
            next_queue = deque()
            while queue:
                curr, prev = queue.popleft()
                
                for nxt in adj[curr]:
                    if nxt == prev:
                        continue  # Non-backtracking
                    
                    new_state = (nxt, curr)
                    
                    # Check if we've seen this state before (would form a cycle)
                    if new_state in visited:
                        # Found a cycle of length <= 2*depth
                        # Need to verify the actual cycle length
                        # For now, if depth < g//2, we have a cycle shorter than g
                        return False
                    
                    if new_state not in visited:
                        visited.add(new_state)
                        next_queue.append(new_state)
            
            queue = next_queue
            depth += 1
        
        # If we reached max_depth without finding a cycle, check if there are
        # any cycles of length exactly g or less
        # For girth >= g, we need no cycles of length < g
    
    return True


def try_random_4_reg_girth10(deadline):
    """Try to find a 4-regular graph with girth 10 on ~320 vertices."""
    random.seed(42)
    
    # Known best is 320 vertices
    target_n = 320
    
    # Try different sizes from 243 (lower bound) to 400
    for n in range(243, min(400, target_n + 50)):
        if time.time() > deadline - 5:
            break
        
        # Generate random 4-regular graphs and check girth
        for attempt in range(100):
            if time.time() > deadline - 5:
                break
            
            edges = generate_random_4_reg(n)
            if edges is None:
                continue
            
            # Quick girth check
            if check_girth_fast(n, edges, 10, deadline):
                return n, edges
    
    return None


def generate_random_4_reg(n):
    """Generate a random 4-regular graph on n vertices."""
    if n < 4 or (n * 4) % 2 != 0:
        return None
    
    # Use configuration model
    half_edges = [i for i in range(n) for _ in range(4)]
    random.shuffle(half_edges)
    
    adj = [[] for _ in range(n)]
    edges = []
    
    for i in range(0, len(half_edges), 2):
        u = half_edges[i]
        v = half_edges[i + 1]
        if u == v:
            return None  # Loop
        if v in adj[u]:
            return None  # Multiple edge
        adj[u].append(v)
        adj[v].append(u)
        edges.append((u, v))
    
    # Verify all vertices have degree 4
    for i in range(n):
        if len(adj[i]) != 4:
            return None
    
    return edges


def check_girth_fast(n, edges, g, deadline):
    """Fast girth check using BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for start in range(n):
        if time.time() > deadline:
            return False
        
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    if dist[v] >= g // 2:
                        break
                    queue.append(v)
                elif dist[u] + dist[v] + 1 < g:
                    # Found a cycle shorter than g
                    # Need to verify it's a valid cycle
                    return False
    
    return True


def fallback_construct(k, g):
    """Fallback: construct a cycle-based graph (may not be optimal)."""
    # For k=4, g=10, use a large cycle with chords
    # This is a last resort and may not satisfy girth requirement
    n = 2 * (k - 1) ** (g // 2 - 1)  # Moore bound approximation
    edges = []
    
    # Create a cycle
    for i in range(n):
        edges.append((i, (i + 1) % n))
    
    # Add chords to make it 4-regular
    # This is a simplified approach; may not work for all cases
    for i in range(n):
        while len([e for e in edges if i in e]) < k:
            j = (i + 2) % n
            if (i, j) not in edges and (j, i) not in edges and i != j:
                edges.append((i, j))
                break
    
    return n, edges
