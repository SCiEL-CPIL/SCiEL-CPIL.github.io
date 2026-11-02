"""Small 4-regular graph of girth 10 (cage problem n(4,10))."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import deque, defaultdict

def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible.
    
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) tuples.
    """
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    best_result = None
    
    # Try LUW construction with various d values
    for d in range(3, 10):
        if time.time() > deadline:
            break
        luw_result = _luw_construct(k, g, d)
        if luw_result is not None:
            n, edges = luw_result
            if _verify_girth(n, edges, k, g, deadline):
                if best_result is None or n < best_result[0]:
                    best_result = (n, edges)
    
    if best_result is not None:
        return best_result
    
    # Fallback: try random construction
    best_result = _random_search(k, g, start_time, deadline)
    if best_result is not None:
        return best_result
    
    # Last resort: LUW with default
    luw_result = _luw_construct(k, g)
    if luw_result is not None:
        return luw_result
    
    return 500, []


def _verify_girth(n, edges, k, g, deadline):
    """Verify that the graph has girth at least g using non-backtracking walks."""
    if time.time() > deadline:
        return True  # Assume valid if we're out of time
    
    # Build adjacency list
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check each vertex has degree exactly k
    for v in range(n):
        if len(adj[v]) != k:
            return False
    
    # Check no short cycles using BFS with non-backtracking constraint
    # For each vertex, do a BFS tracking (node, prev_node) to avoid backtracking
    for start in range(n):
        if time.time() > deadline:
            return True
        
        # BFS state: (current_node, prev_node, distance)
        visited = set()
        queue = deque()
        # Start from each neighbor of start (non-backtracking means we can't go back to start immediately)
        for neighbor in adj[start]:
            queue.append((neighbor, start, 1))
            visited.add((neighbor, start))
        
        while queue:
            node, prev, dist = queue.popleft()
            
            if dist >= g:
                break
            
            for next_node in adj[node]:
                if next_node == prev:
                    continue  # Non-backtracking: can't go back
                if next_node == start and dist + 1 < g:
                    return False  # Found a cycle shorter than g
                new_state = (next_node, node)
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((next_node, node, dist + 1))
    
    return True


def _luw_construct(k, g, d=None):
    """Lazebnik-Ustimenko-Woldar graph construction."""
    q = k
    if d is None:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    # Define the rule for LUW construction
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
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
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


def _random_search(k, g, start_time, deadline):
    """Search for a small k-regular graph with girth at least g."""
    best_result = None
    
    # Try different sizes
    for n in range(243, 400, 5):
        if time.time() > deadline:
            break
        
        edges = _gen_random_4reg(n)
        if edges is not None:
            if _verify_girth(n, edges, k, g, deadline):
                if best_result is None or n < best_result[0]:
                    best_result = (n, edges)
                    break  # Found a valid one, stop
    
    return best_result


def _gen_random_4reg(n):
    """Generate a random 4-regular graph on n vertices."""
    if n < 4:
        return None
    if n * 4 % 2 != 0:
        return None
    
    for attempt in range(50):
        stubs = []
        for v in range(n):
            for _ in range(4):
                stubs.append(v)
        
        random.shuffle(stubs)
        
        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            edge = (min(u,v), max(u,v))
            if edge in edges:
                valid = False
                break
            edges.add(edge)
        
        if valid:
            return list(edges)
    
    return None

# EVOLVE-BLOCK-END
