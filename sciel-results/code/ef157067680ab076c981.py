import numpy as np
import random
import time
from collections import deque

def construct(k, g):
    if k != 4 or g != 10:
        return _luw_construct(k, g)
    
    start_time = time.time()
    time_limit = 110.0
    
    # Try to find a 4-regular girth-10 graph starting from small N
    # Best known is 320, Moore bound is 242
    for N in range(242, 330):
        if time.time() - start_time > time_limit:
            break
        
        edges = _find_cage_structured(N, k, g)
        if edges is not None:
            return N, edges
    
    return _luw_construct(k, g)

def _find_cage_structured(N, k, g):
    """Find k-regular graph of girth >= g using targeted cycle-breaking."""
    # Generate initial random k-regular graph
    A = _random_regular_graph_fast(N, k)
    if A is None:
        return None
    
    # Get edge list for efficient manipulation
    edge_list = _adj_to_edge_list(A, N)
    
    # Iteratively break short cycles
    max_iters = 20000
    for it in range(max_iters):
        # Find all short cycles (length < g)
        short_cycles = _find_short_cycles(A, N, g)
        if not short_cycles:
            return _adj_to_edges(A)
        
        # Pick a random short cycle and break it
        cycle = random.choice(short_cycles)
        if not _break_cycle(A, edge_list, N, k, cycle):
            # If we can't break this cycle, try another
            continue
    
    # Final check
    if not _find_short_cycles(A, N, g):
        return _adj_to_edges(A)
    return None

def _random_regular_graph_fast(N, k):
    """Generate random k-regular graph using configuration model with retry."""
    for attempt in range(50):
        stubs = []
        for i in range(N):
            stubs.extend([i] * k)
        random.shuffle(stubs)
        
        A = np.zeros((N, N), dtype=np.int8)
        valid = True
        for i in range(0, N * k, 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v or A[u, v]:
                valid = False
                break
            A[u, v] = A[v, u] = 1
        if valid:
            return A
    return None

def _adj_to_edge_list(A, N):
    """Convert adjacency matrix to list of edges (u, v) with u < v."""
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if A[i, j]:
                edges.append((i, j))
    return edges

def _find_short_cycles(A, N, g):
    """Find all cycles of length < g using BFS from each vertex."""
    cycles = []
    for start in range(N):
        # BFS up to depth g//2 - 1
        max_depth = g // 2 - 1
        # visited[v][parent] = depth, to track non-backtracking paths
        # For detecting cycles of length < g, we do BFS and look for back-edges
        visited = [dict() for _ in range(N)]  # visited[v] = {parent: depth}
        queue = deque([(start, -1, 0)])
        visited[start][-1] = 0
        
        while queue:
            node, parent, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for neighbor in np.nonzero(A[node])[0]:
                if neighbor == parent:
                    continue
                if neighbor in visited[node] and visited[node][neighbor] < depth + 1:
                    # Found a cycle
                    # Reconstruct cycle
                    cycle = _reconstruct_cycle(start, node, neighbor, visited, A, N)
                    if cycle and len(cycle) < g:
                        cycles.append(cycle)
                elif neighbor not in visited[node] or visited[node].get(neighbor, float('inf')) > depth + 1:
                    visited[node][neighbor] = depth + 1
                    visited[neighbor][node] = depth + 1
                    queue.append((neighbor, node, depth + 1))
    
    return cycles

def _reconstruct_cycle(start, node, neighbor, visited, A, N):
    """Reconstruct a cycle from BFS tree."""
    # This is complex; for simplicity, we just return a marker
    # In practice, we just need to know a short cycle exists
    return [start, node, neighbor]

def _break_cycle(A, edge_list, N, k, cycle):
    """Try to break a cycle by swapping edges."""
    # Pick a random edge from the cycle
    if len(cycle) < 3:
        return False
    u, v = cycle[0], cycle[1]
    if not A[u, v]:
        return False
    
    # Find another edge to swap with
    for _ in range(10):
        idx = random.randint(0, len(edge_list) - 1)
        u2, v2 = edge_list[idx]
        if (u2, v2) == (u, v) or (u2, v2) == (v, u):
            continue
        # Try swap: (u,v) and (u2,v2) -> (u,u2) and (v,v2)
        if A[u, u2] == 0 and A[v, v2] == 0 and u != u2 and v != v2:
            A[u, v] = A[v, u] = 0
            A[u2, v2] = A[v2, u2] = 0
            A[u, u2] = A[u2, u] = 1
            A[v, v2] = A[v2, v] = 1
            # Update edge_list
            edge_list[idx] = (u, u2) if u < u2 else (u2, u)
            edge_list.append((v, v2) if v < v2 else (v2, v))
            return True
        # Try swap: (u,v) and (u2,v2) -> (u,v2) and (v,u2)
        if A[u, v2] == 0 and A[v, u2] == 0 and u != v2 and v != u2:
            A[u, v] = A[v, u] = 0
            A[u2, v2] = A[v2, u2] = 0
            A[u, v2] = A[v2, u] = 1
            A[v, u2] = A[u2, v] = 1
            edge_list[idx] = (u, v2) if u < v2 else (v2, u)
            edge_list.append((v, u2) if v < u2 else (u2, v))
            return True
    return False

def _adj_to_edges(A):
    edges = []
    n = A.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j]:
                edges.append((i, j))
    return edges

def _luw_construct(k, g):
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
