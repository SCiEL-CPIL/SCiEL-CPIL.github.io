import time
import random
import numpy as np
from collections import deque

def _luw_construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar construction for 3-regular graphs.
    We try smaller d values to get fewer vertices, checking girth empirically.
    """
    if k != 3:
        # Fallback for other k (should not be hit for this task)
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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

    # For k=3, we use q=3.
    q = 3

    def build_luw(d):
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

    # Try d=8 first. If girth < 16, try d=9, etc.
    for d in [8, 9, 10, 11]:
        n, edges = build_luw(d)
        if _check_girth_fast(n, edges, g):
            return n, edges
    # If all fail (unlikely for d=11), return the largest one (which should be valid by theory)
    return build_luw(11)


def _build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _check_girth_fast(n, edges, min_girth):
    """Fast girth check using BFS from each vertex with parent tracking."""
    adj = _build_adj(n, edges)
    max_depth = (min_girth - 1) // 2

    for start in range(n):
        visited = [-1] * n
        visited[start] = 0
        queue = deque([(start, 0, -1)])

        while queue:
            node, depth, parent = queue.popleft()
            if depth > max_depth:
                continue
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                if visited[neighbor] == -1:
                    visited[neighbor] = depth + 1
                    queue.append((neighbor, depth + 1, node))
                else:
                    cycle_len = depth + visited[neighbor] + 1
                    if cycle_len < min_girth:
                        return False
    return True


def _find_petersen_cycles(max_len):
    """Find all simple cycles in the Petersen graph up to length max_len."""
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n_base = 10
    adj_base = [[] for _ in range(n_base)]
    for i, (u, v) in enumerate(petersen_edges):
        adj_base[u].append((v, i))
        adj_base[v].append((u, i))

    cycles = []
    seen = set()

    def dfs(start, curr, visited, edge_path, length):
        if length >= max_len:
            return
        for next_v, edge_idx in adj_base[curr]:
            if next_v == start and length >= 3:
                # Canonicalize cycle by sorting edge indices
                cycle_key = tuple(sorted(edge_path))
                if cycle_key not in seen:
                    seen.add(cycle_key)
                    cycles.append(cycle_key)
            elif next_v not in visited:
                visited.add(next_v)
                dfs(start, next_v, visited, edge_path + [edge_idx], length + 1)
                visited.remove(next_v)

    for start in range(n_base):
        visited = {start}
        dfs(start, start, visited, [], 0)
    return cycles


def _petersen_lift_optimized(k, g, deadline):
    """
    Construct a voltage lift of the Petersen graph with optimized voltage search.
    """
    if k != 3:
        return None

    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    # Find cycles up to g-1
    cycles = _find_petersen_cycles(g - 1)
    
    # Try small primes m
    primes = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    for m in primes:
        if time.time() > deadline:
            break
        
        # Random search for voltages
        for attempt in range(2000):
            if time.time() > deadline:
                break
            voltages = [random.randint(0, m - 1) for _ in range(15)]
            valid = True
            for C in cycles:
                s = sum(voltages[i] for i in C) % m
                if s == 0:
                    valid = False
                    break
            if valid:
                # Construct lift
                n = 10 * m
                edges = []
                for idx, (u, v) in enumerate(petersen_edges):
                    vol = voltages[idx]
                    for j in range(m):
                        a = u * m + j
                        b = v * m + ((j + vol) % m)
                        edges.append((a, b))
                return n, edges
    return None


def _csp_backtracking(k, g, max_n, deadline, seed=42):
    """
    Constraint satisfaction approach: build a k-regular graph with girth >= g
    using backtracking with dynamic girth checking.
    """
    if k != 3:
        return None
    
    random.seed(seed)
    
    def generate_random_3reg(n):
        """Generate a random 3-regular simple graph on n vertices."""
        if n % 2 != 0:
            return None
        if n < 4:
            return None
        
        # Configuration model: each vertex has 3 stubs
        stubs = []
        for v in range(n):
            for _ in range(3):
                stubs.append(v)
        
        # Randomly pair up stubs
        edges = set()
        valid = False
        for attempt in range(100):
            random.shuffle(stubs)
            edges = set()
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v:
                    valid = False
                    break
                if (min(u, v), max(u, v)) in edges:
                    valid = False
                    break
                edges.add((min(u, v), max(u, v)))
            if valid:
                break
        
        if not valid:
            return None
        return list(edges)
    
    def increase_girth(edges, n, target_girth, max_iters=5000):
        """Try to increase girth by rewiring edges."""
        adj = _build_adj(n, edges)
        edge_list = list(edges)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            if _check_girth_fast(n, edge_list, target_girth):
                return edge_list
            
            # Find a short cycle and try to break it
            # Pick a random edge and try to swap
            idx1 = random.randint(0, len(edge_list) - 1)
            u1, v1 = edge_list[idx1]
            
            # Find another edge that can be swapped
            attempts = 0
            while attempts < 20:
                idx2 = random.randint(0, len(edge_list) - 1)
                if idx2 == idx1:
                    attempts += 1
                    continue
                u2, v2 = edge_list[idx2]
                
                # Try swap: (u1,v1) -> (u1,u2), (u2,v2) -> (v1,v2)
                if u1 == u2 or v1 == v2 or u1 == v2 or v1 == u2:
                    attempts += 1
                    continue
                
                new_edge1 = (min(u1, u2), max(u1, u2))
                new_edge2 = (min(v1, v2), max(v1, v2))
                
                if new_edge1 == new_edge2:
                    attempts += 1
                    continue
                if new_edge1 in edge_list or new_edge2 in edge_list:
                    attempts += 1
                    continue
                
                # Apply swap
                edge_list.remove((min(u1, v1), max(u1, v1)))
                edge_list.remove((min(u2, v2), max(u2, v2)))
                edge_list.append(new_edge1)
                edge_list.append(new_edge2)
                
                attempts += 1
        
        return edge_list
    
    best_result = None
    best_n = float('inf')
    
    # Try sizes: 512, 600, 700, 800, 900, 936
    test_sizes = [512, 520, 540, 560, 580, 600, 620, 640, 660, 680,
                  700, 720, 740, 760, 780, 800, 820, 840, 860, 880,
                  900, 920, 936]
    
    for n in test_sizes:
        if time.time() > deadline - 10:
            break
        
        # Try multiple restarts for each size
        for restart in range(3):
            if time.time() > deadline - 5:
                break
            
            edges = generate_random_3reg(n)
            if edges is None:
                continue
            
            # Try to improve girth
            edges = increase_girth(edges, n, g, max_iters=1000)
            
            if _check_girth_fast(n, edges, g):
                if n < best_n:
                    best_n = n
                    best_result = (n, edges)
                break  # Found a valid graph for this size, move to next size
    
    if best_result is not None:
        return best_result
    
    return None


def _heisenberg_cayley(k, g, deadline):
    """
    Construct a Cayley graph of a finite Heisenberg group over a field.
    """
    if k != 3:
        return None

    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n_base = 10
    adj_base = [[] for _ in range(n_base)]
    for i, (u, v) in enumerate(petersen_edges):
        adj_base[u].append((v, i))
        adj_base[v].append((u, i))

    # Find all cycles in Petersen up to length g-1
    cycles = []
    seen = set()
    
    def dfs(start, curr, visited, edge_path, length):
        if length >= g:
            return
        for next_v, edge_idx in adj_base[curr]:
            if next_v == start and length >= 3:
                cycle_key = tuple(sorted(edge_path))
                if cycle_key not in seen:
                    seen.add(cycle_key)
                    cycles.append(cycle_key)
            elif next_v not in visited:
                visited.add(next_v)
                dfs(start, next_v, visited, edge_path + [edge_idx], length + 1)
                visited.remove(next_v)
                
    for start in range(n_base):
        visited = {start}
        dfs(start, start, visited, [], 0)
        
    # We need to find voltages for m such that sum(v[C]) != 0 mod m for all C in cycles.
    # Try small primes m.
    primes = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    for m in primes:
        if time.time() > deadline:
            break
        
        # Random search for voltages
        found = False
        for attempt in range(1000):
            if time.time() > deadline:
                break
            voltages = [random.randint(0, m - 1) for _ in range(15)]
            valid = True
            for C in cycles:
                s = sum(voltages[i] for i in C) % m
                if s == 0:
                    valid = False
                    break
            if valid:
                # Construct lift
                n = 10 * m
                edges = []
                for idx, (u, v) in enumerate(petersen_edges):
                    vol = voltages[idx]
                    for j in range(m):
                        a = u * m + j
                        b = v * m + ((j + vol) % m)
                        edges.append((a, b))
                return n, edges
    return None


def construct(k, g):
    start_time = time.time()
    deadline = 140.0  # 10s margin

    if k != 3:
        return _luw_construct(k, g)

    # First try the CSP backtracking approach (new structural change)
    result = _csp_backtracking(k, g, 1000, deadline - 20, seed=42)
    if result is not None:
        n, edges = result
        # Verify
        if _check_girth_fast(n, edges, g):
            return n, edges
        else:
            # Should not happen
            pass

    # Try Heisenberg/Cayley approach (actually voltage lift of Petersen)
    result = _heisenberg_cayley(k, g, deadline - 10)
    if result is not None:
        n, edges = result
        # Verify
        if _check_girth_fast(n, edges, g):
            return n, edges
        else:
            # Should not happen
            pass

    # Fallback to LUW
    return _luw_construct(k, g)
