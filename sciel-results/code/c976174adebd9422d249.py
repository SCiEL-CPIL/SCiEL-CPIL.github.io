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

    for d in [8, 9, 10, 11]:
        n, edges = build_luw(d)
        if _check_girth_fast(n, edges, g):
            return n, edges
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


def _greedy_tree_extension(k, g, target_n, deadline, seed=42):
    """
    Randomized greedy tree-extension: build a k-regular graph incrementally
    by adding vertices via BFS layering, checking girth at each step.
    
    Strategy:
    1. Start with a small k-regular tree (depth ~ g/2 - 1)
    2. Add new vertices one at a time, connecting to existing leaves
    3. After each addition, verify no short cycles were created
    4. Continue until we reach target_n vertices with all degrees = k
    5. Final girth check
    """
    if k != 3:
        return None
    
    random.seed(seed)
    max_depth = (g - 1) // 2  # 7 for g=16
    
    # Start with a 3-regular tree of depth max_depth
    # Root at 0, each node has 3 neighbors (one parent, 2 children except leaves)
    # For depth d: levels 0..d, level i has 3*2^i nodes (for i>=1), but root has 3 children
    # Total nodes: 1 + 3 + 6 + 12 + ... + 3*2^(d-1) = 1 + 3*(2^d - 1)
    # For d=7: 1 + 3*127 = 382 nodes. That's already a lot.
    # Actually, let's use a smaller initial tree and grow.
    
    # Start with just the root and its 3 neighbors
    n_current = 4
    adj = {0: [1, 2, 3], 1: [0], 2: [0], 3: [0]}
    edges = [(0, 1), (0, 2), (0, 3)]
    
    # Track which vertices are "leaves" (degree < k)
    # Initially: vertices 1, 2, 3 are leaves (degree 1)
    leaves = [1, 2, 3]
    
    # For each new vertex, we need to connect it to k existing leaves
    # But each leaf can only accept k-1 more connections (since it already has 1)
    # Wait, this approach is tricky. Let me think differently.
    
    # Alternative: Use the configuration model but with girth checking.
    # Generate stubs, pair them randomly, but reject pairings that create short cycles.
    
    def would_create_short_cycle(u, v, adj_dict, edge_set, max_check_depth):
        """Check if adding edge (u,v) would create a cycle shorter than g."""
        # BFS from u to v, avoiding the edge (u,v), with depth limit
        # If we find a path of length < g-1, then adding (u,v) creates a cycle of length <= g-1
        
        # Actually, we need to check if there's a path from u to v of length < g-1
        # that doesn't use the edge (u,v)
        
        visited = {u: 0}
        queue = deque([(u, 0)])
        
        while queue:
            node, depth = queue.popleft()
            if depth >= max_check_depth:
                continue
            for neighbor in adj_dict.get(node, []):
                if neighbor == v and depth + 1 < g:
                    # Found a path from u to v of length depth+1
                    # Adding (u,v) creates a cycle of length depth+2
                    if depth + 2 < g:
                        return True
                if neighbor not in visited:
                    visited[neighbor] = depth + 1
                    queue.append((neighbor, depth + 1))
        
        return False
    
    # This is too slow. Let's use a different approach.
    # Use the non-backtracking walk check more efficiently.
    
    # Actually, let's use a simpler approach:
    # 1. Generate a random 3-regular graph on target_n vertices
    # 2. Check girth
    # 3. If girth < g, find a short cycle and try to break it with edge swaps
    # 4. Repeat until girth >= g or time runs out
    
    def generate_random_3reg(n):
        """Generate a random 3-regular simple graph on n vertices using configuration model."""
        if n < 4 or n % 2 != 0:
            return None
        
        stubs = list(range(n * 3))
        for _ in range(200):  # 200 attempts
            random.shuffle(stubs)
            edge_set = set()
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i] // 3, stubs[i + 1] // 3
                if u == v:
                    valid = False
                    break
                key = (min(u, v), max(u, v))
                if key in edge_set:
                    valid = False
                    break
                edge_set.add(key)
            if valid:
                return list(edge_set)
        return None
    
    def check_girth_local(n, edges, min_girth, adj_cache=None):
        """Check if graph has girth >= min_girth using BFS from each vertex."""
        adj = adj_cache if adj_cache is not None else _build_adj(n, edges)
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
    
    def find_short_cycle(n, edges, min_girth):
        """Find a cycle shorter than min_girth, return it or None."""
        adj = _build_adj(n, edges)
        max_depth = (min_girth - 1) // 2
        
        for start in range(n):
            visited = [-1] * n
            parent = [-1] * n
            visited[start] = 0
            queue = deque([(start, 0)])
            
            while queue:
                node, depth = queue.popleft()
                if depth > max_depth:
                    continue
                for neighbor in adj[node]:
                    if visited[neighbor] == -1:
                        visited[neighbor] = depth + 1
                        parent[neighbor] = node
                        queue.append((neighbor, depth + 1))
                    elif neighbor != parent[node]:
                        cycle_len = depth + visited[neighbor] + 1
                        if cycle_len < min_girth:
                            # Reconstruct cycle
                            cycle = [node]
                            curr = node
                            while curr != neighbor:
                                curr = parent[curr]
                                cycle.append(curr)
                            # Add path from neighbor back
                            curr = neighbor
                            while curr != node:
                                curr = parent[curr]
                                cycle.append(curr)
                            return cycle
        return None
    
    # Try to find a valid graph
    best_result = None
    best_n = float('inf')
    
    # Try target_n and nearby values
    candidates = [target_n]
    for delta in range(2, 50, 2):
        candidates.append(target_n - delta)
        candidates.append(target_n + delta)
    candidates = [c for c in candidates if c >= 512 and c <= 1200 and c % 2 == 0]
    
    for n in candidates:
        if time.time() > deadline - 5:
            break
        
        for restart in range(5):
            if time.time() > deadline - 3:
                break
            
            edges = generate_random_3reg(n)
            if edges is None:
                continue
            
            # Try to improve girth via edge swaps
            adj = _build_adj(n, edges)
            edge_list = list(edges)
            
            for it in range(500):
                if time.time() > deadline - 2:
                    break
                
                if check_girth_local(n, edge_list, g, adj):
                    if n < best_n:
                        best_n = n
                        best_result = (n, edge_list)
                    break
                
                # Find a short cycle
                cycle = find_short_cycle(n, edge_list, g)
                if cycle is None:
                    break
                
                # Try to break the cycle by swapping an edge
                # Pick a random edge in the cycle and try to rewire it
                if len(cycle) >= 2:
                    # The cycle is a list of vertices, edges are between consecutive vertices
                    # Pick a random edge in the cycle
                    edge_idx = random.randint(0, len(cycle) - 1)
                    u = cycle[edge_idx]
                    v = cycle[(edge_idx + 1) % len(cycle)]
                    
                    # Find another edge to swap with
                    # We need to find an edge (a, b) not in the cycle such that
                    # swapping (u,v) and (a,b) with (u,a) and (v,b) or (u,b) and (v,a)
                    # maintains 3-regularity and simplicity
                    swapped = False
                    for _ in range(50):
                        idx = random.randint(0, len(edge_list) - 1)
                        a, b = edge_list[idx]
                        if a == u or a == v or b == u or b == v:
                            continue
                        
                        # Try swap 1: (u,v) -> (u,a), (a,b) -> (v,b)
                        new1 = (min(u, a), max(u, a))
                        new2 = (min(v, b), max(v, b))
                        
                        if new1 == new2:
                            continue
                        if new1 in edge_set or new2 in edge_set:
                            continue
                        if u == a or v == b or u == b or v == a:
                            continue
                        
                        # Check if this creates a short cycle
                        # Quick check: just try it and verify
                        old1 = (min(u, v), max(u, v))
                        old2 = (min(a, b), max(a, b))
                        
                        if old1 not in edge_list or old2 not in edge_list:
                            continue
                        
                        # Apply swap
                        edge_list.remove(old1)
                        edge_list.remove(old2)
                        edge_list.append(new1)
                        edge_list.append(new2)
                        swapped = True
                        break
                    
                    if not swapped:
                        # Try another approach: just pick a random edge swap
                        idx1 = random.randint(0, len(edge_list) - 1)
                        idx2 = random.randint(0, len(edge_list) - 1)
                        if idx1 == idx2:
                            continue
                        
                        u1, v1 = edge_list[idx1]
                        u2, v2 = edge_list[idx2]
                        
                        if u1 == u2 or v1 == v2 or u1 == v2 or v1 == u2:
                            continue
                        
                        new1 = (min(u1, u2), max(u1, u2))
                        new2 = (min(v1, v2), max(v1, v2))
                        
                        if new1 == new2:
                            continue
                        if new1 in edge_set or new2 in edge_set:
                            continue
                        
                        old1 = (min(u1, v1), max(u1, v1))
                        old2 = (min(u2, v2), max(u2, v2))
                        
                        edge_list.remove(old1)
                        edge_list.remove(old2)
                        edge_list.append(new1)
                        edge_list.append(new2)
            
            # Update edge_set for next iteration
            edge_set = set(edge_list)
            adj = _build_adj(n, edge_list)
    
    return best_result


def _dodecahedron_voltage_lift(k, g, deadline):
    """
    Voltage lift of the dodecahedron graph (3-regular, girth 5, 20 vertices, 30 edges).
    Find voltages on edges such that for all closed walks of length < g in the base,
    the voltage sum is not 0 mod m.
    
    The dodecahedron has cycles of length 5, 6, 8, 9, 10, 12, 14, 16, ...
    For g=16, we need to avoid voltage sum = 0 mod m for cycles of length 5, 6, 8, 9, 10, 12, 14.
    
    With m prime, if voltage sum != 0 mod m, the lift cycle length is m * L >= m * 5.
    For g=16, we need m * 5 >= 16, so m >= 4. Use m=5 or larger.
    
    n = 20 * m. For m=5, n=100. For m=7, n=140. Etc.
    """
    if k != 3:
        return None
    
    # Dodecahedron graph edges (20 vertices, 30 edges)
    # Standard dodecahedron: outer 10-cycle, inner 10-cycle (rotated), 10 spokes
    dodeca_edges = [
        # Outer 10-cycle: vertices 0-9
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 0),
        # Inner 10-cycle: vertices 10-19 (rotated)
        (10, 14), (14, 18), (18, 12), (12, 16), (16, 10),
        (11, 15), (15, 19), (19, 13), (13, 17), (17, 11),
        # Spokes: connect outer to inner
        (0, 10), (1, 11), (2, 12), (3, 13), (4, 14),
        (5, 15), (6, 16), (7, 17), (8, 18), (9, 19)
    ]
    
    n_base = 20
    n_edges = len(dodeca_edges)  # 30
    
    adj_base = [[] for _ in range(n_base)]
    for i, (u, v) in enumerate(dodeca_edges):
        adj_base[u].append((v, i))
        adj_base[v].append((u, i))
    
    # Find all cycles in dodecahedron up to length g-1
    cycles = []
    seen = set()
    
    def dfs_cycles(start, curr, visited, edge_path, length):
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
                dfs_cycles(start, next_v, visited, edge_path + [edge_idx], length + 1)
                visited.remove(next_v)
    
    for start in range(n_base):
        visited = {start}
        dfs_cycles(start, start, visited, [], 0)
    
    # Try different primes m
    primes = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    
    for m in primes:
        if time.time() > deadline:
            break
        
        # Random search for voltages
        for attempt in range(2000):
            if time.time() > deadline:
                break
            voltages = [random.randint(1, m - 1) for _ in range(n_edges)]  # Non-zero voltages
            valid = True
            for C in cycles:
                s = sum(voltages[i] for i in C) % m
                if s == 0:
                    valid = False
                    break
            if valid:
                # Construct lift
                n = n_base * m
                edges = []
                for idx, (u, v) in enumerate(dodeca_edges):
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

    # First try dodecahedron voltage lift (fast, deterministic if voltages found)
    result = _dodecahedron_voltage_lift(k, g, deadline - 30)
    if result is not None:
        n, edges = result
        if _check_girth_fast(n, edges, g):
            return n, edges

    # Try greedy tree extension / random search with local improvement
    result = _greedy_tree_extension(k, g, 936, deadline - 10, seed=42)
    if result is not None:
        n, edges = result
        if _check_girth_fast(n, edges, g):
            return n, edges

    # Fallback to LUW
    return _luw_construct(k, g)
