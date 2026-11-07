"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import time
import random
import itertools
from collections import defaultdict

def _enumerate_simple_cycles(adj, n, max_cycles=50000):
    """Enumerate simple cycles in an undirected graph. Returns set of cycle lengths."""
    cycle_lengths = set()
    adj_set = [set() for _ in range(n)]
    for u in range(n):
        for v in adj[u]:
            adj_set[u].add(v)
    
    def dfs(start, current, visited, path):
        nonlocal cycle_lengths, max_cycles, _cycle_count
        if _cycle_count >= max_cycles:
            return
        for neighbor in adj_set[current]:
            if neighbor == start and len(path) >= 3:
                # Found a cycle
                cycle_lengths.add(len(path))
                _cycle_count += 1
                if _cycle_count >= max_cycles:
                    return
            elif neighbor not in visited and neighbor > start:
                # Only visit neighbors with higher index to avoid duplicates
                # This ensures each cycle is found exactly once (starting from smallest vertex)
                visited.add(neighbor)
                path.append(neighbor)
                dfs(start, neighbor, visited, path)
                path.pop()
                visited.remove(neighbor)
    
    _cycle_count = 0
    for start in range(n):
        if _cycle_count >= max_cycles:
            break
        visited = {start}
        path = [start]
        dfs(start, start, visited, path)
    
    return cycle_lengths

def _check_graph(edges, n=12):
    """Check if graph is valid and return distinct cycle lengths count."""
    adj = defaultdict(list)
    edge_set = set()
    for u, v in edges:
        if u == v:
            return None
        if u > v:
            u, v = v, u
        if (u, v) in edge_set:
            return None
        edge_set.add((u, v))
        adj[u].append(v)
        adj[v].append(u)
    
    # Check min degree
    for i in range(n):
        if len(adj[i]) < 3:
            return None
    
    # Check triangle-free
    for u in range(n):
        for v in adj[u]:
            for w in adj[v]:
                if w == u:
                    continue
                if w in adj[u]:
                    return None
    
    # Check has a cycle
    # BFS/DFS for cycle detection
    visited = [False] * n
    has_cycle = False
    for start in range(n):
        if not visited[start]:
            stack = [(start, -1, 0)]
            while stack:
                node, parent, depth = stack.pop()
                if visited[node]:
                    continue
                visited[node] = True
                for neighbor in adj[node]:
                    if neighbor == parent:
                        continue
                    if visited[neighbor]:
                        has_cycle = True
                        break
                    stack.append((neighbor, node, depth + 1))
                if has_cycle:
                    break
        if has_cycle:
            break
    
    if not has_cycle:
        return None
    
    # Enumerate cycles
    cycle_lengths = _enumerate_simple_cycles(adj, n, max_cycles=100000)
    if not cycle_lengths:
        return None
    
    return len(cycle_lengths)

def _generate_random_bipartite_graph(n=12, min_degree=3, seed=None):
    """Generate a random bipartite graph with min degree >= min_degree."""
    if seed is not None:
        random.seed(seed)
    
    # Random bipartition
    left_size = random.randint(3, n - 3)
    right_size = n - left_size
    left = list(range(left_size))
    right = list(range(left_size, n))
    
    edges = []
    for u in left:
        for v in right:
            if random.random() < 0.6:
                edges.append([u, v])
    
    # Ensure min degree
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for i in range(n):
        if len(adj[i]) < min_degree:
            # Add edges to boost degree
            if i < left_size:
                candidates = [v for v in right if v not in adj[i]]
            else:
                candidates = [u for u in left if u not in adj[i]]
            random.shuffle(candidates)
            for c in candidates:
                if len(adj[i]) >= min_degree:
                    break
                edges.append([min(i, c), max(i, c)])
                adj[i].append(c)
                adj[c].append(i)
    
    # Deduplicate
    edge_set = set()
    unique_edges = []
    for u, v in edges:
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key not in edge_set:
            edge_set.add(key)
            unique_edges.append([key[0], key[1]])
    
    return unique_edges

def construct():
    """
    Constructs an undirected simple graph on vertices 0..11.
    Searches for a graph minimizing the number of distinct simple cycle lengths.
    """
    deadline = time.time() + 50  # Safety margin before 60s limit
    best_score = 2
    best_edges = None
    
    # Initial: two K_{3,3}
    initial_edges = []
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            initial_edges.append([u, v])
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            initial_edges.append([u, v])
    
    score = _check_graph(initial_edges)
    if score is not None and score < best_score:
        best_score = score
        best_edges = [list(e) for e in initial_edges]
    
    # Random search
    rng = random.Random(42)
    attempts = 0
    while time.time() < deadline and attempts < 2000:
        attempts += 1
        edges = _generate_random_bipartite_graph(seed=rng.randint(0, 2**31))
        s = _check_graph(edges)
        if s is not None and s < best_score:
            best_score = s
            best_edges = [list(e) for e in edges]
            if best_score == 1:
                break
    
    if best_edges is None:
        best_edges = initial_edges
    
    return [[int(u), int(v)] for u, v in best_edges]

# EVOLVE-BLOCK-END
