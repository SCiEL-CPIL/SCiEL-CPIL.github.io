"""Optimized search for few cycle lengths under degree and girth constraints."""
# EVOLVE-BLOCK-START
import itertools
import time

def _build_adjacency(edges, n=12):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _validate(edges, n=12):
    """Check all constraints: min degree 3, triangle-free, has a cycle, no loops/duplicates."""
    if len(edges) == 0:
        return False
    # Check no loops or duplicates
    seen = set()
    for u, v in edges:
        if u == v:
            return False
        key = (min(u, v), max(u, v))
        if key in seen:
            return False
        seen.add(key)
    adj = _build_adjacency(edges, n)
    # Check degree >= 3
    for i in range(n):
        if len(adj[i]) < 3:
            return False
    # Check triangle-free
    for u in range(n):
        for v in adj[u]:
            for w in adj[v]:
                if w != u and w in adj[u]:
                    return False
    # Check has at least one cycle (all vertices have degree >= 3, so definitely has cycles)
    return True

def _get_distinct_cycle_lengths(edges, n=12, max_cycles=5000):
    """Find all distinct simple cycle lengths using DFS."""
    adj = _build_adjacency(edges, n)
    lengths = set()
    
    # For each vertex, find all simple cycles starting and ending there
    # where the vertex is the minimum in the cycle (to avoid duplicates)
    for start in range(n):
        # DFS to find all cycles
        def dfs(current, path, visited):
            if len(path) > 12:
                return
            if len(path) >= 3 and current == start and min(path) == start:
                lengths.add(len(path) - 1)
                return
            for neighbor in adj[current]:
                if neighbor == start and len(path) >= 4:
                    # Found a cycle back to start
                    if min(path) == start:
                        lengths.add(len(path))
                elif neighbor not in visited and neighbor > start:
                    # Only visit vertices greater than start to avoid duplicate cycles
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(neighbor, path, visited)
                    path.pop()
                    visited.remove(neighbor)
        
        visited = {start}
        dfs(start, [start], visited)
    
    return lengths

def _try_k33_union():
    """Return edges for two disjoint K_{3,3} components (cycle lengths {4, 6})."""
    edges = []
    part_a_1 = [0, 1, 2]
    part_b_1 = [3, 4, 5]
    for u in part_a_1:
        for v in part_b_1:
            edges.append([u, v])
    
    part_a_2 = [6, 7, 8]
    part_b_2 = [9, 10, 11]
    for u in part_a_2:
        for v in part_b_2:
            edges.append([u, v])
    
    result = [[int(u), int(v)] for u, v in edges]
    return result

def _search_for_single_length_graph(n=12, deadline=85.0):
    """Search for a graph with exactly 1 distinct cycle length."""
    start_time = time.time()
    
    # Try random 3-regular bipartite graphs
    # A 3-regular bipartite graph on 12 vertices has 18 edges
    # Try to construct such graphs and check cycle lengths
    
    import random
    rng = random.Random(42)
    
    # Generate random 3-regular bipartite graphs
    # Part A: 0-5, Part B: 6-11
    part_a = list(range(6))
    part_b = list(range(6, 12))
    
    found = None
    
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        # Generate a random 3-regular bipartite graph using configuration model
        # Each vertex in A needs degree 3, each in B needs degree 3
        a_stubs = []
        for v in part_a:
            a_stubs.extend([v, v, v])
        b_stubs = []
        for v in part_b:
            b_stubs.extend([v, v, v])
        
        # Randomly pair stubs
        b_perm = b_stubs[:]
        rng.shuffle(b_perm)
        edges = set()
        valid = True
        for a, b in zip(a_stubs, b_perm):
            if a == b:
                valid = False
                break
            key = (min(a, b), max(a, b))
            if key in edges:
                valid = False
                break
            edges.add(key)
        
        if not valid or len(edges) != 18:
            continue
        
        edge_list = [[u, v] for u, v in edges]
        
        # Check constraints
        if not _validate(edge_list, n):
            continue
        
        # Get distinct cycle lengths
        cycle_lengths = _get_distinct_cycle_lengths(edge_list, n)
        
        if len(cycle_lengths) == 1:
            found = edge_list
            break
    
    return found

def construct():
    """
    Returns edges for a graph on 12 vertices (0-11) with minimum degree 3,
    triangle-free, and minimizing the number of distinct cycle lengths.
    """
    # First, try to find a graph with only 1 distinct cycle length
    # Use a time-limited search
    single_length_graph = _search_for_single_length_graph(deadline=80.0)
    
    if single_length_graph is not None:
        # Verify it's valid
        if _validate(single_length_graph):
            result = [[int(u), int(v)] for u, v in single_length_graph]
            return result
    
    # Fall back to the known optimal (value 2) solution: two K_{3,3} components
    result = _try_k33_union()
    return result
# EVOLVE-BLOCK-END
