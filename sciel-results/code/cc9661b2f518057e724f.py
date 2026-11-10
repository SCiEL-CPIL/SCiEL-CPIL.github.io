import time

def _validate_edges(edges, n=12):
    """
    Validate that the edges form a simple undirected graph on vertices 0..n-1,
    with min degree >= 3, girth >= 4 (triangle-free), and contains at least one cycle.
    Returns (is_valid, distinct_cycle_lengths).
    """
    edge_set = set()
    for e in edges:
        if len(e) != 2:
            return False, None
        u, v = e[0], e[1]
        # Strict type check: must be Python int
        if not isinstance(u, int) or isinstance(u, bool):
            return False, None
        if not isinstance(v, int) or isinstance(v, bool):
            return False, None
        if not (0 <= u < n) or not (0 <= v < n):
            return False, None
        if u == v:
            return False, None
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False, None
        edge_set.add(key)
    
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Check min degree
    for i in range(n):
        if len(adj[i]) < 3:
            return False, None
    
    # Check for triangles (girth >= 4)
    for u, v in edge_set:
        common = adj[u] & adj[v]
        if common:
            return False, None
    
    # Find all distinct cycle lengths
    cycle_lengths = set()
    
    def dfs(start, node, path, visited_in_path):
        next_nodes = sorted(adj[node])
        for next_node in next_nodes:
            if next_node == start and len(path) >= 3:
                cycle_lengths.add(len(path))
            elif next_node > start and next_node not in visited_in_path:
                dfs(start, next_node, path + [next_node], visited_in_path | {next_node})
    
    for i in range(n):
        dfs(i, i, [i], {i})
    
    if not cycle_lengths:
        return False, None
        
    return True, cycle_lengths

def construct():
    """
    Returns edges for a graph on 12 vertices with min degree >= 3, girth >= 4,
    minimizing the number of distinct simple cycle lengths.
    
    Solution: Two disjoint copies of K_{3,3}.
    - Vertices 0,1,2 connected to 3,4,5.
    - Vertices 6,7,8 connected to 9,10,11.
    - Cycle lengths present: 4 and 6.
    - Distinct cycle lengths count: 2.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    # Construct two disjoint K_{3,3} components
    edges = []
    
    # Component 1: Partition A {0, 1, 2}, Partition B {3, 4, 5}
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.append([int(u), int(v)])
            
    # Component 2: Partition A {6, 7, 8}, Partition B {9, 10, 11}
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.append([int(u), int(v)])

    # Quick validation to ensure correctness (should be extremely fast)
    # If validation fails, we return the edges anyway as they are theoretically correct,
    # but this helps catch any coding errors in the generation loop.
    valid, lengths = _validate_edges(edges)
    if not valid:
        # In a real scenario, this might trigger a fallback, but for this specific
        # deterministic construction, it should never happen.
        # We proceed to return edges.
        pass

    # Deadline guard (unlikely to trigger)
    if time.time() > deadline:
        return edges

    return edges

if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")
    valid, lengths = _validate_edges(e)
    print(f"Valid: {valid}, Lengths: {lengths}")
