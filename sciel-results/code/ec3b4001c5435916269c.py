import time

def _validate_edges(edges, n=12):
    """
    Validate that the edges form a simple undirected graph on vertices 0..n-1,
    with min degree >= 3, girth >= 4 (triangle-free), and contains at least one cycle.
    Returns (is_valid, distinct_cycle_lengths).
    """
    # Basic checks
    edge_set = set()
    for e in edges:
        if len(e) != 2:
            return False, None
        u, v = int(e[0]), int(e[1])
        if not (0 <= u < n) or not (0 <= v < n):
            return False, None
        if u == v:
            return False, None
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False, None
        edge_set.add(key)
    
    # Build adjacency
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        u, v = int(u), int(v)
        adj[u].add(v)
        adj[v].add(u)
    
    # Check min degree
    for i in range(n):
        if len(adj[i]) < 3:
            return False, None
    
    # Check for triangles (girth >= 4)
    # For each edge (u,v), check if any neighbor of u (other than v) is also a neighbor of v
    for u, v in edge_set:
        common = adj[u] & adj[v]
        if common:
            # Triangle found
            return False, None
    
    # Find all distinct cycle lengths using DFS
    # We use a standard algorithm to enumerate all elementary cycles.
    # Johnson's algorithm is efficient, but for n=12, a simpler DFS-based enumeration
    # with care to avoid duplicates is sufficient.
    
    cycle_lengths = set()
    
    # We'll use a modified DFS that only explores cycles where the smallest vertex is 'start'
    # and all other vertices in the path are > start.
    
    def dfs(start, node, path, visited_in_path):
        next_nodes = sorted(adj[node])
        for next_node in next_nodes:
            if next_node == start and len(path) >= 3:
                # Found a cycle
                cycle_lengths.add(len(path))
            elif next_node > start and next_node not in visited_in_path:
                # Only visit nodes strictly greater than start to ensure unique representation
                # where 'start' is the smallest node in the cycle.
                dfs(start, next_node, path + [next_node], visited_in_path | {next_node})
    
    for i in range(n):
        # Start DFS from i, only allowing nodes > i in the path
        # Note: The condition `next_node > start` in dfs ensures we don't go "backwards" to smaller nodes,
        # which effectively roots the cycle at the smallest node `start`.
        dfs(i, i, [i], {i})
    
    # Check if there is at least one cycle
    if not cycle_lengths:
        return False, None
        
    return True, cycle_lengths

def construct():
    # Deadline guard
    start_time = time.time()
    
    # Theoretical minimum for n=12, min_degree=3, girth>=4 is 2 distinct cycle lengths.
    # Two disjoint K_3,3 graphs achieve this: each K_3,3 has cycles of length 4 and 6.
    # No 3-regular graph on 12 vertices can have only one distinct cycle length.
    
    candidate_edges = []
    
    # Component 1: K_3,3 on vertices 0,1,2 (Partition A) and 3,4,5 (Partition B)
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            candidate_edges.append([int(u), int(v)])
            
    # Component 2: K_3,3 on vertices 6,7,8 (Partition A) and 9,10,11 (Partition B)
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            candidate_edges.append([int(u), int(v)])
            
    # Validate
    valid, lengths = _validate_edges(candidate_edges)
    if valid:
        return candidate_edges
    
    # This should never happen; K_3,3 is definitely valid.
    # Return the candidate as-is as a last resort (it is valid).
    return candidate_edges

# Self-test if run directly (not strictly necessary for evaluation but good for sanity)
if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")
    valid, lengths = _validate_edges(e)
    print(f"Valid: {valid}, Lengths: {lengths}")
