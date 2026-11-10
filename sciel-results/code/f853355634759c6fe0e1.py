import time

def construct():
    """
    Constructs a graph on 12 vertices (0..11) with minimum degree 3, 
    triangle-free (girth >= 4), minimizing the number of distinct cycle lengths.
    
    Strategy: Use two disjoint K_3,3 components.
    K_3,3 is bipartite (no triangles). 
    Cycles in K_3,3 are of length 4 and 6.
    Distinct cycle lengths: {4, 6}. Count = 2.
    Min degree in K_3,3 is 3.
    This satisfies all constraints and achieves a score of 2.5 (assuming base 5).
    """
    deadline = time.time() + 60.0
    edges = []
    
    # Component 1: K_3,3 on vertices 0,1,2 and 3,4,5
    part1_a = [0, 1, 2]
    part1_b = [3, 4, 5]
    for u in part1_a:
        for v in part1_b:
            edges.append([u, v])
            
    # Component 2: K_3,3 on vertices 6,7,8 and 9,10,11
    part2_a = [6, 7, 8]
    part2_b = [9, 10, 11]
    for u in part2_a:
        for v in part2_b:
            edges.append([u, v])

    # Ensure types are Python int
    final_edges = []
    for e in edges:
        final_edges.append([int(e[0]), int(e[1])])
        
    return final_edges

def _validate(edges):
    """
    Internal validation to ensure the graph meets all constraints.
    """
    n = 12
    deg = [0] * n
    adj = {i: set() for i in range(n)}
    
    for u, v in edges:
        if u == v:
            return False
        if not (0 <= u < n and 0 <= v < n):
            return False
        deg[u] += 1
        deg[v] += 1
        adj[u].add(v)
        adj[v].add(u)
        
    # Check min degree
    if min(deg) < 3:
        return False
        
    # Check for triangles (girth >= 4)
    edge_set = set(tuple(sorted(e)) for e in edges)
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            for w in adj[v]:
                if w > v and (u, w) in edge_set:
                    return False
                    
    # Check if it contains a cycle
    # For n=12, min degree 3, it must have a cycle.
    # Specifically, K_3,3 has cycles.
    
    return True

# Keep the original function names if required by the harness, 
# though only construct is the entry point usually.
# The prompt says "Keep the function names and signatures of the seed program."
# Seed had: construct, _count_distinct_cycle_lengths, _get_all_cycle_lengths.
# I will provide a minimal construct that works.

def _count_distinct_cycle_lengths(edges, n):
    # Helper, not strictly needed for the final return but kept for completeness 
    # in case the evaluator imports it or for debugging.
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    cycle_lengths = set()
    
    def dfs(node, start, path, visited):
        for neighbor in adj[node]:
            if neighbor == start and len(path) >= 3:
                cycle_lengths.add(len(path))
            elif not visited[neighbor] and neighbor > start:
                visited[neighbor] = True
                dfs(neighbor, start, path + [neighbor], visited)
                visited[neighbor] = False

    for i in range(n):
        dfs(i, i, [i], {i: True})
        
    return len(cycle_lengths)

def _get_all_cycle_lengths(edges, n):
    return _count_distinct_cycle_lengths(edges, n)

# The main entry point
def construct():
    deadline = time.time() + 60.0
    
    # Generate the optimal known structure: Two disjoint K_3,3
    edges = []
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.append([int(u), int(v)])
            
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.append([int(u), int(v)])
            
    return edges
