"""Path decomposition of three fixed connected graphs (Gallai's conjecture)."""
import time
import random
from collections import defaultdict, deque

def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _hungarian(cost):
    """
    Solve assignment problem using Hungarian algorithm (Kuhn-Munkres).
    cost: n x n matrix. Returns (assignment, total_cost) where assignment[i] = j
    means left node i is matched to right node j.
    """
    n = len(cost)
    if n == 0:
        return [], 0
    
    INF = float('inf')
    # u[i] for left nodes, v[j] for right nodes
    u = [0.0] * n
    v = [0.0] * n
    # match_left[i] = j means left i matched to right j
    # match_right[j] = i means right j matched to left i
    match_left = [-1] * n
    match_right = [-1] * n
    
    # For each left node, try to find augmenting path
    for i in range(n):
        # Initialize
        prev = [-1] * n  # prev[j] = i' means in the tree, j was reached from i'
        used_left = [False] * n
        used_right = [False] * n
        
        # Start from left node i
        queue = deque()
        queue.append(i)
        used_left[i] = True
        prev[i] = -1
        
        while queue:
            curr_left = queue.popleft()
            for j in range(n):
                if used_right[j]:
                    continue
                # Reduced cost
                reduced = cost[curr_left][j] - u[curr_left] - v[j]
                if reduced < 1e-9:
                    # Can go to right node j
                    used_right[j] = True
                    if match_right[j] == -1:
                        # Found augmenting path
                        # Trace back
                        path = []
                        j_curr = j
                        while True:
                            i_curr = prev[j_curr] if prev[j_curr] != -1 else curr_left
                            if i_curr == -1:
                                break
                            path.append((i_curr, j_curr))
                            next_j = match_left[i_curr]
                            if next_j == -1:
                                break
                            j_curr = next_j
                            if used_left[i_curr]:
                                break
                        # Actually let me redo this properly
                        # Standard Hungarian augmenting path
                        break
                    else:
                        # Continue from matched left node
                        next_left = match_right[j]
                        if not used_left[next_left]:
                            used_left[next_left] = True
                            prev[next_left] = j  # reached next_left via right j
                            queue.append(next_left)
            if used_right and any(True for x in used_right):
                # Check if we found a free right node
                break
        
        # Let me implement this more carefully with the standard algorithm
        # Reset and use a cleaner implementation
        pass
    
    # Use a cleaner, well-tested Hungarian implementation
    return _hungarian_clean(cost)

def _hungarian_clean(cost):
    """
    Clean Hungarian algorithm implementation.
    cost: n x n matrix of non-negative costs.
    Returns (assignment, total_cost) where assignment[i] = j.
    """
    n = len(cost)
    if n == 0:
        return [], 0
    
    # u[i], v[j] potentials
    u = [0.0] * n
    v = [0.0] * n
    # p[j] = i means right node j is matched to left node i
    p = [0] * (n + 1)  # 1-indexed, p[j] for j in 1..n
    # way[j] = i means in the augmenting path, right j was reached from left i
    way = [0] * (n + 1)
    
    for i in range(1, n + 1):
        # Augment matching for left node i-1
        p[0] = i
        j0 = 0
        minv = [float('inf')] * (n + 1)
        used = [False] * (n + 1)
        
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float('inf')
            j1 = -1
            
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0 - 1] - v[j - 1]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            
            for j in range(n + 1):
                if used[j]:
                    u[p[j] - 1] += delta
                    v[j - 1] -= delta
                else:
                    minv[j] -= delta
            
            j0 = j1
            if p[j0] == 0:
                break
        
        # Update matching
        while j0 != 0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    
    # p[j] = i means left node i-1 is matched to right node j-1
    assignment = [-1] * n
    total_cost = 0.0
    for j in range(1, n + 1):
        if p[j] != 0:
            i = p[j] - 1
            assignment[i] = j - 1
            total_cost += cost[i][j - 1]
    
    return assignment, total_cost

def _solve_graph(n, p, seed, time_budget=30.0):
    edges = _instance(n, p, seed)
    m = len(edges)
    if m == 0:
        return []
    
    # Build adjacency
    adj = defaultdict(set)
    for (u, v) in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Build bipartite cost matrix for minimum-cost maximum matching
    # Left nodes: 0..n-1, Right nodes: 0..n-1
    # cost[i][j] = 0 if edge (i,j) exists, INF otherwise
    # We want to maximize the number of matched edges, which is equivalent to
    # minimizing the number of unmatched edges.
    # 
    # Key insight: In a connected graph with n vertices and m edges,
    # the minimum number of paths in an edge decomposition is:
    #   n - 1 - (maximum matching size in the bipartite representation)
    # 
    # Actually, the correct formula: min paths = n - 1 - max_matching
    # where max_matching is the size of maximum matching in the bipartite graph
    # with left and right copies of vertices.
    
    INF = 1e9
    cost = [[INF] * n for _ in range(n)]
    for (u, v) in edges:
        cost[u][v] = 0
        cost[v][u] = 0
    
    # Solve minimum-cost maximum matching
    assignment, _ = _hungarian_clean(cost)
    
    # Count matched edges
    matched_edges = set()
    for i in range(n):
        j = assignment[i]
        if j != -1 and cost[i][j] < INF:
            matched_edges.add((min(i, j), max(i, j)))
    
    max_matching = len(matched_edges)
    
    # Minimum number of paths
    min_paths = n - 1 - max_matching
    
    # Now construct the actual paths
    # Build a directed graph from the matching:
    # If left i is matched to right j, add directed edge i -> j
    # This forms a set of directed paths and cycles in the original undirected graph
    # We need to break cycles and extract paths
    
    # Build successor map from matching
    # succ[i] = j means there's a directed edge i -> j
    succ = {}
    for i in range(n):
        j = assignment[i]
        if j != -1 and cost[i][j] < INF:
            succ[i] = j
    
    # Build predecessor map
    pred = {}
    for i, j in succ.items():
        pred[j] = i
    
    # Find all paths: start from vertices with no predecessor
    # and follow successors
    # Also handle cycles by breaking them
    
    # First, identify which vertices are in cycles
    # A vertex is in a cycle if following succ leads back to itself
    
    visited_global = set()
    paths = []
    
    # Find all directed paths (vertices with no incoming matched edge)
    starts = [v for v in range(n) if v not in pred]
    
    for start in starts:
        if start in visited_global:
            continue
        path = [start]
        current = start
        visited_global.add(current)
        while current in succ:
            current = succ[current]
            if current in visited_global:
                # This shouldn't happen for a proper path from a start
                break
            path.append(current)
            visited_global.add(current)
        if len(path) > 1:
            paths.append(path)
    
    # Handle remaining vertices (those in cycles)
    for v in range(n):
        if v in visited_global:
            continue
        # v is in a cycle, break it by removing one edge
        # Find the cycle
        cycle = [v]
        current = v
        while True:
            current = succ[current]
            if current == v:
                break
            cycle.append(current)
        
        # Break the cycle by removing the last edge
        # The path becomes cycle[0] -> cycle[1] -> ... -> cycle[-2]
        # And cycle[-1] becomes a new start
        path = cycle[:-1]  # Remove the last vertex to break the cycle
        # Actually, we want to keep all vertices but break one edge
        # The path is cycle[0], cycle[1], ..., cycle[-2], and cycle[-1] is disconnected
        # So the path is cycle[0] -> cycle[1] -> ... -> cycle[-2]
        paths.append(cycle[:-1])
        visited_global.update(cycle)
    
    # Verify and fix if needed
    # Check that all edges are covered exactly once
    covered = set()
    valid = True
    for path in paths:
        if len(path) < 2:
            valid = False
            break
        if len(set(path)) != len(path):
            valid = False
            break
        for i in range(len(path) - 1):
            a, b = path[i], path[i+1]
            e = (min(a, b), max(a, b))
            if e in covered:
                valid = False
                break
            if e not in set((min(u,v), max(u,v)) for u,v in edges):
                valid = False
                break
            covered.add(e)
        if not valid:
            break
    
    all_edges_set = set((min(u,v), max(u,v)) for u,v in edges)
    
    if not valid or len(covered) != len(all_edges_set):
        # Fallback: use greedy approach
        paths = _greedy_solve(n, edges, time_budget)
    
    return paths

def _greedy_solve(n, edges, time_budget=30.0):
    """Fallback greedy path decomposition."""
    adj = defaultdict(set)
    for (u, v) in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    remaining_edges = set()
    for e in edges:
        remaining_edges.add((min(e[0], e[1]), max(e[0], e[1])))
        
    rem_adj = defaultdict(set)
    for (u, v) in remaining_edges:
        rem_adj[u].add(v)
        rem_adj[v].add(u)
        
    paths = []
    start_time = time.time()
    
    while remaining_edges:
        if time.time() - start_time > time_budget:
            for e in remaining_edges:
                paths.append([e[0], e[1]])
            break
            
        start_v = -1
        max_deg = -1
        for v in range(n):
            if len(rem_adj[v]) > max_deg:
                max_deg = len(rem_adj[v])
                start_v = v
        
        if start_v == -1:
            break
            
        path = [start_v]
        visited_in_path = {start_v}
        current = start_v
        
        while True:
            next_v = -1
            max_next_deg = -1
            for neighbor in rem_adj[current]:
                if neighbor not in visited_in_path:
                    if len(rem_adj[neighbor]) > max_next_deg:
                        max_next_deg = len(rem_adj[neighbor])
                        next_v = neighbor
            
            if next_v == -1:
                break
            
            path.append(next_v)
            visited_in_path.add(next_v)
            
            e = (min(current, next_v), max(current, next_v))
            remaining_edges.discard(e)
            rem_adj[current].discard(next_v)
            rem_adj[next_v].discard(current)
            
            current = next_v
            
            if not any(nb not in visited_in_path for nb in rem_adj[current]):
                break
                
        if len(path) > 1:
            paths.append(path)
            
    return paths

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s kill
    
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            # Not enough time left, use fallback
            edges = _instance(*spec)
            results.append([[u, v] for (u, v) in edges])
        else:
            # Allocate time budget
            remaining_time = deadline - elapsed - 5
            time_budget = max(1.0, min(20.0, remaining_time / 3))
            paths = _solve_graph(*spec, time_budget=time_budget)
            results.append(paths)
    
    return results
