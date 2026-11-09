"""Rewritten program: greedy grow-and-verify with max-flow disjoint path verification."""
# EVOLVE-BLOCK-START
import time
from collections import deque, defaultdict
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _max_flow_disjoint_paths(branches, adj, n):
    """
    Try to find pairwise internally disjoint paths between all pairs of branches.
    Uses a flow network where each non-branch vertex has capacity 1.
    Returns dict {(u,v): path} or None if not possible.
    """
    from collections import deque as dq
    
    # Build flow network
    # Source connects to each branch, each branch connects to sink
    # Non-branch vertices have in/out with capacity 1
    # Edges between vertices have infinite capacity
    # We need to route |branches|*(|branches|-1)/2 units of flow total
    
    num_branches = len(branches)
    num_pairs = num_branches * (num_branches - 1) // 2
    
    if num_pairs == 0:
        return {}
    
    # Node mapping for flow network
    # For each vertex v:
    #   If v is a branch: single node
    #   Else: split into v_in, v_out with capacity 1 edge between them
    # 
    # Actually, let's use a simpler approach:
    # Create a flow network where:
    # - Each non-branch vertex v has an edge v_in -> v_out with capacity 1
    # - Each original edge (u,v) becomes edges u_out -> v_in and v_out -> u_in (capacity inf)
    #   But if u or v is a branch, they don't have in/out split
    # 
    # Simpler: use node-splitting for all non-branch vertices
    # Source -> each branch (capacity inf)
    # Each branch -> sink (capacity inf)  
    # But we need to route specific pairs...
    
    # Alternative approach: iterative path finding with constraint checking
    # For each pair (u,v), find a path avoiding all previously used internal vertices
    
    # Let's try a direct approach: find paths greedily
    used_interior = set()
    paths = {}
    
    # Order pairs by difficulty (shortest path first might be better)
    pair_list = []
    for u, v in combinations(branches, 2):
        pair_list.append((u, v))
    
    # Try to find paths in order
    for u, v in pair_list:
        # BFS for shortest path avoiding used_interior and all branches except u,v
        visited = {u}
        parent = {}
        queue = dq([u])
        found = False
        while queue:
            node = queue.popleft()
            if node == v:
                found = True
                break
            for nb in adj[node]:
                if nb in visited:
                    continue
                if nb in used_interior:
                    continue
                # nb must not be a branch unless it's u or v
                if nb in branches and nb != u and nb != v:
                    continue
                visited.add(nb)
                parent[nb] = node
                queue.append(nb)
        
        if not found:
            return None
        
        # Reconstruct path
        path = [v]
        while path[-1] != u:
            path.append(parent[path[-1]])
        path.reverse()
        
        # Mark interior vertices as used
        for node in path[1:-1]:
            used_interior.add(node)
        
        paths[(u, v)] = path
    
    return paths

def _try_add_vertex(branches, candidate, adj, n):
    """Try adding candidate to branches and verify disjoint paths exist."""
    new_branches = branches + [candidate]
    result = _max_flow_disjoint_paths(new_branches, adj, n)
    if result is not None:
        return new_branches, result
    return None, None

def construct():
    deadline = time.time() + 80
    adj = _build_adj()
    n = P['n']
    
    # Start with two adjacent vertices
    start_u, start_v = P['edges'][0]
    branches = [start_u, start_v]
    paths = {(start_u, start_v): [start_u, start_v]}
    
    # Greedy: try to add vertices one at a time
    # Score candidates by how well-connected they are to existing branches
    while time.time() < deadline - 5:
        best_candidate = None
        best_new_branches = None
        best_paths = None
        
        for candidate in range(n):
            if candidate in branches:
                continue
            if time.time() > deadline - 5:
                break
            
            new_branches, new_paths = _try_add_vertex(branches, candidate, adj, n)
            if new_branches is not None:
                # Prefer candidates that are adjacent to more existing branches
                adj_count = sum(1 for b in branches if candidate in adj[b])
                if best_candidate is None or adj_count > 0:
                    if best_candidate is None or adj_count >= sum(1 for b in branches if best_candidate in adj[b]):
                        best_candidate = candidate
                        best_new_branches = new_branches
                        best_paths = new_paths
        
        if best_candidate is None:
            break
        
        branches = best_new_branches
        paths = best_paths
        
        # Check if we can still add more
        if len(branches) >= n:
            break
    
    # Convert paths to the required format
    path_list = []
    for (u, v) in combinations(branches, 2):
        if (u, v) in paths:
            path_list.append(list(paths[(u, v)]))
        elif (v, u) in paths:
            path_list.append(list(paths[(v, u)]))
        else:
            # Fallback: direct edge or BFS
            if v in adj[u]:
                path_list.append([u, v])
            else:
                visited = {u}
                parent = {}
                queue = deque([u])
                found = False
                while queue:
                    node = queue.popleft()
                    if node == v:
                        found = True
                        break
                    for nb in adj[node]:
                        if nb not in visited:
                            visited.add(nb)
                            parent[nb] = node
                            queue.append(nb)
                if found:
                    p = [v]
                    while p[-1] != u:
                        p.append(parent[p[-1]])
                    p.reverse()
                    path_list.append(p)
                else:
                    path_list.append([u, v])
    
    # Ensure at least 2 branches
    if len(branches) < 2:
        branches = [start_u, start_v]
        path_list = [[start_u, start_v]]
    
    branches = sorted(branches)
    
    return {'branches': branches, 'paths': path_list}
# EVOLVE-BLOCK-END
