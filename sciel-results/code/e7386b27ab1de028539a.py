"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import deque

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
    Use max-flow to verify and extract pairwise internally vertex-disjoint paths
    between all pairs of branch vertices.
    
    Returns dict with 'branches' and 'paths' on success, None on failure.
    """
    k = len(branches)
    if k < 2:
        return None
    
    pairs = list(combinations(range(k), 2))
    num_pairs = len(pairs)
    
    # We need to find num_pairs internally vertex-disjoint paths
    # Each path connects a pair of branch vertices
    
    # Build a flow network:
    # - For each non-branch vertex v, split into v_in -> v_out with capacity 1
    # - For each branch vertex b, it can be used as endpoint (no capacity limit on being endpoint)
    # - For each edge (u,v) in original graph, add edges u_out -> v_in and v_out -> u_in with capacity 1
    # - Source connects to branch vertices (as starting points), sink from branch vertices (as ending points)
    #
    # Actually, the standard approach for disjoint paths between multiple pairs:
    # Create a super source S and super sink T.
    # For each pair (i,j), we need a path from branches[i] to branches[j].
    # 
    # Better approach: model as a flow problem where we send 1 unit of flow for each pair.
    # But standard max-flow doesn't handle multiple commodities.
    #
    # Alternative: Use the fact that we want k(k-1)/2 disjoint paths.
    # We can model this as: create a node for each branch vertex.
    # For internal vertices, split them with capacity 1.
    # Then find if there exist disjoint paths.
    #
    # Actually, let me use a different approach:
    # 1. For each pair, find candidate paths using BFS
    # 2. Use backtracking to assign paths ensuring interior disjointness
    # 3. Use max-flow as a quick check for feasibility
    
    # Quick feasibility check using max-flow:
    # Create a network where each non-branch vertex has capacity 1
    # Source -> each branch vertex (capacity = k-1, since each branch connects to k-1 others)
    # Each branch vertex -> sink (capacity = k-1)
    # Edges between vertices with capacity 1 (for non-branch) or infinity (for branch)
    
    # Actually, the cleanest approach: 
    # Split each non-branch vertex v into v_in and v_out with edge (v_in, v_out) capacity 1
    # For branch vertices, no splitting needed (they can be used multiple times as endpoints)
    # For each original edge (u, v):
    #   - If u is branch and v is non-branch: u -> v_in, capacity 1
    #   - If u is non-branch and v is branch: u_out -> v, capacity 1  
    #   - If both non-branch: u_out -> v_in, capacity 1
    #   - If both branch: u -> v, capacity 1 (but these are direct edges, handled separately)
    #
    # Source S connects to each branch vertex with capacity k-1
    # Each branch vertex connects to sink T with capacity k-1
    #
    # If max flow = k*(k-1)/2 * 2 = k*(k-1), then we have enough capacity.
    # Wait, that's not right either. Let me think more carefully.
    
    # The issue is that max-flow with a single source/sink doesn't capture the pairing constraint.
    # We need paths from specific source to specific sink pairs.
    #
    # Let me use a different strategy:
    # 1. Find candidate paths for each pair using BFS (multiple candidates)
    # 2. Use backtracking with constraint propagation to find a valid assignment
    # 3. Use flow-based pruning to detect infeasibility early
    
    branch_set = set(branches)
    
    # For each pair, find up to 10 candidate shortest paths
    candidate_paths = {}
    for (i, j) in pairs:
        u, v = branches[i], branches[j]
        # BFS to find multiple shortest paths
        from collections import deque as _deque
        queue = _deque([(u, [u])])
        visited = {u}
        found_paths = []
        # Limit search to avoid explosion
        max_search_nodes = 500
        search_count = 0
        
        while queue and len(found_paths) < 10 and search_count < max_search_nodes:
            node, path = queue.popleft()
            search_count += 1
            if node == v:
                found_paths.append(path)
                continue
            for neighbor in sorted(adj[node]):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        if not found_paths:
            return None
        
        # Sort by length, then by number of interior vertices
        found_paths.sort(key=lambda p: (len(p), sum(1 for x in p[1:-1] if x not in branch_set)))
        candidate_paths[(i, j)] = found_paths
    
    # Backtracking with constraint propagation
    # Order pairs by number of candidates (most constrained first)
    pair_order = sorted(range(num_pairs), key=lambda idx: len(candidate_paths[pairs[idx]]))
    
    def backtrack(pos, used_interiors, assignment):
        if pos == num_pairs:
            return True
        
        idx = pair_order[pos]
        pair = pairs[idx]
        candidates = candidate_paths[pair]
        
        # Try candidates, preferring shorter ones
        for path in candidates:
            interior = set(path[1:-1])
            # Check interior disjointness
            if interior & used_interiors:
                continue
            # Check no branch vertex in interior
            if interior & branch_set:
                continue
            
            # Try this path
            assignment[idx] = path
            new_used = used_interiors | interior
            if backtrack(pos + 1, new_used, assignment):
                return True
            
            del assignment[idx]
        
        return False
    
    assignment = {}
    if not backtrack(0, set(), assignment):
        return None
    
    # Build result
    paths_list = []
    for idx in range(num_pairs):
        paths_list.append(list(assignment[idx]))
    
    return {'branches': list(branches), 'paths': paths_list}

def _find_max_clique(adj, n):
    """Find maximum clique using Bron-Kerbosch with pivot."""
    best_clique = []
    
    def bronk(R, P, X):
        nonlocal best_clique
        if not P and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        
        if P:
            u = max(P, key=lambda v: len(adj[v] & P) + len(adj[v] & X))
            candidates = [v for v in P if v not in adj[u]]
        else:
            candidates = list(P)
        
        for v in candidates:
            bronk(R | {v}, P & adj[v], X & adj[v])
            P.discard(v)
            X.add(v)
    
    bronk(set(), set(range(n)), set())
    return best_clique

def _try_extend(branches, adj, n, vertices_by_degree, deadline):
    """Try to extend the branch set by adding one vertex at a time."""
    current = list(branches)
    improved = True
    while improved:
        improved = False
        for v in vertices_by_degree:
            if v in current:
                continue
            if time.time() > deadline:
                return current
            new_branches = sorted(current + [v])
            result = _max_flow_disjoint_paths(new_branches, adj, n)
            if result is not None:
                current = new_branches
                improved = True
                break
    return current

def _local_search(branches, adj, n, vertices_by_degree, deadline):
    """Local search: swap one branch vertex for another."""
    current = list(branches)
    improved = True
    while improved:
        improved = False
        for i in range(len(current)):
            if time.time() > deadline:
                break
            for v in vertices_by_degree:
                if v in current:
                    continue
                new_branches = current[:i] + current[i+1:] + [v]
                new_branches = sorted(new_branches)
                result = _max_flow_disjoint_paths(new_branches, adj, n)
                if result is not None:
                    # Check if this leads to a better solution
                    # For now, just accept if valid
                    # We could also check if we can extend further
                    current = new_branches
                    improved = True
                    break
            if improved:
                break
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 75  # Leave 15s margin for safety
    
    n = P['n']
    adj = _build_adj()
    
    vertices_by_degree = sorted(range(n), key=lambda v: -len(adj[v]))
    
    best_result = None
    best_k = 0
    
    # Strategy 1: Find maximum clique
    clique = _find_max_clique(adj, n)
    if len(clique) >= 2:
        result = _max_flow_disjoint_paths(clique, adj, n)
        if result is not None:
            best_result = result
            best_k = len(clique)
    
    # Strategy 2: Try to extend beyond clique by adding vertices
    if best_result is not None:
        extended = _try_extend(best_result['branches'], adj, n, vertices_by_degree, deadline)
        if len(extended) > best_k:
            result = _max_flow_disjoint_paths(extended, adj, n)
            if result is not None:
                best_result = result
                best_k = len(extended)
    
    # Strategy 3: Local search with swaps
    if best_result is not None and time.time() < deadline - 5:
        searched = _local_search(best_result['branches'], adj, n, vertices_by_degree, deadline)
        if len(searched) > best_k:
            result = _max_flow_disjoint_paths(searched, adj, n)
            if result is not None:
                best_result = result
                best_k = len(searched)
    
    # Strategy 4: Try random subsets from high-degree vertices
    if time.time() < deadline - 10:
        random.seed(42)
        for k in range(max(best_k + 1, 4), min(n + 1, 10)):
            for attempt in range(30):
                if time.time() > deadline:
                    break
                top_vertices = vertices_by_degree[:min(18, n)]
                if len(top_vertices) < k:
                    break
                subset = random.sample(top_vertices, k)
                subset = sorted(subset)
                result = _max_flow_disjoint_paths(subset, adj, n)
                if result is not None:
                    best_result = result
                    best_k = k
                    break
    
    # Strategy 5: Try all subsets of size best_k+1 from top vertices
    if time.time() < deadline - 5:
        target_k = best_k + 1
        top_vertices = vertices_by_degree[:min(15, n)]
        if len(top_vertices) >= target_k:
            for subset in combinations(top_vertices, target_k):
                if time.time() > deadline:
                    break
                subset_list = sorted(list(subset))
                result = _max_flow_disjoint_paths(subset_list, adj, n)
                if result is not None:
                    best_result = result
                    best_k = target_k
                    break
    
    # Fallback: find any edge
    if best_result is None:
        u, v = P['edges'][0]
        best_result = {'branches': [u, v], 'paths': [[u, v]]}
    
    # Ensure types are correct: integers must be Python int, sequences must be list
    best_result['branches'] = [int(v) for v in best_result['branches']]
    best_result['paths'] = [[int(v) for v in p] for p in best_result['paths']]
    
    return best_result

# EVOLVE-BLOCK-END
