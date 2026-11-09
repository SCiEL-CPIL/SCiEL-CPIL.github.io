"""Combined: Bron-Kerbosch clique finder + max-flow disjoint path verification."""
# EVOLVE-BLOCK-START
import time
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

def _degeneracy_order(adj):
    n = P['n']
    remaining = list(range(n))
    remaining_set = set(remaining)
    order = []
    while remaining:
        min_deg = min(len(adj[v] & remaining_set) for v in remaining)
        v = next(v for v in remaining if len(adj[v] & remaining_set) == min_deg)
        order.append(v)
        remaining.remove(v)
        remaining_set.discard(v)
    return order

def _bron_kerbosch(adj, R, P_set, X, best_clique, deadline):
    if time.time() > deadline:
        return best_clique
    if not P_set and not X:
        if len(R) > len(best_clique):
            best_clique = list(R)
        return best_clique
    if len(R) + len(P_set) <= len(best_clique):
        return best_clique
    PX = P_set | X
    if PX:
        pivot = max(PX, key=lambda v: len(adj[v] & P_set))
        candidates = list(P_set - adj[pivot])
    else:
        candidates = list(P_set)
    for v in candidates:
        if time.time() > deadline:
            return best_clique
        new_R = R | {v}
        new_P = P_set & adj[v]
        new_X = X & adj[v]
        best_clique = _bron_kerbosch(adj, new_R, new_P, new_X, best_clique, deadline)
        P_set.discard(v)
        X.add(v)
    return best_clique

def _max_clique(adj, deadline):
    n = P['n']
    order = _degeneracy_order(adj)
    best = []
    for i, v in enumerate(order):
        if time.time() > deadline:
            break
        P_set = set(w for w in adj[v] if w in order[i+1:])
        if not P_set:
            continue
        X = set(w for w in adj[v] if w in order[:i])
        best = _bron_kerbosch(adj, {v}, P_set, X, best, deadline)
    if not best:
        best = _bron_kerbosch(adj, set(), set(range(n)), set(), best, deadline)
    return best

def _max_flow_disjoint_paths(adj, branches, deadline):
    """
    Try to find k(k-1)/2 pairwise internally vertex-disjoint paths
    connecting all pairs of branch vertices.
    Returns (success, paths_dict) where paths_dict maps (u,v) to path list.
    """
    n = P['n']
    k = len(branches)
    if k < 2:
        return False, {}
    required = k * (k - 1) // 2
    
    # Build flow network with vertex splitting
    # Each vertex v becomes v_in -> v_out with capacity 1 (for non-branch) or infinity (for branch)
    # Branch vertices have infinite capacity to allow multiple paths through them
    # Edges go from u_out to v_in with capacity 1
    
    # We'll use a super source and super sink approach
    # For each pair (u,v) of branches, we need one unit of flow from u to v
    # But paths must be internally disjoint, so we use vertex capacities
    
    # Network construction:
    # Node indices: for each vertex v, we have 2*v (in) and 2*v+1 (out)
    # Plus super source SS and super sink TT
    SS = 2 * n
    TT = 2 * n + 1
    total_nodes = 2 * n + 2
    
    # Adjacency for flow network: list of [to, capacity, reverse_index]
    flow_adj = [[] for _ in range(total_nodes)]
    
    def add_edge(u, v, cap):
        flow_adj[u].append([v, cap, len(flow_adj[v])])
        flow_adj[v].append([u, 0, len(flow_adj[u]) - 1])
    
    # Vertex splitting edges
    for v in range(n):
        if v in branches:
            add_edge(2*v, 2*v+1, 10**9)  # infinite capacity for branches
        else:
            add_edge(2*v, 2*v+1, 1)  # capacity 1 for non-branches
    
    # Original graph edges
    for u in range(n):
        for v in adj[u]:
            if u < v:  # avoid duplicate
                add_edge(2*u+1, 2*v, 1)
                add_edge(2*v+1, 2*u, 1)
    
    # For each pair of branches, add source/sink connections
    # We need to route flow for each pair independently but with shared vertex capacities
    # This is a multi-commodity flow problem, which is hard.
    # Instead, let's try a different approach: use the fact that we can check
    # if there exist disjoint paths by trying to assign paths greedily with backtracking,
    # or use a simpler verification.
    
    # Alternative: For a clique, paths are just edges. For larger sets, we need
    # to find disjoint paths. Let's use a simpler approach:
    # Try to find paths using BFS with used-vertex tracking.
    
    # Sort branch pairs by "difficulty" (fewest available paths first)
    branch_set = set(branches)
    pairs = list(combinations(sorted(branches), 2))
    
    # For each pair, find candidate paths
    pair_paths = {}
    for u, v in pairs:
        # BFS from u to v avoiding other branches in interior
        found = []
        visited = {u}
        queue = deque([(u, [u])])
        while queue and len(found) < 50:  # limit paths found
            node, path = queue.popleft()
            if node == v:
                found.append(path)
                continue
            for nb in adj[node]:
                if nb not in visited:
                    # If nb is a branch other than u or v, skip (can't be interior)
                    if nb in branch_set and nb != u and nb != v:
                        continue
                    visited.add(nb)
                    queue.append((nb, path + [nb]))
        pair_paths[(u, v)] = found
    
    # Check if we can assign disjoint paths
    # Greedy: sort pairs by number of available paths (ascending)
    pairs_sorted = sorted(pairs, key=lambda p: len(pair_paths[p]))
    
    used_interior = set()
    assignment = {}
    
    for u, v in pairs_sorted:
        assigned = False
        for path in pair_paths[(u, v)]:
            interior = set(path[1:-1])
            if interior.isdisjoint(used_interior):
                assignment[(u, v)] = path
                used_interior.update(interior)
                assigned = True
                break
        if not assigned:
            return False, {}
    
    return True, assignment

def _try_add_vertex(adj, branches, candidate, deadline):
    """Try to add candidate to branches and verify disjoint paths exist."""
    new_branches = branches + [candidate]
    success, _ = _max_flow_disjoint_paths(adj, new_branches, deadline)
    return success

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    n = P['n']
    adj = _build_adj()
    
    # Step 1: Find maximum clique
    clique = _max_clique(adj, deadline)
    if len(clique) < 2:
        u, v = P['edges'][0]
        clique = [u, v]
    
    branches = sorted(clique)
    
    # Verify the clique has valid disjoint paths (should be trivial - just edges)
    success, paths_dict = _max_flow_disjoint_paths(adj, branches, deadline)
    if not success:
        # Fallback to just an edge
        u, v = P['edges'][0]
        branches = [u, v]
        success, paths_dict = _max_flow_disjoint_paths(adj, branches, deadline)
    
    # Step 2: Try to grow the branch set by adding non-branch vertices
    branch_set = set(branches)
    candidates = [v for v in range(n) if v not in branch_set]
    
    # Sort candidates by degree (higher degree first - more likely to help)
    candidates.sort(key=lambda v: -len(adj[v]))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for cand in candidates[:]:
            if cand in branch_set:
                continue
            if time.time() > deadline:
                break
            if _try_add_vertex(adj, branches, cand, deadline):
                branches.append(cand)
                branch_set.add(cand)
                candidates.remove(cand)
                improved = True
                # Re-verify paths with new branch set
                success, paths_dict = _max_flow_disjoint_paths(adj, branches, deadline)
                if not success:
                    # Roll back
                    branches.remove(cand)
                    branch_set.discard(cand)
                    candidates.append(cand)
                    break
    
    # Final path extraction
    success, paths_dict = _max_flow_disjoint_paths(adj, branches, deadline)
    if not success:
        # Fallback: use clique paths
        branches = sorted(clique)
        success, paths_dict = _max_flow_disjoint_paths(adj, branches, deadline)
        if not success:
            u, v = P['edges'][0]
            branches = [u, v]
            paths_dict = {(u, v): [u, v]}
    
    # Build output
    paths = []
    for u, v in combinations(sorted(branches), 2):
        if (u, v) in paths_dict:
            paths.append(list(paths_dict[(u, v)]))
        elif (v, u) in paths_dict:
            paths.append(list(paths_dict[(v, u)]))
        else:
            # Should not happen, but fallback
            paths.append([u, v])
    
    return {'branches': branches, 'paths': paths}

# EVOLVE-BLOCK-END
