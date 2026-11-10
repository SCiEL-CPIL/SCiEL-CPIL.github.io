"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _max_clique_correct(n, adj, deadline):
    """Correct Tomita MCQ algorithm for maximum clique."""
    best = []
    best_size = 0

    def color_sort(P_list):
        """Greedy coloring to get upper bounds."""
        if not P_list:
            return [], []
        # Order vertices by degree in P_list (descending)
        P_set = set(P_list)
        deg = [len(adj[v] & P_set) for v in P_list]
        order = list(range(len(P_list)))
        order.sort(key=lambda i: -deg[i])
        
        colors = [0] * len(P_list)
        # Greedy coloring in the given order
        for idx in order:
            v = P_list[idx]
            used = set()
            for j in order:
                if j >= idx:
                    break
                u = P_list[j]
                if u in adj[v] and colors[j] > 0:
                    used.add(colors[j])
            c = 1
            while c in used:
                c += 1
            colors[idx] = c
        
        # Sort by color ascending
        order.sort(key=lambda i: colors[i])
        return order, colors

    def expand(clique, P_list):
        nonlocal best, best_size
        if time.time() > deadline:
            return
        if len(P_list) == 0:
            if len(clique) > best_size:
                best_size = len(clique)
                best = list(clique)
            return
        
        order, colors = color_sort(P_list)
        
        i = len(order) - 1
        while i >= 0:
            if time.time() > deadline:
                return
            v_idx = order[i]
            v = P_list[v_idx]
            if len(clique) + colors[v_idx] <= best_size:
                return
            clique.append(v)
            new_P = [P_list[order[j]] for j in range(i) if P_list[order[j]] in adj[v]]
            expand(clique, new_P)
            clique.pop()
            i -= 1

    expand([], list(range(n)))
    return best


def _find_disjoint_paths(n, adj, branches, deadline):
    """
    Find internally disjoint paths for all pairs of branch vertices.
    Returns list of paths or None if not possible.
    Uses a greedy approach with BFS avoiding used vertices.
    """
    if len(branches) < 2:
        return []
    
    branch_set = set(branches)
    # We need C(k,2) paths with disjoint interiors
    # Strategy: process pairs in order, for each pair find a path avoiding
    # all branch vertices and all previously used interior vertices
    
    used_interior = set()
    paths = []
    
    # Sort pairs to prioritize shorter potential paths
    pairs = list(combinations(branches, 2))
    # Sort by whether they're adjacent (adjacent first, then by shortest path)
    def pair_key(p):
        u, v = p
        if v in adj[u]:
            return (0, 0)
        return (1, 0)  # non-adjacent
    
    pairs.sort(key=pair_key)
    
    for u, v in pairs:
        if v in adj[u]:
            # Direct edge, no interior
            paths.append([u, v])
            continue
        # BFS avoiding branch_set and used_interior
        visited = {u}
        parent = {u: None}
        q = deque([u])
        found = False
        while q:
            cur = q.popleft()
            if cur == v:
                found = True
                break
            for nb in sorted(adj[cur]):  # deterministic
                if nb in branch_set and nb != v:
                    continue
                if nb in used_interior:
                    continue
                if nb not in visited:
                    visited.add(nb)
                    parent[nb] = cur
                    q.append(nb)
        if not found:
            return None
        # Reconstruct path
        path = []
        cur = v
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        # Mark interior vertices as used
        for vertex in path[1:-1]:
            used_interior.add(vertex)
        paths.append(path)
    
    return paths


def construct():
    n = P['n']
    adj = _build_adj(n, P['edges'])
    deadline = time.time() + 85  # safety margin
    
    # Find maximum clique
    clique = _max_clique_correct(n, adj, deadline)
    
    # Ensure at least 2 branches
    if len(clique) < 2:
        u, v = P['edges'][0]
        clique = [u, v]
    
    # Try to extend the clique with non-adjacent vertices
    # A vertex w can be added if for every existing branch b,
    # there's a path from w to b avoiding all branches and used interiors
    # This is complex, so let's first verify the clique works
    
    # For a clique, all pairs are adjacent, so paths are just edges
    # This is always valid
    branches = list(clique)
    paths = []
    for u, v in combinations(branches, 2):
        if v in adj[u]:
            paths.append([u, v])
        else:
            # Should not happen in a clique
            paths.append([u, v])  # fallback
    
    # Try to add more vertices beyond the clique
    # Greedy: try each non-clique vertex, see if we can maintain validity
    remaining = [v for v in range(n) if v not in set(branches)]
    remaining.sort(key=lambda v: -len(adj[v]))  # high degree first
    
    for w in remaining:
        if time.time() > deadline:
            break
        # Try adding w to branches
        new_branches = branches + [w]
        # Check if we can find disjoint paths for all new pairs
        # New pairs are (w, b) for each b in branches
        # Existing pairs already have paths; we need to check if
        # adding w's paths conflicts
        
        # For simplicity: check if w is adjacent to all existing branches
        # If so, we can just use direct edges for new pairs
        all_adj = all(b in adj[w] for b in branches)
        if all_adj:
            # Add w and its direct-edge paths
            branches = new_branches
            for b in list(branches)[:-1]:
                paths.append([b, w])
        else:
            # Try to find a path from w to each existing branch
            # that avoids all branches and previously used interiors
            # This is complex; skip for now
            pass
    
    # Verify the solution
    branch_set = set(branches)
    all_interiors = set()
    valid = True
    for p in paths:
        if len(p) < 2:
            valid = False
            break
        if p[0] not in branch_set or p[-1] not in branch_set:
            valid = False
            break
        # Check adjacency
        for i in range(len(p) - 1):
            if p[i+1] not in adj[p[i]]:
                valid = False
                break
        if not valid:
            break
        # Check interiors are disjoint and avoid branches
        for vertex in p[1:-1]:
            if vertex in branch_set:
                valid = False
                break
            if vertex in all_interiors:
                valid = False
                break
            all_interiors.add(vertex)
        if not valid:
            break
    
    if not valid or len(branches) < 2:
        # Fallback: use just a clique of size 2
        u, v = P['edges'][0]
        branches = [u, v]
        paths = [[u, v]]
    
    # Convert to proper Python types
    branches = [int(x) for x in branches]
    paths_out = [[int(x) for x in p] for p in paths]
    
    return {'branches': branches, 'paths': paths_out}
# EVOLVE-BLOCK-END
