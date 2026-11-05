import time
import random
from collections import defaultdict
from math import inf

def instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _hopcroft_karp(adj, n_left, n_right):
    """
    Hopcroft-Karp algorithm for maximum bipartite matching.
    adj: dict mapping left node -> list of right nodes
    n_left, n_right: number of nodes on each side
    Returns: list of (left, right) pairs in matching
    """
    pair_u = [-1] * n_left
    pair_v = [-1] * n_right
    dist = [0] * n_left
    
    def bfs():
        queue = []
        for u in range(n_left):
            if pair_u[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = inf
        found = False
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                pu = pair_v[v]
                if pu == -1:
                    found = True
                elif dist[pu] == inf:
                    dist[pu] = dist[u] + 1
                    queue.append(pu)
        return found

    def dfs(u):
        for v in adj[u]:
            pu = pair_v[v]
            if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = inf
        return False

    matching = 0
    while bfs():
        for u in range(n_left):
            if pair_u[u] == -1:
                if dfs(u):
                    matching += 1
    
    result = []
    for u in range(n_left):
        if pair_u[u] != -1:
            result.append((u, pair_u[u]))
    return result

def _initial_decompose(edges):
    """
    Greedy decomposition to get an initial set of paths.
    """
    if not edges:
        return []
    
    remaining = set(edges)
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    paths = []
    
    while remaining:
        # Pick an edge to start with
        # Heuristic: pick edge with highest sum of degrees
        best_edge = None
        best_score = -1
        for u, v in sorted(remaining):
            score = len(adj[u]) + len(adj[v])
            if score > best_score:
                best_score = score
                best_edge = (u, v)
        
        u, v = best_edge
        remaining.discard(best_edge)
        adj[u].discard(v)
        adj[v].discard(u)
        
        path = [u, v]
        path_set = {u, v}
        
        # Extend greedily
        extended = True
        while extended:
            extended = False
            for side in ['front', 'back']:
                curr = path[0] if side == 'front' else path[-1]
                best_next = None
                best_deg = -1
                for w in sorted(adj[curr]):
                    if w in path_set:
                        continue
                    deg = len(adj[w])
                    if deg > best_deg:
                        best_deg = deg
                        best_next = w
                
                if best_next is not None:
                    e = tuple(sorted((curr, best_next)))
                    if e in remaining:
                        remaining.discard(e)
                        adj[curr].discard(best_next)
                        adj[best_next].discard(curr)
                        if side == 'front':
                            path = [best_next] + path
                        else:
                            path = path + [best_next]
                        path_set.add(best_next)
                        extended = True
        
        paths.append(path)
    
    return paths

def decompose(edges):
    if not edges:
        return []
    
    graph_edges = set(edges)
    # 1. Initial greedy decomposition
    paths = _initial_decompose(edges)
    
    if len(paths) <= 1:
        return paths

    # 2. Iterative Matching-based Merging
    # We will repeatedly try to merge paths using max bipartite matching.
    # Left side: Path starts (index 0)
    # Right side: Path ends (index -1)
    # Edge (i, j) exists if:
    #   1. i != j
    #   2. Edge (paths[i][0], paths[j][-1]) is in graph_edges
    #   3. The edge is NOT already used in any path
    #   4. The merged path is simple (no repeated vertices)
    
    # To efficiently check "edge is not already used", we maintain a set of used edges.
    used_edges = set()
    for p in paths:
        for k in range(len(p) - 1):
            used_edges.add(tuple(sorted((p[k], p[k+1]))))
    
    max_iterations = 100
    for _ in range(max_iterations):
        n = len(paths)
        if n <= 1:
            break
        
        # Build auxiliary graph
        # Left nodes: 0..n-1 (represents path i start)
        # Right nodes: 0..n-1 (represents path j end)
        # We want to match start i to end j to form a new path: paths[i] + paths[j]
        # Wait, connecting start of i to end of j?
        # Path i: s_i ... e_i
        # Path j: s_j ... e_j
        # If we connect e_j to s_i, we get s_j ... e_j, s_i ... e_i.
        # This requires edge (e_j, s_i) to exist.
        # So Left = End of path j, Right = Start of path i?
        # Or Left = Start of path i, Right = End of path j?
        # Let's say Left node i represents "I am the start of path i".
        # Right node j represents "I am the end of path j".
        # Edge (i, j) means "Connect end of path j to start of path i".
        # Resulting path: paths[j] + paths[i] (concatenated, skipping duplicate vertex).
        
        adj_aux = defaultdict(list)
        
        # Precompute vertex sets for simplicity check
        path_sets = [set(p) for p in paths]
        
        for i in range(n):
            start_i = paths[i][0]
            set_i = path_sets[i]
            for j in range(n):
                if i == j:
                    continue
                end_j = paths[j][-1]
                set_j = path_sets[j]
                
                # Check edge existence
                e = tuple(sorted((start_i, end_j)))
                if e not in graph_edges:
                    continue
                if e in used_edges:
                    continue
                
                # Check simplicity
                # Merged path will contain vertices from path_i and path_j.
                # They share the connection vertex? No, we connect end_j to start_i.
                # The new path is paths[j] + paths[i].
                # The vertices are set_j U set_i.
                # For the path to be simple, set_j and set_i must be disjoint?
                # Actually, if they share ANY vertex other than the connection point, it's not simple?
                # Wait.
                # Path j: v1...vk (end is vk)
                # Path i: u1...um (start is u1)
                # Connect vk to u1.
                # New path: v1...vk, u1...um.
                # Vertices are {v1..vk} U {u1..um}.
                # Since vk is connected to u1, and vk != u1 (unless self loop, which is not allowed in simple graph edge list usually, but even if vk==u1, that would mean they share a vertex).
                # If set_j and set_i are not disjoint, there is a repeated vertex.
                # So we require set_j.isdisjoint(set_i).
                
                if not set_j.isdisjoint(set_i):
                    continue
                
                adj_aux[i].append(j)
        
        if not adj_aux:
            break
            
        # Find max matching
        # Left size n, Right size n
        matches = _hopcroft_karp(adj_aux, n, n)
        
        if not matches:
            break
        
        # Apply matches
        # matches is list of (i, j) where i is left (start of i), j is right (end of j)
        # This means we merge path j and path i into path j + path i.
        # We need to be careful: if we merge (0, 1) and (1, 2), we can't do both if they share path 1.
        # Hopcroft-Karp gives a matching, so each left node appears at most once, and each right node appears at most once.
        # Left node i is used at most once. Right node j is used at most once.
        # This means path i can be a "start" in at most one merge, and path j can be an "end" in at most one merge.
        # Can path k be both a start in one merge and an end in another?
        # Yes, if k is matched on Left in one pair and on Right in another.
        # Example: (k, a) and (b, k).
        # Merge 1: path a + path k.
        # Merge 2: path k + path b.
        # This forms a chain: path a + path k + path b.
        # This is valid!
        
        # We need to resolve the chains.
        # Let's create a mapping of merges.
        # merge_map: path_index -> (predecessor, successor)
        # If (i, j) is in matches:
        #   i is the "start" path, j is the "end" path.
        #   So j is predecessor of i.
        #   pred[i] = j
        
        pred = {}
        succ = {}
        for i, j in matches:
            pred[i] = j
            succ[j] = i
            
        # Find all chain heads (paths that are not a successor, i.e., not in succ.values())
        # And traverse to find full chains.
        
        # Actually, simpler:
        # Identify all paths involved in matches.
        # A path is involved if it is a key in pred or a key in succ.
        # We want to group them into new paths.
        
        # Let's build the new list of paths.
        # Paths not involved in any match remain unchanged.
        # Paths involved form chains.
        
        # Let's identify the "roots" of the chains.
        # A root is a path that has a successor (it is an end of a merge) but no predecessor (it is not a start of a merge).
        # Wait, j is end, i is start.
        # Chain: ... -> j -> i -> ...
        # j has succ i. i has pred j.
        # Head of chain: path with no pred.
        # Tail of chain: path with no succ.
        
        # Let's find all heads.
        heads = []
        for p in range(n):
            if p not in pred:
                # Check if it's part of a chain (has succ)
                if p in succ:
                    heads.append(p)
        
        # If a path is isolated (no pred, no succ), it's not in heads, and not merged.
        
        new_paths = []
        merged_indices = set()
        
        for head in heads:
            # Traverse chain
            current = head
            chain = [current]
            while current in succ:
                current = succ[current]
                chain.append(current)
            
            # Build merged path
            # chain = [h, n1, n2, ...]
            # Merged = paths[h] + paths[n1] + paths[n2] ...
            merged = []
            for idx in chain:
                if not merged:
                    merged = list(paths[idx])
                else:
                    # Append, skipping the first vertex of paths[idx] which is the same as last of merged
                    # Actually, the connection is end of prev to start of curr.
                    # So we append paths[idx][1:]
                    merged.extend(paths[idx][1:])
            
            new_paths.append(merged)
            merged_indices.update(chain)
        
        # Add paths not involved in merges
        for p in range(n):
            if p not in merged_indices:
                new_paths.append(paths[p])
        
        # Update used edges
        # We added edges (end_j, start_i) for each match (i, j).
        for i, j in matches:
            e = tuple(sorted((paths[i][0], paths[j][-1])))
            used_edges.add(e)
        
        paths = new_paths
        
        if len(paths) <= 1:
            break

    return paths

def construct():
    start = time.time()
    deadline = start + 140
    results = []
    
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)
        
        if time.time() > deadline - 5:
            while len(results) < 3:
                results.append([])
            break
    
    while len(results) < 3:
        results.append([])
    
    return results
