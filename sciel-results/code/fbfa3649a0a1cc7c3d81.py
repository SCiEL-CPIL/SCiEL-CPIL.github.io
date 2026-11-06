"""Decompose fixed random graphs into few edge-disjoint cycles and single edges."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _hopcroft_karp(left, right, adj):
    """
    Maximum bipartite matching.
    left: list of left vertices
    right: list of right vertices
    adj: dict left_vertex -> list of right vertices
    Returns matching dict: left_vertex -> right_vertex
    """
    if not left:
        return {}
    # Map right vertices to indices for array access
    right_to_idx = {r: i for i, r in enumerate(right)}
    left_to_idx = {l: i for i, l in enumerate(left)}
    
    # adj_idx: for each left index, list of right indices
    adj_idx = defaultdict(list)
    for l in left:
        for r in adj.get(l, []):
            if r in right_to_idx:
                adj_idx[left_to_idx[l]].append(right_to_idx[r])
    
    match_l = [-1] * len(left)   # match_l[i] = right index matched to left i, or -1
    match_r = [-1] * len(right)  # match_r[j] = left index matched to right j, or -1
    
    def bfs():
        # Returns distance array dist for left vertices, and whether an augmenting path exists
        dist = [-1] * len(left)
        queue = deque()
        for i in range(len(left)):
            if match_l[i] == -1:
                dist[i] = 0
                queue.append(i)
        found = False
        while queue:
            u = queue.popleft()
            for v in adj_idx[u]:
                mu = match_r[v]
                if mu == -1:
                    found = True
                elif dist[mu] == -1:
                    dist[mu] = dist[u] + 1
                    queue.append(mu)
        return dist, found
    
    def dfs(u, dist, visited):
        for v in adj_idx[u]:
            mu = match_r[v]
            if mu == -1 or (dist[mu] == dist[u] + 1 and dfs(mu, dist, visited)):
                match_l[u] = v
                match_r[v] = u
                return True
        dist[u] = -1
        return False
    
    while True:
        dist, found = bfs()
        if not found:
            break
        for i in range(len(left)):
            if match_l[i] == -1:
                dfs(i, dist, set())
    
    # Convert back to vertex matching
    matching = {}
    for i, l in enumerate(left):
        if match_l[i] != -1:
            matching[l] = right[match_l[i]]
    return matching

def _decompose(edges):
    """
    Decompose edge set into edge-disjoint cycles and single edges.
    Strategy:
      1. Build directed line graph where each undirected edge is two directed arcs.
      2. Find maximum matching in the bipartite graph of arcs (tail -> head continuity).
      3. Extract cycles from the matching.
      4. Remaining edges are single pieces.
    """
    if not edges:
        return []
    
    # Map each undirected edge to two directed arcs
    # arc_id for (u,v) direction
    arc_to_edge = {}
    edge_to_arcs = {}
    arc_id = 0
    for (u, v) in edges:
        a1 = (u, v)
        a2 = (v, u)
        arc_to_edge[arc_id] = (u, v)
        edge_to_arcs[(u, v)] = arc_id
        arc_id += 1
        arc_to_edge[arc_id] = (u, v)
        edge_to_arcs[(v, u)] = arc_id
        arc_id += 1
    
    # Build adjacency for matching: left arc -> right arcs it can connect to
    # Arc (u,v) can connect to arc (v,w) if w != u
    # So left vertex is arc (u,v), right vertex is arc (v,w)
    adj = defaultdict(list)
    # Group arcs by their head vertex
    arcs_by_head = defaultdict(list)
    for aid in range(arc_id):
        u, v = arc_to_edge[aid]
        arcs_by_head[v].append(aid)
    
    # For each arc (u,v) as left, it can connect to any arc (v,w) where w != u
    for aid in range(arc_id):
        u, v = arc_to_edge[aid]
        for aid2 in arcs_by_head[v]:
            if aid2 != aid:
                w = arc_to_edge[aid2][1]
                if w != u:
                    adj[aid].append(aid2)
    
    left = list(range(arc_id))
    right = list(range(arc_id))
    
    matching = _hopcroft_karp(left, right, adj)
    
    # Now extract cycles from the matching
    # Each matched pair (aid1 -> aid2) means arc aid1 is followed by arc aid2
    # This forms directed paths in the line graph
    # We need to find cycles in this functional graph
    
    # Build next pointer: for each arc, what arc follows it (from matching)
    # matching maps left arc -> right arc
    next_arc = {}
    for a1, a2 in matching.items():
        next_arc[a1] = a2
    
    # Find cycles in the directed graph defined by next_arc
    visited = set()
    cycles = []  # list of lists of arcs
    used_edges = set()
    
    for start in range(arc_id):
        if start in visited:
            continue
        # Follow the chain
        path = []
        cur = start
        in_path = {}
        while cur is not None and cur not in visited:
            if cur in in_path:
                # Found a cycle
                cycle_start = in_path[cur]
                cycle_arcs = path[cycle_start:]
                cycles.append(cycle_arcs)
                break
            in_path[cur] = len(path)
            path.append(cur)
            visited.add(cur)
            cur = next_arc.get(cur, None)
    
    # Convert arc cycles to vertex cycles
    pieces = []
    for cycle_arcs in cycles:
        # Extract vertices from arcs
        vertices = []
        for aid in cycle_arcs:
            u, v = arc_to_edge[aid]
            if not vertices or vertices[-1] != u:
                vertices.append(u)
            vertices.append(v)
        # The last vertex should equal the first
        if len(vertices) >= 2 and vertices[0] == vertices[-1]:
            vertices = vertices[:-1]
        # Check if it's a valid cycle (at least 3 distinct vertices)
        if len(set(vertices)) >= 3 and len(vertices) >= 3:
            # Mark edges as used
            for aid in cycle_arcs:
                u, v = arc_to_edge[aid]
                used_edges.add(tuple(sorted((u, v))))
            pieces.append(vertices[:])
    
    # Add remaining edges as single pieces
    for (u, v) in edges:
        key = tuple(sorted((u, v)))
        if key not in used_edges:
            pieces.append([u, v])
    
    return pieces

def construct():
    deadline = time.time() + 130  # Safety margin before 150s limit
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    for spec in specs:
        if time.time() > deadline:
            # Fallback: all edges as single pieces
            edges = _instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = _instance(*spec)
        pieces = _decompose(edges)
        results.append(pieces)
    return results
# EVOLVE-BLOCK-END
