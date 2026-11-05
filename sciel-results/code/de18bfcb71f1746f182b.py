"""Path decomposition via iterative matching-based merging."""
import time
from collections import defaultdict
from random import Random

def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _max_matching(bipartite, left, right):
    """Hopcroft-Karp style max matching. Returns list of (left_node, right_node) pairs."""
    # Build adjacency
    adj = defaultdict(list)
    for l, r in bipartite:
        adj[l].append(r)
    match_l = {}
    match_r = {}
    from collections import deque
    
    def bfs():
        dist = {}
        q = deque()
        for l in left:
            if l not in match_l:
                dist[l] = 0
                q.append(l)
        found = False
        while q:
            u = q.popleft()
            for v in adj.get(u, []):
                mu = match_r.get(v)
                if mu is None:
                    found = True
                elif mu not in dist:
                    dist[mu] = dist[u] + 1
                    q.append(mu)
        return found
    
    def dfs(u):
        for v in adj.get(u, []):
            mu = match_r.get(v)
            if mu is None or (mu in dist and dist[mu] == dist[u] + 1 and dfs(mu)):
                match_l[u] = v
                match_r[v] = u
                return True
        dist[u] = -1
        return False
    
    while bfs():
        for l in left:
            if l not in match_l:
                dfs(l)
    
    return [(l, match_l[l]) for l in match_l]

def _decompose(edges, deadline):
    """Decompose edge set into minimum number of edge-disjoint simple paths."""
    if not edges:
        return []
    
    # Build adjacency with edge tracking
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Initialize: each edge is a path
    # paths: list of (start, end, set_of_edges, vertex_set)
    # We track edges by frozenset to avoid duplicates
    edge_set = set(edges)
    paths = []
    for u, v in edges:
        paths.append((u, v, frozenset([(u,v)]), frozenset([u, v])))
    
    # Iteratively merge paths
    # A path (a, b, edges_a, verts_a) can merge with (c, d, edges_c, verts_c)
    # if b == c or b == d or a == c or a == d (endpoint matching)
    # AND the merged path is simple (no repeated vertices)
    # AND no shared edges (guaranteed since we track edge sets)
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Build auxiliary bipartite graph for matching
        # Left: path indices (representing one endpoint), Right: path indices (representing other endpoint)
        # Actually, let's think of it differently:
        # Each path has two endpoints. We want to connect endpoint of path i to endpoint of path j
        # such that they share a vertex (the connection point).
        
        # Simpler approach: for each pair of paths that can be merged, create a merge option.
        # Find a matching of merge operations (each path used at most once per merge round).
        
        # Build list of possible merges: (i, j, new_start, new_end)
        n_paths = len(paths)
        possible_merges = []
        
        for i in range(n_paths):
            for j in range(i + 1, n_paths):
                a1, b1, e1, v1 = paths[i]
                a2, b2, e2, v2 = paths[j]
                
                # Try all 4 combinations of endpoint connections
                for (end1, start1, end2, start2) in [(b1, a1, a2, b2), (b1, a1, b2, a2),
                                                      (a1, b1, a2, b2), (a1, b1, b2, a2)]:
                    # end1 of path i connects to end2 of path j at shared vertex
                    if end1 != end2:
                        continue
                    # Check simplicity: merged vertex set must have no duplicates
                    # The shared vertex is end1 == end2
                    # New path: start1 ... end1(=end2) ... start2
                    # Vertices: v1 | v2, but end1/end2 counted once
                    new_verts = v1 | v2
                    # Check that no vertex appears twice in the combined path
                    # Since each path is simple, we just need v1 and v2 to intersect only at the connection point
                    if v1 & v2 != frozenset([end1]):
                        continue
                    # New path is simple
                    new_edges = e1 | e2
                    possible_merges.append((i, j, start1, start2, new_edges, new_verts))
        
        if not possible_merges:
            break
        
        # Now find a matching: each path index can be used in at most one merge
        # Build bipartite: left = path indices, right = path indices
        # Edge (i, j) exists if there's a merge combining i and j
        # We want max matching to maximize merges per round
        
        # For matching, we just need to know which pairs can merge
        pair_can_merge = set()
        merge_info = {}
        for (i, j, ns, ne, ne_e, ne_v) in possible_merges:
            pair_can_merge.add((i, j))
            if (i, j) not in merge_info:
                merge_info[(i, j)] = (ns, ne, ne_e, ne_v)
        
        # Build bipartite for matching
        bipartite_edges = []
        for (i, j) in pair_can_merge:
            bipartite_edges.append((i, j))
        
        left_nodes = list(set(i for i, j in pair_can_merge))
        right_nodes = list(set(j for i, j in pair_can_merge))
        
        matching = _max_matching(bipartite_edges, left_nodes, right_nodes)
        
        if not matching:
            break
        
        # Apply merges
        used = set()
        new_paths = []
        for (i, j) in matching:
            if i in used or j in used:
                continue
            ns, ne, ne_e, ne_v = merge_info[(i, j)]
            new_paths.append((ns, ne, ne_e, ne_v))
            used.add(i)
            used.add(j)
        
        for idx in range(n_paths):
            if idx not in used:
                new_paths.append(paths[idx])
        
        paths = new_paths
        changed = True
        
        # Early termination if we hit lower bound
        odd_count = sum(1 for v in adj if len(adj[v]) % 2 == 1)
        lower_bound = max(odd_count // 2, 1) if odd_count > 0 else (1 if edges else 0)
        if len(paths) <= lower_bound:
            break
    
    # Convert back to vertex lists
    result = []
    for (start, end, e_set, v_set) in paths:
        # Reconstruct path by walking
        # We need to recover the actual sequence. Since we lost it, let's track it differently.
        pass
    
    # Actually, let's redo this with proper path tracking
    return _decompose_v2(edges, deadline)

def _decompose_v2(edges, deadline):
    """Proper path decomposition with full path tracking."""
    if not edges:
        return []
    
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Each path stored as list of vertices
    paths = [[u, v] for u, v in edges]
    edge_set = set(edges)
    
    def path_edges(p):
        return frozenset({tuple(sorted((p[k], p[k+1]))) for k in range(len(p)-1)})
    
    def path_verts(p):
        return frozenset(p)
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        n_paths = len(paths)
        
        # Find all valid merges
        possible_merges = []
        for i in range(n_paths):
            for j in range(i + 1, n_paths):
                p1, p2 = paths[i], paths[j]
                # Try 4 connections
                for (idx1, idx2, reverse1, reverse2) in [(len(p1)-1, 0, False, False),
                                                          (len(p1)-1, len(p2)-1, False, True),
                                                          (0, 0, True, False),
                                                          (0, len(p2)-1, True, True)]:
                    v1 = p1[idx1]
                    v2 = p2[idx2]
                    if v1 != v2:
                        continue
                    # Build merged path
                    part1 = p1 if not reverse1 else p1[::-1]
                    part2 = p2 if not reverse2 else p2[::-1]
                    # part1 ends at v1, part2 starts at v2 (after potential reversal)
                    # Actually let me be more careful
                    # If reverse1: we use p1 reversed, so it ends at p1[0]
                    # Let me just try all 4 endpoint pairings explicitly
                    
                # Simpler: try connecting each endpoint of p1 to each endpoint of p2
                for e1 in [0, len(p1)-1]:
                    for e2 in [0, len(p2)-1]:
                        v1 = p1[e1]
                        v2 = p2[e2]
                        if v1 != v2:
                            continue
                        # Determine orientation
                        # p1 segment from other end to e1
                        if e1 == 0:
                            seg1 = p1  # p1[0]=v1, goes to p1[-1]
                            # We want to start from p1[-1] and end at v1=p1[0]
                            seg1 = p1[::-1]  # now ends at v1
                        else:
                            seg1 = p1  # ends at v1
                        if e2 == 0:
                            seg2 = p2  # starts at v2=v1
                        else:
                            seg2 = p2[::-1]  # starts at v2
                        # Merge: seg1 (ends at v1) + seg2 (starts at v2=v1), drop duplicate
                        merged = seg1 + seg2[1:]
                        # Check simplicity
                        if len(merged) != len(set(merged)):
                            continue
                        # Check edge validity
                        m_edges = set()
                        valid = True
                        for k in range(len(merged)-1):
                            a, b = merged[k], merged[k+1]
                            et = tuple(sorted((a, b)))
                            if et not in edge_set:
                                valid = False
                                break
                            if et in m_edges:
                                valid = False
                                break
                            m_edges.add(et)
                        if not valid:
                            continue
                        possible_merges.append((i, j, merged))
        
        if not possible_merges:
            break
        
        # Build matching
        pair_set = set()
        merge_map = {}
        for (i, j, merged) in possible_merges:
            pair_set.add((i, j))
            if (i, j) not in merge_map:
                merge_map[(i, j)] = merged
        
        bipartite = list(pair_set)
        left_nodes = list(set(i for i, j in pair_set))
        right_nodes = list(set(j for i, j in pair_set))
        
        matching = _max_matching(bipartite, left_nodes, right_nodes)
        
        if not matching:
            break
        
        used = set()
        new_paths = []
        for (i, j) in matching:
            if i in used or j in used:
                continue
            new_paths.append(merge_map[(i, j)])
            used.add(i)
            used.add(j)
        
        for idx in range(n_paths):
            if idx not in used:
                new_paths.append(paths[idx])
        
        paths = new_paths
        changed = True
        
        # Check lower bound
        odd_count = sum(1 for v in adj if len(adj[v]) % 2 == 1)
        lb = odd_count // 2 if odd_count > 0 else (1 if edges else 0)
        if len(paths) <= lb:
            break
    
    return paths

def construct():
    deadline = time.time() + 110  # Safety margin before 150s kill
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = _instance(*spec)
        paths = _decompose_v2(edges, deadline)
        results.append(paths)
        if time.time() > deadline:
            # Fallback: trivial decomposition
            for spec2 in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))[len(results):]:
                edges2 = _instance(*spec2)
                results.append([[u, v] for u, v in edges2])
            break
    return results
