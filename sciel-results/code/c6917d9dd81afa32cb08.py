import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflicts_bitmask(edges, n):
        edge_index = {e: i for i, e in enumerate(edges)}
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        n_edges = len(edges)
        conflicts = [0] * n_edges
        
        for i, (u, v) in enumerate(edges):
            near_vertices = set(adj[u]) | set(adj[v])
            near_vertices.add(u)
            near_vertices.add(v)
            
            for x in near_vertices:
                for y in adj[x]:
                    e2 = (min(x, y), max(x, y))
                    if e2 in edge_index:
                        j = edge_index[e2]
                        if j != i:
                            conflicts[i] |= (1 << j)
                            conflicts[j] |= (1 << i)
                            
        return conflicts

    def hopcroft_karp(n_left, n_right, adj):
        """
        Hopcroft-Karp maximum bipartite matching.
        adj: list of lists, adj[i] = list of right vertices adjacent to left vertex i
        Returns: (matching_size, match_left, match_right)
        """
        match_l = [-1] * n_left
        match_r = [-1] * n_right
        dist = [0] * n_left
        
        def bfs():
            queue = deque()
            for u in range(n_left):
                if match_l[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = -1
            found = False
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    mu = match_r[v]
                    if mu == -1:
                        found = True
                    elif dist[mu] == -1:
                        dist[mu] = dist[u] + 1
                        queue.append(mu)
            return found
        
        def dfs(u):
            for v in adj[u]:
                mu = match_r[v]
                if mu == -1 or (dist[mu] == dist[u] + 1 and dfs(mu)):
                    match_l[u] = v
                    match_r[v] = u
                    return True
            dist[u] = -1
            return False
        
        matching = 0
        while bfs():
            for u in range(n_left):
                if match_l[u] == -1:
                    if dfs(u):
                        matching += 1
        return matching, match_l, match_r

    def max_induced_matching_bipartite(edges, conflicts, n_edges, remaining_mask):
        """
        Find a maximum induced matching in the subgraph defined by remaining_mask.
        Uses bipartite matching on a derived graph.
        """
        rem_indices = [i for i in range(n_edges) if (remaining_mask >> i) & 1]
        if not rem_indices:
            return []
        
        k = len(rem_indices)
        
        # For each pair of non-conflicting edges, we can potentially include both.
        # The induced matching problem is NP-hard in general, but for bounded degree graphs
        # we can use a greedy approach or approximation.
        
        # Since exact maximum induced matching is hard, use a greedy heuristic:
        # Repeatedly pick the edge with minimum degree in the conflict graph among remaining,
        # add it to the matching, and remove all conflicting edges.
        
        # Build local conflict structure
        local_conflicts = [0] * k
        for i_idx, i_global in enumerate(rem_indices):
            mask = remaining_mask & conflicts[i_global]
            for j_idx, j_global in enumerate(rem_indices):
                if (mask >> j_global) & 1:
                    local_conflicts[i_idx] |= (1 << j_idx)
        
        # Greedy: pick edges with minimum local degree
        remaining_local = (1 << k) - 1
        matching = []
        
        while remaining_local:
            # Find vertex with minimum degree in remaining_local
            min_deg = float('inf')
            best_v = -1
            for i in range(k):
                if (remaining_local >> i) & 1:
                    deg = bin(local_conflicts[i] & remaining_local).count('1')
                    if deg < min_deg:
                        min_deg = deg
                        best_v = i
            
            if best_v == -1:
                break
                
            matching.append(best_v)
            # Remove best_v and all its neighbors from remaining
            remaining_local &= ~(1 << best_v)
            remaining_local &= ~local_conflicts[best_v]
        
        return [rem_indices[i] for i in matching]

    def greedy_color(order, conflicts, n_edges):
        """Standard greedy coloring for initialization or fallback."""
        color = [-1] * n_edges
        color_to_edges = []
        color_masks = []  # bitmask of edges in each color
        
        for idx in order:
            used_mask = 0
            for c_idx, mask in enumerate(color_masks):
                if conflicts[idx] & mask:
                    used_mask |= (1 << c_idx)
            
            c = 0
            while (used_mask >> c) & 1:
                c += 1
            
            color[idx] = c
            if c >= len(color_to_edges):
                color_to_edges.append(set())
                color_masks.append(0)
            color_to_edges[c].add(idx)
            color_masks[c] |= (1 << idx)
            
        return color, len(color_to_edges)

    def local_polish(color, conflicts, n_edges, deadline):
        """
        Local search to improve the coloring.
        """
        n_colors = max(color) + 1 if color else 0
        if n_colors == 0:
            return color, n_colors
            
        classes = [set() for _ in range(n_colors)]
        class_masks = [0] * n_colors
        for i, c in enumerate(color):
            classes[c].add(i)
            class_masks[c] |= (1 << i)
            
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            
            # 1. Try to eliminate colors
            for c in range(n_colors - 1, -1, -1):
                if not classes[c]:
                    continue
                temp_assign = {}
                success = True
                for e in classes[c]:
                    placed = False
                    for c2 in range(n_colors):
                        if c2 == c:
                            continue
                        if not (conflicts[e] & class_masks[c2]):
                            temp_assign[e] = c2
                            placed = True
                            break
                    if not placed:
                        success = False
                        break
                
                if success:
                    for e, c2 in temp_assign.items():
                        classes[c].discard(e)
                        classes[c2].add(e)
                        class_masks[c] &= ~(1 << e)
                        class_masks[c2] |= (1 << e)
                        color[e] = c2
                    n_colors -= 1
                    improved = True
                    break
            
            # 2. Move individual edges to lower colors
            if not improved:
                for c in range(n_colors - 1, -1, -1):
                    if not classes[c]:
                        continue
                    for e in list(classes[c]):
                        for c2 in range(c):
                            if not (conflicts[e] & class_masks[c2]):
                                classes[c].discard(e)
                                classes[c2].add(e)
                                class_masks[c] &= ~(1 << e)
                                class_masks[c2] |= (1 << e)
                                color[e] = c2
                                improved = True
                                break
                        if improved:
                            if not classes[n_colors - 1]:
                                n_colors -= 1
                            break
                    if improved:
                        break
                        
        return color, n_colors

    def construct_imim(edges, conflicts, n_edges, deadline):
        """
        Iterative Maximum Induced Matching using greedy heuristic.
        """
        color = [-1] * n_edges
        remaining = (1 << n_edges) - 1
        current_color = 0
        
        while remaining:
            if time.time() > deadline - 10.0:
                # Fallback: greedy color remaining edges
                rem_indices = [i for i in range(n_edges) if (remaining >> i) & 1]
                # Assign new colors to remaining edges using greedy
                for idx in rem_indices:
                    c = current_color
                    while True:
                        # Check if idx can use color c
                        valid = True
                        for j in range(n_edges):
                            if color[j] == c and (conflicts[idx] >> j) & 1:
                                valid = False
                                break
                        if valid:
                            break
                        c += 1
                    color[idx] = c
                    if c >= current_color:
                        current_color = c + 1
                break
            
            # Find a large induced matching in remaining
            matching = max_induced_matching_bipartite(edges, conflicts, n_edges, remaining)
            
            if not matching:
                matching = [i for i in range(n_edges) if (remaining >> i) & 1][:1]
            
            # Assign color
            for i_global in matching:
                color[i_global] = current_color
                remaining &= ~(1 << i_global)
            
            current_color += 1
            
        return color, current_color

    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        n_edges = len(edges)
        conflicts = build_conflicts_bitmask(edges, n)
        
        # 1. IMIM Construction
        color, n_colors = construct_imim(edges, conflicts, n_edges, deadline)
        
        # 2. Local Polish
        color, n_colors = local_polish(color, conflicts, n_edges, deadline)
        
        results.append(color)
        
    return results
