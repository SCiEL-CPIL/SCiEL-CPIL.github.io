import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 143.0  # safety margin before 150s kill

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

    def build_conflict(edges, adj):
        """Build conflict sets: two edges conflict if they share a vertex or are joined by an edge."""
        n_edges = len(edges)
        edge_index = {e: i for i, e in enumerate(edges)}
        conflicts = [set() for _ in range(n_edges)]
        
        for i, (u, v) in enumerate(edges):
            # Edges sharing u or v
            for w in adj[u]:
                e1 = (min(u, w), max(u, w))
                if e1 in edge_index:
                    j = edge_index[e1]
                    if j != i:
                        conflicts[i].add(j)
                        conflicts[j].add(i)
            for w in adj[v]:
                e1 = (min(v, w), max(v, w))
                if e1 in edge_index:
                    j = edge_index[e1]
                    if j != i:
                        conflicts[i].add(j)
                        conflicts[j].add(i)
            # Edges joined by an edge (distance 2 in line graph)
            for w in adj[u]:
                for z in adj[w]:
                    if z == u or z == v:
                        continue
                    e2 = (min(w, z), max(w, z))
                    if e2 in edge_index:
                        j = edge_index[e2]
                        if j != i:
                            conflicts[i].add(j)
                            conflicts[j].add(i)
            for w in adj[v]:
                for z in adj[w]:
                    if z == u or z == v:
                        continue
                    e2 = (min(w, z), max(w, z))
                    if e2 in edge_index:
                        j = edge_index[e2]
                        if j != i:
                            conflicts[i].add(j)
                            conflicts[j].add(i)
        
        return conflicts

    def hopcroft_karp(left_nodes, right_nodes, match_left, match_right, adj_lr):
        """Hopcroft-Karp bipartite matching. adj_lr maps left node -> list of right nodes."""
        # match_left: dict left->right, match_right: dict right->left
        INF = float('inf')
        
        def bfs():
            dist = {u: 0 for u in left_nodes}
            found = False
            q = deque()
            for u in left_nodes:
                if match_left.get(u) is None:
                    q.append(u)
            while q:
                u = q.popleft()
                for v in adj_lr.get(u, []):
                    mu = match_right.get(v)
                    if mu is None:
                        found = True
                    elif dist.get(mu, INF) == INF:
                        dist[mu] = dist[u] + 1
                        q.append(mu)
            return found
        
        def dfs(u, visited):
            for v in adj_lr.get(u, []):
                mu = match_right.get(v)
                if mu is None or (dist.get(mu, INF) == dist.get(u, 0) + 1 and dfs(mu, visited)):
                    match_left[u] = v
                    match_right[v] = u
                    return True
            dist[u] = INF
            return False
        
        while bfs():
            for u in left_nodes:
                if match_left.get(u) is None:
                    dfs(u, set())
        
        return sum(1 for u in left_nodes if match_left.get(u) is not None)

    def max_induced_matching_residual(remaining_edges, conflicts, edge_index):
        """
        Find a maximum induced matching among remaining edges.
        remaining_edges: set of edge indices
        conflicts: list of sets
        Returns: list of edge indices forming the induced matching
        """
        if not remaining_edges:
            return []
        
        # Order edges for the bipartite matching reduction
        # We use a simple approach: try to find a large induced matching greedily
        # with backtracking for small cases, or use the conflict structure
        
        # For the bipartite matching approach to induced matching:
        # This is actually NP-hard in general, but for bounded degree graphs we can
        # use a greedy approach with local search
        
        # Greedy: repeatedly pick the edge with minimum conflict degree among remaining
        result = []
        remaining = set(remaining_edges)
        # Sort by degree in the conflict subgraph
        while remaining:
            if time.time() > deadline - 10:
                break
            # Find edge with minimum number of conflicts within remaining
            min_conf = None
            min_count = float('inf')
            for e in remaining:
                cnt = len(conflicts[e] & remaining)
                if cnt < min_count:
                    min_count = cnt
                    min_conf = e
            if min_conf is None:
                break
            result.append(min_conf)
            # Remove min_conf and all its conflicts from remaining
            to_remove = conflicts[min_conf] | {min_conf}
            remaining -= to_remove
        
        return result

    def greedy_color(order, conflicts, n_edges):
        """Color edges in given order using first-fit."""
        color = [-1] * n_edges
        color_edges = []  # color_edges[c] = set of edge indices with color c
        for idx in order:
            used = set()
            for c in range(len(color_edges)):
                if color_edges[c] & conflicts[idx]:
                    used.add(c)
            c = 0
            while c in used:
                c += 1
            color[idx] = c
            if c >= len(color_edges):
                color_edges.append(set())
            color_edges[c].add(idx)
        return color, len(color_edges)

    def improve_coloring(color, conflicts, n_edges, max_iter=3000):
        """Try to reduce number of colors by moving edges between classes."""
        n_colors = max(color) + 1 if color else 0
        if n_colors == 0:
            return color, n_colors
        
        classes = [set() for _ in range(n_colors)]
        for i, c in enumerate(color):
            classes[c].add(i)
        
        improved = True
        while improved and max_iter > 0:
            improved = False
            max_iter -= 1
            
            # Try to remove a color class
            for c in range(n_colors):
                if len(classes[c]) == 0:
                    continue
                # Try to move all edges in class c to other classes
                temp_assign = {}
                success = True
                for e in sorted(classes[c]):
                    placed = False
                    for c2 in range(n_colors):
                        if c2 == c:
                            continue
                        if not (conflicts[e] & classes[c2]):
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
                        color[e] = c2
                    n_colors -= 1
                    improved = True
                    break
            
            # Also try swapping edges between two classes to enable removal
            if not improved and n_colors >= 2:
                # Try 1-swap: move one edge from class A to class B and vice versa
                for c1 in range(n_colors):
                    if len(classes[c1]) == 0:
                        continue
                    for c2 in range(n_colors):
                        if c2 == c1 or len(classes[c2]) == 0:
                            continue
                        # Try moving e1 from c1 to c2 and e2 from c2 to c1
                        for e1 in list(classes[c1]):
                            for e2 in list(classes[c2]):
                                # Check if e1 can go to c2 (after removing e2 from c2)
                                if conflicts[e1] & (classes[c2] - {e2}):
                                    continue
                                # Check if e2 can go to c1 (after removing e1 from c1)
                                if conflicts[e2] & (classes[c1] - {e1}):
                                    continue
                                # Perform swap
                                classes[c1].discard(e1)
                                classes[c1].add(e2)
                                classes[c2].discard(e2)
                                classes[c2].add(e1)
                                color[e1] = c2
                                color[e2] = c1
                                improved = True
                                break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
        
        return color, n_colors

    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        edges = instance(*spec)
        n_edges = len(edges)
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        conflicts = build_conflict(edges, adj)
        conf_deg = [len(conflicts[i]) for i in range(n_edges)]
        
        best_color = None
        best_n = float('inf')
        
        # Strategy 1: Iterative maximum induced matching (greedy min-degree)
        remaining = set(range(n_edges))
        color_iter = [-1] * n_edges
        c = 0
        while remaining:
            if time.time() > deadline - 5:
                break
            # Find induced matching using min-degree greedy
            match = []
            rem = set(remaining)
            while rem:
                if time.time() > deadline - 10:
                    break
                # Pick edge with min conflict degree in rem
                min_e = min(rem, key=lambda e: len(conflicts[e] & rem))
                match.append(min_e)
                rem -= (conflicts[min_e] | {min_e})
            for e in match:
                color_iter[e] = c
            remaining -= set(match)
            c += 1
        
        if best_color is None or c < best_n:
            best_n = c
            best_color = color_iter[:]
        
        # Strategy 2: Greedy with descending conflict degree
        order1 = sorted(range(n_edges), key=lambda i: -conf_deg[i])
        color1, n1 = greedy_color(order1, conflicts, n_edges)
        color1, n1 = improve_coloring(color1, conflicts, n_edges)
        if n1 < best_n:
            best_n = n1
            best_color = color1[:]
        
        # Strategy 3: Random restarts with greedy + local search
        rng = random.Random(123)
        for attempt in range(100):
            if time.time() > deadline - 5:
                break
            order = list(range(n_edges))
            rng.shuffle(order)
            color, n = greedy_color(order, conflicts, n_edges)
            color, n = improve_coloring(color, conflicts, n_edges, max_iter=1500)
            if n < best_n:
                best_n = n
                best_color = color[:]
        
        # Strategy 4: Weighted random ordering
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            # Higher conflict degree = higher priority
            order = sorted(range(n_edges), key=lambda i: (-conf_deg[i], rng.random()))
            color, n = greedy_color(order, conflicts, n_edges)
            color, n = improve_coloring(color, conflicts, n_edges, max_iter=1500)
            if n < best_n:
                best_n = n
                best_color = color[:]
        
        if best_color is None:
            best_color = list(range(n_edges))
            best_n = n_edges
        
        results.append([best_color[i] for i in range(n_edges)])
    
    return results
