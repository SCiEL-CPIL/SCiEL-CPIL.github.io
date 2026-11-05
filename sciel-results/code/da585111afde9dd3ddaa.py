import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

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

    def build_conflict_graph(edges, adj):
        """Build conflict graph: two edges conflict if they share a vertex or are joined by an edge."""
        m = len(edges)
        if m == 0:
            return []
        
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        conflicts = [set() for _ in range(m)]
        
        for i, (u, v) in enumerate(edges):
            for x in adj[u]:
                for y in adj[x]:
                    e2 = (min(x, y), max(x, y))
                    if e2 in edge_to_idx:
                        conflicts[i].add(edge_to_idx[e2])
            for x in adj[v]:
                for y in adj[x]:
                    e2 = (min(x, y), max(x, y))
                    if e2 in edge_to_idx:
                        conflicts[i].add(edge_to_idx[e2])
                        
            conflicts[i].discard(i)
            
        return conflicts

    def greedy_color_with_order(edges, conflicts, order):
        """Color edges in the given order using smallest available color."""
        m = len(edges)
        color_of_edge = [-1] * m
        used_colors = 0
        
        for i in order:
            forbidden = set()
            for j in conflicts[i]:
                if color_of_edge[j] != -1:
                    forbidden.add(color_of_edge[j])
            
            c = 0
            while c in forbidden:
                c += 1
            
            color_of_edge[i] = c
            if c + 1 > used_colors:
                used_colors = c + 1
        
        return color_of_edge, used_colors

    def backtracking_k_coloring(conflicts, k, time_limit):
        """
        Try to color the conflict graph with exactly k colors using backtracking
        with forward checking and MRV (Minimum Remaining Values) heuristic.
        Returns a valid coloring (list of colors) or None if not possible within time.
        """
        m = len(conflicts)
        if m == 0:
            return []
        
        # Convert conflicts to lists for faster access
        conflict_lists = [list(c) for c in conflicts]
        
        # Precompute degree for ordering
        degrees = [len(c) for c in conflict_lists]
        
        # Start with a greedy ordering by degree (descending)
        order = list(range(m))
        order.sort(key=lambda i: degrees[i], reverse=True)
        
        colors = [-1] * m
        # available[c] = set of vertices that can use color c
        # Actually, let's use a different approach: for each vertex, track which colors are still available
        # available_colors[v] = set of colors 0..k-1 that are not used by any colored neighbor
        
        available = [set(range(k)) for _ in range(m)]
        
        # Forward checking: when we assign color c to vertex v, remove c from available of all uncolored neighbors
        # We need to track which vertices are colored
        
        def solve(idx, deadline):
            if time.time() > deadline:
                return False
            
            if idx == m:
                return True
            
            v = order[idx]
            
            # Try each available color for v
            for c in sorted(available[v]):
                # Check if assignment is valid (it should be by construction of available)
                # Assign
                colors[v] = c
                
                # Forward check: remove c from available of uncolored neighbors
                removed = []
                valid = True
                for nb in conflict_lists[v]:
                    if colors[nb] == -1:  # uncolored
                        if c in available[nb]:
                            available[nb].remove(c)
                            removed.append(nb)
                            if not available[nb]:  # conflict
                                valid = False
                                break
                
                if valid:
                    if solve(idx + 1, deadline):
                        return True
                
                # Backtrack
                for nb in removed:
                    available[nb].add(c)
                colors[v] = -1
            
            return False
        
        # Try to solve
        result = solve(0, time_limit)
        if result:
            return colors
        return None

    def find_min_colors(conflicts, initial_upper_bound, time_per_attempt=5.0):
        """
        Binary search / iterative deepening to find minimum k such that
        the conflict graph can be colored with k colors.
        """
        m = len(conflicts)
        if m == 0:
            return [], 0
        
        # Lower bound: max degree + 1
        max_degree = max(len(c) for c in conflicts)
        lower = max_degree + 1
        
        # Upper bound from greedy
        order = list(range(m))
        order.sort(key=lambda i: len(conflicts[i]), reverse=True)
        greedy_colors, upper = greedy_color_with_order(None, conflicts, order)
        
        if upper < lower:
            upper = lower
        
        # Try k from lower to upper
        best_colors = greedy_colors
        best_k = upper
        
        for k in range(lower, upper + 1):
            if time.time() > deadline - 20.0:
                break
            
            # Try backtracking with time limit
            time_limit = time.time() + time_per_attempt
            bt_colors = backtracking_k_coloring(conflicts, k, time_limit)
            
            if bt_colors is not None:
                best_colors = bt_colors
                best_k = k
                break  # Found minimum, stop
        
        return best_colors, best_k

    def local_search_improve(colors, conflicts, max_iters=200):
        """Try to reduce the number of colors used by recoloring."""
        m = len(colors)
        if m == 0:
            return colors, 0
        
        current_max = max(colors) + 1 if colors else 0
        conflict_lists = [list(c) for c in conflicts]
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            for i in range(m):
                current_color = colors[i]
                if current_color == 0:
                    continue
                # Try to reduce to a lower color
                forbidden = set()
                for j in conflict_lists[i]:
                    forbidden.add(colors[j])
                
                for c in range(current_color):
                    if c not in forbidden:
                        colors[i] = c
                        improved = True
                        break
        
        new_max = max(colors) + 1 if colors else 0
        return colors, new_max

    def color_graph(edges, adj, time_budget):
        m = len(edges)
        if m == 0:
            return [], 0
        
        conflicts = build_conflict_graph(edges, adj)
        
        # First get a good upper bound with greedy
        order = list(range(m))
        order.sort(key=lambda i: len(conflicts[i]), reverse=True)
        greedy_colors, greedy_k = greedy_color_with_order(None, conflicts, order)
        
        # Try to find optimal or near-optimal with backtracking
        # Allocate most of the time budget to backtracking
        bt_budget = time_budget * 0.8
        ls_budget = time_budget * 0.2
        
        # Try k from lower bound up to greedy_k
        max_degree = max(len(c) for c in conflicts) if conflicts else 0
        lower = max_degree + 1
        
        best_colors = greedy_colors[:]
        best_k = greedy_k
        
        # Try each k from lower to greedy_k - 1
        time_per_k = bt_budget / max(1, greedy_k - lower + 1)
        
        for k in range(lower, greedy_k):
            if time.time() > deadline - 15.0:
                break
            
            time_limit = time.time() + time_per_k
            bt_colors = backtracking_k_coloring(conflicts, k, time_limit)
            
            if bt_colors is not None:
                best_colors = bt_colors
                best_k = k
                break  # Found minimum
        
        # Local search to potentially improve
        if time.time() < deadline - 10.0:
            ls_colors, ls_k = local_search_improve(best_colors, conflicts, max_iters=100)
            if ls_k < best_k:
                best_colors = ls_colors
                best_k = ls_k
        
        return best_colors, best_k

    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))
    
    # Pre-generate graphs
    graphs = []
    for spec in specs:
        n, d, seed = spec
        edges = instance(n, d, seed)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        graphs.append((edges, adj))
    
    # Allocate time: give more time to harder (larger) graphs
    # Graph sizes: ~80 edges, ~150 edges, ~240 edges approximately
    # Allocate time proportional to difficulty
    total_time_available = deadline - start_time - 5.0  # Leave 5s margin
    # Rough allocation: 20%, 30%, 50%
    time_alloc = [total_time_available * 0.2, total_time_available * 0.3, total_time_available * 0.5]
    
    best_results = []
    
    for idx, (edges, adj) in enumerate(graphs):
        t_start = time.time()
        time_budget = min(time_alloc[idx], deadline - t_start - 5.0)
        
        if time_budget < 1.0:
            # Not enough time, just do greedy
            order = list(range(len(edges)))
            conflicts = build_conflict_graph(edges, adj)
            order.sort(key=lambda i: len(conflicts[i]), reverse=True)
            colors, _ = greedy_color_with_order(None, conflicts, order)
            best_results.append(colors)
            continue
        
        colors, k = color_graph(edges, adj, time_budget)
        best_results.append(colors)
    
    return best_results
