import time
import random
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = set()
        for i in range(n - 1):
            u, v = order[i], order[i + 1]
            edges.add((min(u, v), max(u, v)))
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return list(edges)

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    all_results = []

    for spec in specs:
        n, p, seed = spec
        edges = instance(n, p, seed)
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        deg = {i: 0 for i in range(n)}
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        odd_count = sum(1 for d in deg.values() if d % 2 != 0)
        lower_bound = max(1, odd_count // 2)
        
        best_paths = None
        best_count = float('inf')
        
        # 1. Try Exact Matching Approach
        # Build Cost Matrix
        # Cost 0 if edge exists, 1 if not.
        # We want to maximize matches with cost 0.
        cost = np.ones((n, n))
        for u, v in edges:
            cost[u][v] = 0
            cost[v][u] = 0
            
        # Hungarian Algorithm
        row_ind, col_ind = linear_sum_assignment(cost)
        
        # Reconstruct Matching
        # match_left[u] = v if (u,v) is in matching
        match_next = {}
        match_prev = {}
        m_size = 0
        for u, v in zip(row_ind, col_ind):
            if cost[u][v] == 0:
                match_next[u] = v
                match_prev[v] = u
                m_size += 1
        
        # Reconstruct Paths/Cycles
        # Vertices not in match_prev are starts
        starts = [i for i in range(n) if i not in match_prev]
        
        paths_from_matching = []
        visited_vertices_in_path = set() # Not needed for simple path check if we follow links
        
        for s in starts:
            path = [s]
            curr = s
            while curr in match_next:
                curr = match_next[curr]
                path.append(curr)
            if len(path) > 1:
                paths_from_matching.append(path)
            # If path is length 1, it's an isolated vertex? 
            # But we cover edges. If a vertex has no edges, it's isolated.
            # In our graph, all vertices are in the path cover if they have edges.
            # Wait, if a vertex has degree 0, it's not in any path.
            # But our graph is connected and N>=20, so no isolated vertices.
            
        # Check for Cycles
        # A cycle is a path where start == end? No, start is not in match_prev.
        # If there is a cycle, no vertex in it is a start.
        # So cycles are not captured in `starts`.
        # We need to find cycles in the matching.
        # A cycle exists if we follow `match_next` and return to start.
        # Since we started from `starts` (nodes with in-degree 0 in matching),
        # we only found paths.
        # Any node not visited in `paths_from_matching` must be part of a cycle.
        
        visited_all = set()
        for p in paths_from_matching:
            visited_all.update(p)
            
        cycle_nodes = [i for i in range(n) if i not in visited_all]
        
        # Extract cycles
        for c in cycle_nodes:
            if c in visited_all: continue # Already processed? No, cycle nodes are not in paths.
            # Trace cycle
            path = [c]
            curr = c
            while True:
                curr = match_next[curr]
                path.append(curr)
                if curr == c:
                    break
            # This is a cycle. Break it into a path.
            # Remove last element (which is c) to make it a path?
            # The edges are (c, v1), (v1, v2)...(vk, c).
            # Path: c, v1, ..., vk.
            # This uses all edges of the cycle.
            # It is a simple path.
            if len(path) > 1:
                paths_from_matching.append(path[:-1])
            visited_all.update(path)
            
        # Count paths
        count_matching = len(paths_from_matching)
        
        # Verify validity
        # Edge disjointness and coverage
        used_edges = set()
        valid = True
        for p in paths_from_matching:
            for i in range(len(p) - 1):
                u, v = p[i], p[i+1]
                e = (min(u, v), max(u, v))
                if e in used_edges:
                    valid = False
                    break
                if e not in edges:
                    valid = False
                    break
                used_edges.add(e)
            if not valid: break
            
        if valid and used_edges == set(edges):
            if count_matching < best_count:
                best_count = count_matching
                best_paths = paths_from_matching
        
        # 2. If matching didn't find optimal or failed, use Greedy
        if best_count > lower_bound:
             # Run Greedy with restarts
             # (Keep the existing greedy logic but limit time)
             pass # Placeholder, will implement in full code

        all_results.append(best_paths if best_paths else [list(e) for e in edges])
        
    return all_results
