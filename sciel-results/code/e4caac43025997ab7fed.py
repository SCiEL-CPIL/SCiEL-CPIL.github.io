"""
Path decomposition via edge-disjoint trail packing with local search.

Strategy:
1. Greedy initialization using longest-path heuristic (Warnsdorff's rule).
2. Aggressive merging of paths that share endpoints.
3. Local search: try to reassign edges between paths to enable further merges.
4. Multiple restarts with different random seeds for the greedy phase.
"""
import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def verify_decomposition(paths, edges, n):
        all_edges = set(tuple(sorted(e)) for e in edges)
        used_edges = set()
        for path in paths:
            if len(path) < 2:
                continue
            if len(set(path)) != len(path):
                return False
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i+1])))
                if e not in all_edges:
                    return False
                if e in used_edges:
                    return False
                used_edges.add(e)
        return used_edges == all_edges

    def is_valid_decomposition_fast(paths, edges_set):
        """Fast check assuming edge-disjointness is maintained by construction.
        Only checks simplicity and that all edges are covered."""
        used_edges = set()
        for path in paths:
            if len(path) < 2:
                continue
            if len(set(path)) != len(path):
                return False
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i+1])))
                if e not in edges_set:
                    return False
                if e in used_edges:
                    return False
                used_edges.add(e)
        return used_edges == edges_set

    def greedy_init(edges, n, rng):
        """Greedy initialization: repeatedly find longest simple path in remaining graph."""
        remaining = set(tuple(sorted(e)) for e in edges)
        adj = defaultdict(set)
        for u, v in remaining:
            adj[u].add(v)
            adj[v].add(u)
        
        paths = []
        while remaining:
            # Pick a starting vertex: prefer high degree
            degs = {v: len(neigh) for v, neigh in adj.items() if neigh}
            if not degs:
                break
            max_deg = max(degs.values())
            candidates = [v for v, d in degs.items() if d == max_deg]
            start = rng.choice(candidates)
            
            # Find a long simple path starting from 'start' using Warnsdorff's rule
            path = [start]
            visited = {start}
            curr = start
            
            while True:
                neighbors = list(adj[curr])
                # Remove neighbors already in path (to keep simple)
                neighbors = [nb for nb in neighbors if nb not in visited]
                if not neighbors:
                    break
                
                # Warnsdorff: prefer neighbor with fewest unvisited neighbors
                def score(nb):
                    return sum(1 for x in adj[nb] if x not in visited)
                
                neighbors.sort(key=score)
                
                extended = False
                for nb in neighbors:
                    e = tuple(sorted((curr, nb)))
                    if e in remaining:
                        path.append(nb)
                        visited.add(nb)
                        remaining.remove(e)
                        adj[curr].discard(nb)
                        adj[nb].discard(curr)
                        curr = nb
                        extended = True
                        break
                if not extended:
                    break
            
            if len(path) > 1:
                paths.append(path)
        
        # Handle any remaining edges (shouldn't happen if logic is correct)
        if remaining:
            for e in remaining:
                paths.append([e[0], e[1]])
        
        # Verify and repair if necessary
        if not verify_decomposition(paths, edges, n):
            # Fallback: trivial decomposition
            return [list(e) for e in edges]
            
        return paths

    def merge_paths(paths, edges, n, time_limit=None):
        """
        Iteratively merge paths that can be combined at shared endpoints.
        Two paths can be merged if they share an endpoint and the merged path is simple.
        Optimized to avoid full verification inside the loop.
        """
        current = [list(p) for p in paths]
        
        improved = True
        max_rounds = 50
        round_count = 0
        while improved and round_count < max_rounds:
            improved = False
            round_count += 1
            if time_limit and time.time() > time_limit:
                break
            
            # Build endpoint map: vertex -> list of (path_index, is_head)
            endpoint_map = defaultdict(list)
            # Also maintain sets of vertices for each path for fast O(1) lookup
            path_vertex_sets = [set(p) for p in current]
            
            for i, p in enumerate(current):
                if len(p) >= 2:
                    endpoint_map[p[0]].append((i, True))   # head
                    endpoint_map[p[-1]].append((i, False))  # tail
            
            for v in list(endpoint_map.keys()):
                entries = endpoint_map[v]
                if len(entries) < 2:
                    continue
                
                # Try all pairs of paths that share endpoint v
                for i in range(len(entries)):
                    idx1, is_head1 = entries[i]
                    for j in range(i+1, len(entries)):
                        idx2, is_head2 = entries[j]
                        if idx1 == idx2:
                            continue
                        
                        p1 = current[idx1]
                        p2 = current[idx2]
                        s1 = path_vertex_sets[idx1]
                        s2 = path_vertex_sets[idx2]
                        
                        new_path = None
                        new_set = None
                        
                        # Case 1: p1 ends at v, p2 starts at v -> p1 + p2[1:]
                        if not is_head1 and is_head2:
                            # Check if p2[1:] has any vertex in p1
                            if not (s1 & s2) - {v}: # Intersection minus v must be empty
                                new_path = p1 + p2[1:]
                                new_set = s1 | s2
                        # Case 2: p1 starts at v, p2 ends at v -> p1[1:] + p2
                        elif is_head1 and not is_head2:
                            if not (s1 & s2) - {v}:
                                new_path = p1[1:] + p2
                                new_set = s1 | s2
                        # Case 3: p1 ends at v, p2 ends at v -> p1 + reversed(p2)[1:]
                        elif not is_head1 and not is_head2:
                            if not (s1 & s2) - {v}:
                                new_path = p1 + p2[::-1][1:]
                                new_set = s1 | s2
                        # Case 4: p1 starts at v, p2 starts at v -> reversed(p1)[1:] + p2
                        elif is_head1 and is_head2:
                            if not (s1 & s2) - {v}:
                                new_path = p1[::-1][1:] + p2
                                new_set = s1 | s2
                        
                        if new_path and len(new_set) == len(new_path):
                            # Construct new list of paths
                            new_current = []
                            for idx, p in enumerate(current):
                                if idx != idx1 and idx != idx2:
                                    new_current.append(p)
                            new_current.append(new_path)
                            
                            # Update state
                            current = new_current
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
        return current

    def local_search(paths, edges, n, rng, time_limit):
        """
        Local search: try to move edges between paths to enable merges.
        Optimized to reduce overhead.
        """
        current = [list(p) for p in paths]
        best = [list(p) for p in current]
        best_count = len(current)
        
        edge_set = set(tuple(sorted(e)) for e in edges)
        
        # Limit iterations to prevent infinite loops if time check fails
        max_iters = 5000
        iters = 0
        
        while time.time() < time_limit and iters < max_iters:
            iters += 1
            
            if len(current) < best_count:
                best = [list(p) for p in current]
                best_count = len(current)
            
            # Try a move: pick a random path and a random edge in it
            if not current or len(current) < 2:
                break
            
            i = rng.randrange(len(current))
            P_i = current[i]
            if len(P_i) < 2:
                continue
            
            # Pick a random edge position
            k = rng.randrange(len(P_i) - 1)
            u, v = P_i[k], P_i[k+1]
            
            # Remove edge e from P_i, splitting into left and right
            left = P_i[:k+1]   # ends at u
            right = P_i[k+1:]  # starts at v
            
            # Build candidate paths: remove P_i, add left and right if they have edges
            candidate = []
            for idx, p in enumerate(current):
                if idx == i:
                    if len(left) >= 2:
                        candidate.append(left)
                    if len(right) >= 2:
                        candidate.append(right)
                else:
                    candidate.append(p)
            
            # Try to insert e into another path in candidate
            inserted = False
            # To speed up, only try a few random paths for insertion
            candidate_indices = list(range(len(candidate)))
            rng.shuffle(candidate_indices)
            
            for idx in candidate_indices[:10]: # Limit insertion attempts
                P_j = candidate[idx]
                if len(P_j) < 2:
                    continue
                
                # Check if we can extend P_j
                # Use set for membership check if P_j is large, else list is fine
                s_j = set(P_j) if len(P_j) > 10 else None
                
                if P_j[-1] == u and (s_j is None or v not in s_j):
                    new_pj = P_j + [v]
                    if len(set(new_pj)) == len(new_pj):
                        new_candidate = [p for idx2, p in enumerate(candidate) if idx2 != idx]
                        new_candidate.append(new_pj)
                        # Accept if it reduces path count or randomly
                        if len(new_candidate) < len(current) or (len(new_candidate) == len(current) and rng.random() < 0.2):
                            current = new_candidate
                            inserted = True
                            break
                
                if P_j[0] == v and (s_j is None or u not in s_j):
                    new_pj = [u] + P_j
                    if len(set(new_pj)) == len(new_pj):
                        new_candidate = [p for idx2, p in enumerate(candidate) if idx2 != idx]
                        new_candidate.append(new_pj)
                        if len(new_candidate) < len(current) or (len(new_candidate) == len(current) and rng.random() < 0.2):
                            current = new_candidate
                            inserted = True
                            break
            
            if not inserted:
                # Keep e as a separate path
                new_candidate = candidate + [[u, v]]
                if len(new_candidate) <= len(current):
                    current = new_candidate
            
            # Try to merge after the move, but limit the time for merging
            # We pass a slightly earlier deadline to merge_paths
            merge_deadline = time.time() + 0.05 # 50ms max for merge
            current = merge_paths(current, edges, n, merge_deadline)
        
        return best if len(best) <= len(current) else current

    def solve_one(n, p, seed, time_limit):
        edges = instance(n, p, seed)
        
        rng = random.Random(seed * 1000 + 7)
        
        best_paths = None
        best_count = float('inf')
        
        # Phase 1: Multiple Randomized Greedy Initializations
        # Reduce restarts to ensure we finish in time. 10 restarts is usually enough for good initialization.
        num_restarts = 10
        
        for r in range(num_restarts):
            if time.time() > time_limit - 10:
                break
            paths = greedy_init(edges, n, rng)
            # Merge immediately
            # Pass a deadline to merge_paths to prevent it from running too long
            merge_deadline = time_limit - 10
            paths = merge_paths(paths, edges, n, merge_deadline)
            if len(paths) < best_count:
                best_count = len(paths)
                best_paths = [list(p) for p in paths]
        
        if best_paths is None:
            best_paths = [list(e) for e in edges]
        
        # Phase 2: Local Search on the best solution
        # Allocate remaining time to local search
        remaining_for_ls = time_limit - time.time() - 5
        if remaining_for_ls > 1:
            best_paths = local_search(best_paths, edges, n, rng, time.time() + remaining_for_ls)
            # Final merge
            best_paths = merge_paths(best_paths, edges, n, time_limit - 2)
        
        # Final Verification
        if not verify_decomposition(best_paths, edges, n):
            best_paths = [list(e) for e in edges]
            
        return best_paths

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))

    # Distribute time evenly among the 3 instances, with a safety margin
    time_per_instance = (deadline - time.time() - 10) / 3.0

    for spec in specs:
        n, p, seed = spec
        remaining_time = deadline - time.time()
        
        # Ensure we don't exceed the total deadline
        instance_limit = min(time_per_instance, remaining_time - 5)
        
        if instance_limit < 5:
            # Not enough time, return trivial solution
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue

        paths = solve_one(n, p, seed, time.time() + instance_limit)
        results.append(paths)

    return results
