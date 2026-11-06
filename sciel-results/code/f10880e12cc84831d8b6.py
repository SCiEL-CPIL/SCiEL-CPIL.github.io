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

    def merge_paths(paths, edges, n):
        """
        Iteratively merge paths that can be combined at shared endpoints.
        Two paths can be merged if they share an endpoint and the merged path is simple.
        """
        edge_set = set(tuple(sorted(e)) for e in edges)
        current = [list(p) for p in paths]
        
        improved = True
        max_rounds = 100
        while improved and max_rounds > 0:
            improved = False
            max_rounds -= 1
            
            # Build endpoint map: vertex -> list of (path_index, is_head)
            endpoint_map = defaultdict(list)
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
                        
                        new_path = None
                        
                        # Determine how to merge based on roles
                        # We want to combine p1 and p2 such that they share vertex v
                        # and the result is a simple path.
                        
                        # Case 1: p1 ends at v, p2 starts at v -> p1 + p2[1:]
                        if not is_head1 and is_head2:
                            # p1 tail is v, p2 head is v
                            if not any(x in p1 for x in p2[1:]):
                                new_path = p1 + p2[1:]
                        
                        # Case 2: p1 starts at v, p2 ends at v -> p1[1:] + p2
                        elif is_head1 and not is_head2:
                            # p1 head is v, p2 tail is v
                            if not any(x in p2 for x in p1[1:]):
                                new_path = p1[1:] + p2
                        
                        # Case 3: p1 ends at v, p2 ends at v -> p1 + reversed(p2)[1:]
                        elif not is_head1 and not is_head2:
                            # Both tails at v
                            p2_rev = p2[::-1]
                            if not any(x in p1 for x in p2_rev[1:]):
                                new_path = p1 + p2_rev[1:]
                        
                        # Case 4: p1 starts at v, p2 starts at v -> reversed(p1)[1:] + p2
                        elif is_head1 and is_head2:
                            # Both heads at v
                            p1_rev = p1[::-1]
                            if not any(x in p2 for x in p1_rev[1:]):
                                new_path = p1_rev[1:] + p2
                        
                        if new_path and len(set(new_path)) == len(new_path):
                            # Check validity
                            new_current = []
                            for idx, p in enumerate(current):
                                if idx != idx1 and idx != idx2:
                                    new_current.append(p)
                            new_current.append(new_path)
                            
                            if verify_decomposition(new_current, edges, n):
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
        Move: pick a random edge from a path, remove it, and try to reinsert
        it into another path or keep it as a separate edge-path.
        Accept if the total number of paths decreases or stays the same
        (with some probability to escape local minima).
        """
        current = [list(p) for p in paths]
        best = [list(p) for p in current]
        best_count = len(current)
        
        edge_set = set(tuple(sorted(e)) for e in edges)
        
        # Temperature for simulated annealing
        temperature = 0.1
        min_temperature = 0.001
        
        while time.time() < time_limit:
            if len(current) <= best_count:
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
            e = tuple(sorted((u, v)))
            
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
            for idx, P_j in enumerate(candidate):
                if len(P_j) < 2:
                    continue
                
                # Can we insert edge (u,v) into P_j?
                # We need u and v to be in P_j, and the segment between them
                # in P_j to be "replaceable" by e. But that would remove other edges.
                # 
                # Alternative: extend P_j at an endpoint.
                # If P_j ends at u, we can append v if (u,v) is an edge and v not in P_j.
                # If P_j starts at v, we can prepend u if (u,v) is an edge and u not in P_j.
                
                # Check if we can extend P_j
                if P_j[-1] == u and v not in P_j:
                    # Append v
                    new_pj = P_j + [v]
                    if len(set(new_pj)) == len(new_pj):
                        new_candidate = [p for idx2, p in enumerate(candidate) if idx2 != idx]
                        new_candidate.append(new_pj)
                        if len(new_candidate) < len(current) or (len(new_candidate) == len(current) and rng.random() < 0.3):
                            current = new_candidate
                            inserted = True
                            break
                
                if P_j[0] == v and u not in P_j:
                    # Prepend u
                    new_pj = [u] + P_j
                    if len(set(new_pj)) == len(new_pj):
                        new_candidate = [p for idx2, p in enumerate(candidate) if idx2 != idx]
                        new_candidate.append(new_pj)
                        if len(new_candidate) < len(current) or (len(new_candidate) == len(current) and rng.random() < 0.3):
                            current = new_candidate
                            inserted = True
                            break
                
                # Also try: if P_j has u as an internal vertex, can we split and rejoin?
                # This is complex; skip for now.
            
            if not inserted:
                # Keep e as a separate path
                new_candidate = candidate + [[u, v]]
                if len(new_candidate) <= len(current):
                    current = new_candidate
            
            # Try to merge after the move
            current = merge_paths(current, edges, n)
            
            # Cool down
            temperature = max(min_temperature, temperature * 0.99)
        
        return best if len(best) <= len(current) else current

    def solve_one(n, p, seed, time_limit):
        edges = instance(n, p, seed)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        rng = random.Random(seed * 1000 + 7)
        
        best_paths = None
        best_count = float('inf')
        
        # Phase 1: Multiple Randomized Greedy Initializations
        num_restarts = 30
        per_restart_time = (time_limit - 20) / num_restarts
        
        for r in range(num_restarts):
            if time.time() > time_limit - 10:
                break
            paths = greedy_init(edges, n, rng)
            # Merge immediately
            paths = merge_paths(paths, edges, n)
            if len(paths) < best_count:
                best_count = len(paths)
                best_paths = [list(p) for p in paths]
        
        if best_paths is None:
            best_paths = [list(e) for e in edges]
        
        # Phase 2: Local Search on the best solution
        if time.time() < time_limit - 5:
            best_paths = local_search(best_paths, edges, n, rng, time_limit - 5)
            best_paths = merge_paths(best_paths, edges, n)
        
        # Final Verification
        if not verify_decomposition(best_paths, edges, n):
            best_paths = [list(e) for e in edges]
            
        return best_paths

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))

    for spec in specs:
        n, p, seed = spec
        remaining_time = deadline - time.time()
        if remaining_time < 15:
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue

        paths = solve_one(n, p, seed, time.time() + remaining_time - 5)
        results.append(paths)

    return results
