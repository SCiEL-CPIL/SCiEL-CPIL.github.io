"""Improved path decomposition using randomized greedy + aggressive merging."""
import time
import sys
from collections import defaultdict, deque
import heapq
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def get_adjacency(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def verify_decomposition(paths, edges, n):
        """Verify that paths are valid: simple, use edges, edge-disjoint, cover all edges."""
        all_edges = set(tuple(sorted(e)) for e in edges)
        used_edges = set()
        
        for path in paths:
            if len(set(path)) != len(path):
                return False
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i+1])))
                if e not in all_edges:
                    return False
                if e in used_edges:
                    return False
                used_edges.add(e)
        
        if used_edges != all_edges:
            return False
        
        return True

    def greedy_path_with_warnsdorff(rem_adj, remaining, start):
        """
        Find a long simple path starting from 'start' using Warnsdorff-like heuristic.
        Prefer neighbors with fewer remaining options (to avoid trapping).
        """
        path = [start]
        visited = {start}
        current = start
        used_e = set()
        
        while True:
            extended = False
            # Get neighbors with available edges
            neighbors = []
            for nb in rem_adj[current]:
                e = tuple(sorted((current, nb)))
                if e in remaining and e not in used_e and nb not in visited:
                    # Count how many other options nb has (excluding current)
                    options = 0
                    for nn in rem_adj[nb]:
                        if nn != current and nn not in visited:
                            e2 = tuple(sorted((nb, nn)))
                            if e2 in remaining and e2 not in used_e:
                                options += 1
                    neighbors.append((options, nb))
            
            if not neighbors:
                break
            
            # Sort by fewest options first (Warnsdorff), then by vertex id for determinism
            neighbors.sort(key=lambda x: (x[0], x[1]))
            
            # Try the best few options with some randomization
            try_count = min(3, len(neighbors))
            random.shuffle(neighbors[:try_count])
            
            for _, nb in neighbors[:try_count]:
                e = tuple(sorted((current, nb)))
                if e in remaining and e not in used_e and nb not in visited:
                    path.append(nb)
                    visited.add(nb)
                    used_e.add(e)
                    current = nb
                    extended = True
                    break
            
            if not extended:
                break
        
        return path, used_e

    def decompose_graph(edges, adj, n, time_budget=25):
        """
        Decompose graph into edge-disjoint simple paths.
        Strategy: repeatedly find long paths using randomized greedy with Warnsdorff.
        """
        remaining = set(tuple(sorted(e)) for e in edges)
        paths = []
        
        rem_adj = defaultdict(set)
        for u, v in remaining:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
        
        end_time = time.time() + time_budget
        
        while remaining and time.time() < end_time:
            if not remaining:
                break
            
            degs = {v: len(neigh) for v, neigh in rem_adj.items() if len(neigh) > 0}
            if not degs:
                break
            
            # Choose starting vertices: prefer high degree, but try several
            sorted_vertices = sorted(degs.keys(), key=lambda v: -degs[v])
            
            best_path = None
            best_len = 0
            best_used_e = None
            
            # Try multiple starting points
            num_starts = min(10, len(sorted_vertices))
            for start in sorted_vertices[:num_starts]:
                path, used_e = greedy_path_with_warnsdorff(rem_adj, remaining, start)
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = path
                    best_used_e = used_e
            
            # Also try some random starting points for diversity
            if len(sorted_vertices) > num_starts:
                for _ in range(5):
                    start = random.choice(sorted_vertices)
                    path, used_e = greedy_path_with_warnsdorff(rem_adj, remaining, start)
                    if len(path) > best_len:
                        best_len = len(path)
                        best_path = path
                        best_used_e = used_e
            
            if best_path is None or len(best_path) < 2:
                # Fallback: pick any edge
                e = next(iter(remaining))
                paths.append([e[0], e[1]])
                remaining.remove(e)
                rem_adj[e[0]].discard(e[1])
                rem_adj[e[1]].discard(e[0])
                continue
            
            paths.append(best_path)
            for e in best_used_e:
                if e in remaining:
                    remaining.remove(e)
                    rem_adj[e[0]].discard(e[1])
                    rem_adj[e[1]].discard(e[0])
        
        # Fallback for any remaining edges
        while remaining:
            e = next(iter(remaining))
            paths.append([e[0], e[1]])
            remaining.remove(e)
            rem_adj[e[0]].discard(e[1])
            rem_adj[e[1]].discard(e[0])
        
        return paths

    def merge_paths_aggressive(paths, edges, adj, n, max_rounds=200):
        """
        Aggressively merge paths by finding any pair that can be connected.
        Try all 4 possible endpoint combinations.
        """
        edge_set = set(tuple(sorted(e)) for e in edges)
        current = [list(p) for p in paths]
        
        for _ in range(max_rounds):
            merged = False
            # Shuffle to break symmetry
            random.shuffle(current)
            
            for i in range(len(current)):
                if merged:
                    break
                for j in range(i + 1, len(current)):
                    if merged:
                        break
                    
                    p1, p2 = current[i], current[j]
                    
                    # Check for vertex overlap
                    if set(p1) & set(p2):
                        continue
                    
                    # Try 4 ways to merge
                    candidates = [
                        (p1[-1], p2[0], 'normal'),
                        (p1[0], p2[-1], 'normal'),
                        (p1[-1], p2[-1], 'reverse'),
                        (p1[0], p2[0], 'reverse')
                    ]
                    
                    for a, b, orientation in candidates:
                        e = tuple(sorted((a, b)))
                        if e not in edge_set:
                            continue
                        
                        # Build merged path
                        if orientation == 'normal':
                            if a == p1[-1] and b == p2[0]:
                                merged_path = p1 + p2
                            elif a == p1[0] and b == p2[-1]:
                                merged_path = p2 + p1
                            else:
                                continue
                        else:  # reverse
                            if a == p1[-1] and b == p2[-1]:
                                merged_path = p1 + p2[::-1]
                            elif a == p1[0] and b == p2[0]:
                                merged_path = p1[::-1] + p2
                            else:
                                continue
                        
                        # Verify simplicity
                        if len(set(merged_path)) != len(merged_path):
                            continue
                        
                        # Verify all edges exist
                        valid = True
                        for k in range(len(merged_path) - 1):
                            e2 = tuple(sorted((merged_path[k], merged_path[k+1])))
                            if e2 not in edge_set:
                                valid = False
                                break
                        
                        if valid:
                            # Create new list without i and j
                            new_current = []
                            for idx in range(len(current)):
                                if idx != i and idx != j:
                                    new_current.append(current[idx])
                            new_current.append(merged_path)
                            
                            # Verify full decomposition
                            if verify_decomposition(new_current, edges, n):
                                current = new_current
                                merged = True
                                break
                    
                    if merged:
                        break
                
                if merged:
                    break
            
            if not merged:
                break
        
        return current

    def local_search_improve(paths, edges, n, time_budget=15):
        """
        Try to improve the decomposition by splitting and recombining.
        This is a simple local search: try removing a path and re-adding its edges
        to other paths or creating new shorter paths.
        """
        if len(paths) <= 1:
            return paths
        
        edge_set = set(tuple(sorted(e)) for e in edges)
        current = [list(p) for p in paths]
        best = current[:]
        best_count = len(best)
        
        end_time = time.time() + time_budget
        
        while time.time() < end_time and len(current) > 1:
            # Try to merge any two paths
            merged = False
            random.shuffle(current)
            
            for i in range(len(current)):
                if merged:
                    break
                for j in range(i + 1, len(current)):
                    if merged:
                        break
                    
                    p1, p2 = current[i], current[j]
                    
                    if set(p1) & set(p2):
                        continue
                    
                    candidates = [
                        (p1[-1], p2[0], 'normal'),
                        (p1[0], p2[-1], 'normal'),
                        (p1[-1], p2[-1], 'reverse'),
                        (p1[0], p2[0], 'reverse')
                    ]
                    
                    for a, b, orientation in candidates:
                        e = tuple(sorted((a, b)))
                        if e not in edge_set:
                            continue
                        
                        if orientation == 'normal':
                            if a == p1[-1] and b == p2[0]:
                                merged_path = p1 + p2
                            elif a == p1[0] and b == p2[-1]:
                                merged_path = p2 + p1
                            else:
                                continue
                        else:
                            if a == p1[-1] and b == p2[-1]:
                                merged_path = p1 + p2[::-1]
                            elif a == p1[0] and b == p2[0]:
                                merged_path = p1[::-1] + p2
                            else:
                                continue
                        
                        if len(set(merged_path)) != len(merged_path):
                            continue
                        
                        valid = True
                        for k in range(len(merged_path) - 1):
                            e2 = tuple(sorted((merged_path[k], merged_path[k+1])))
                            if e2 not in edge_set:
                                valid = False
                                break
                        
                        if valid:
                            new_current = []
                            for idx in range(len(current)):
                                if idx != i and idx != j:
                                    new_current.append(current[idx])
                            new_current.append(merged_path)
                            
                            if verify_decomposition(new_current, edges, n):
                                current = new_current
                                if len(current) < best_count:
                                    best = current[:]
                                    best_count = len(best)
                                merged = True
                                break
                    
                    if merged:
                        break
                
                if merged:
                    break
            
            if not merged:
                break
        
        return best

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for spec in specs:
        n, p, seed = spec
        edges = instance(n, p, seed)
        adj = get_adjacency(edges)
        
        remaining_time = deadline - time.time() - 10
        if remaining_time < 5:
            # Not enough time, use simple fallback
            paths = [list(e) for e in edges]
        else:
            # Allocate time budget
            time_budget = min(remaining_time / 3, 30)
            
            # Get initial decomposition
            paths = decompose_graph(edges, adj, n, time_budget)
            
            # Verify
            if not verify_decomposition(paths, edges, n):
                paths = [list(e) for e in edges]
            
            # Aggressive merging
            if len(paths) > 1 and time.time() < deadline - 20:
                merged = merge_paths_aggressive(paths, edges, adj, n, max_rounds=300)
                if verify_decomposition(merged, edges, n):
                    paths = merged
            
            # Local search improvement
            if len(paths) > 1 and time.time() < deadline - 10:
                improved = local_search_improve(paths, edges, n, time_budget=10)
                if verify_decomposition(improved, edges, n):
                    paths = improved
        
        results.append(paths)
        
        if time.time() - start_time > deadline - 10:
            # Run out of time, use fallback for remaining
            for idx in range(len(results), 3):
                n2, p2, seed2 = specs[idx]
                edges2 = instance(n2, p2, seed2)
                results.append([list(e) for e in edges2])
            break
    
    return results
