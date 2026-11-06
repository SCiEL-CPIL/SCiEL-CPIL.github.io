"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
from collections import defaultdict, deque
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    from random import Random

    def instance(n, p, seed):
        rng = Random(seed)
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
            if len(path) < 2:
                return False
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

    def greedy_path_from(start, rem_adj, remaining, rng):
        """Find a long simple path starting from 'start' using randomized greedy."""
        path = [start]
        visited = {start}
        current = start
        
        while True:
            neighbors = [nb for nb in rem_adj[current] 
                        if nb not in visited and tuple(sorted((current, nb))) in remaining]
            if not neighbors:
                break
            # Weight by degree for diversity, but also try random
            if rng.random() < 0.7:
                # Greedy: pick neighbor with highest remaining degree
                neighbors.sort(key=lambda nb: -len(rem_adj[nb]))
                # Among top few, pick randomly for diversity
                top_k = min(3, len(neighbors))
                nb = neighbors[rng.randint(0, top_k - 1)]
            else:
                nb = rng.choice(neighbors)
            path.append(nb)
            visited.add(nb)
            current = nb
        
        return path

    def greedy_trail_decomposition(edges, adj, n, rng, max_time=40):
        """
        Decompose graph into edge-disjoint simple paths using randomized greedy longest-path search.
        """
        remaining = set(tuple(sorted(e)) for e in edges)
        paths = []
        
        rem_adj = defaultdict(set)
        for u, v in remaining:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
        
        while remaining:
            if not remaining:
                break
            
            degs = {v: len(neigh) for v, neigh in rem_adj.items() if neigh}
            if not degs:
                break
            
            # Try multiple starting points and multiple walks per start
            candidates = sorted(degs.keys(), key=lambda v: -degs[v])[:10]
            
            best_path = None
            best_len = 0
            
            for cand_start in candidates:
                if not remaining:
                    break
                
                # Try 3 different random walks from this start
                for _ in range(3):
                    path = greedy_path_from(cand_start, rem_adj, remaining, rng)
                    if len(path) > best_len:
                        best_len = len(path)
                        best_path = path[:]
            
            if best_path is None or len(best_path) < 2:
                e = next(iter(remaining))
                paths.append([e[0], e[1]])
                remaining.remove(e)
                rem_adj[e[0]].discard(e[1])
                rem_adj[e[1]].discard(e[0])
                continue
            
            paths.append(best_path)
            for i in range(len(best_path) - 1):
                e = tuple(sorted((best_path[i], best_path[i+1])))
                if e in remaining:
                    remaining.remove(e)
                    rem_adj[e[0]].discard(e[1])
                    rem_adj[e[1]].discard(e[0])
        
        return paths

    def merge_paths(paths, edges, n, max_iterations=200):
        """Try to merge pairs of paths to reduce total count."""
        edge_set = set(tuple(sorted(e)) for e in edges)
        current = [list(p) for p in paths]
        
        for _ in range(max_iterations):
            merged = False
            n_paths = len(current)
            for i in range(n_paths):
                for j in range(i + 1, n_paths):
                    p1, p2 = current[i], current[j]
                    
                    if set(p1) & set(p2):
                        continue
                    
                    # Try 4 ways to merge
                    attempts = [
                        (p1[-1], p2[0], 'p1_end_to_p2_start'),
                        (p1[0], p2[-1], 'p1_start_to_p2_end'),
                        (p1[-1], p2[-1], 'p1_end_to_p2_end_rev'),
                        (p1[0], p2[0], 'p1_start_to_p2_start_rev'),
                    ]
                    
                    for a, b, mode in attempts:
                        e = tuple(sorted((a, b)))
                        if e not in edge_set:
                            continue
                        
                        if mode == 'p1_end_to_p2_start':
                            merged_path = p1 + p2
                        elif mode == 'p1_start_to_p2_end':
                            merged_path = p2 + p1
                        elif mode == 'p1_end_to_p2_end_rev':
                            merged_path = p1 + p2[::-1]
                        else:  # p1_start_to_p2_start_rev
                            merged_path = p1[::-1] + p2
                        
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

    def improve_paths(paths, edges, n, rng, max_time=15):
        """
        Try to improve the decomposition by:
        1. Re-decomposing subsets of edges to find better paths
        2. Merging more aggressively
        """
        edge_set = set(tuple(sorted(e)) for e in edges)
        current = [list(p) for p in paths]
        
        # Try to find better decompositions by re-extracting paths
        # For each path, try to see if we can split and recombine
        improved = True
        iterations = 0
        while improved and iterations < 30:
            improved = False
            iterations += 1
            
            for i in range(len(current)):
                if len(current[i]) < 3:
                    continue
                
                p = current[i]
                # Try splitting the path at different points and seeing if
                # the pieces can be recombined with other paths
                for split_point in range(1, len(p) - 1):
                    p1 = p[:split_point]
                    p2 = p[split_point:]
                    
                    if len(p1) < 2 or len(p2) < 2:
                        continue
                    
                    # Check if p1 and p2 are valid paths
                    if len(set(p1)) != len(p1) or len(set(p2)) != len(p2):
                        continue
                    
                    # Check edges
                    valid_p1 = all(tuple(sorted((p1[k], p1[k+1]))) in edge_set for k in range(len(p1)-1))
                    valid_p2 = all(tuple(sorted((p2[k], p2[k+1]))) in edge_set for k in range(len(p2)-1))
                    
                    if not valid_p1 or not valid_p2:
                        continue
                    
                    # Replace path i with p1 and p2, then try to merge
                    new_current = []
                    for idx in range(len(current)):
                        if idx == i:
                            new_current.append(p1)
                            new_current.append(p2)
                        else:
                            new_current.append(current[idx])
                    
                    # Try merging
                    merged = merge_paths(new_current, edges, n, max_iterations=20)
                    
                    if len(merged) < len(current):
                        current = merged
                        improved = True
                        break
            
            if improved:
                break
        
        return current

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for spec in specs:
        n, p, seed = spec
        edges = instance(n, p, seed)
        adj = get_adjacency(edges)
        
        # Allocate time based on remaining budget
        elapsed = time.time() - start_time
        remaining_time = deadline - elapsed - 10  # 10s safety margin
        if remaining_time <= 0:
            # Fallback: each edge as a path
            results.append([list(e) for e in edges])
            continue
        
        # Use a deterministic RNG seeded from the graph parameters for reproducibility
        rng = Random(seed * 1000 + n)
        
        # Get initial decomposition with time limit
        time_per_graph = min(remaining_time / max(1, 3 - len(results)), 45)
        paths = greedy_trail_decomposition(edges, adj, n, rng, max_time=time_per_graph)
        
        if not verify_decomposition(paths, edges, n):
            paths = [list(e) for e in edges]
        
        # Try to merge paths to reduce count
        if len(paths) > 1 and time.time() - start_time < deadline - 15:
            merged = merge_paths(paths, edges, n, max_iterations=100)
            if verify_decomposition(merged, edges, n):
                paths = merged
        
        # Try improvement if time allows
        if len(paths) > 2 and time.time() - start_time < deadline - 20:
            improved = improve_paths(paths, edges, n, rng, max_time=10)
            if verify_decomposition(improved, edges, n) and len(improved) <= len(paths):
                paths = improved
        
        results.append(paths)
    
    # Final verification and fallback
    for idx in range(len(results)):
        n, p, seed = specs[idx]
        edges = instance(n, p, seed)
        if not verify_decomposition(results[idx], edges, n):
            results[idx] = [list(e) for e in edges]
    
    return results
# EVOLVE-BLOCK-END
