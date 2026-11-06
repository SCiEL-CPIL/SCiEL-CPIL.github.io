"""Path decomposition of three fixed connected graphs (Gallai's conjecture)."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def time_left():
        return DEADLINE - (time.time() - start_time)

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def build_adjacency(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def path_edges(path):
        """Get the set of edges used by a path."""
        return {(min(path[i], path[i+1]), max(path[i], path[i+1])) for i in range(len(path) - 1)}

    def is_simple(path):
        return len(path) == len(set(path))

    def find_longest_path(adj, start, used_edges, max_len=None):
        """Find a long simple path starting from 'start' using DFS with backtracking (limited)."""
        if max_len is None:
            max_len = 100
        
        best_path = [start]
        best_len = 1
        
        def dfs(current, path, visited):
            nonlocal best_path, best_len
            if len(path) > best_len:
                best_len = len(path)
                best_path = list(path)
            
            if len(path) >= max_len:
                return
            
            # Sort neighbors by degree (Warnsdorff-like)
            neighbors = sorted(
                [v for v in adj[current] 
                 if v not in visited and (min(current, v), max(current, v)) not in used_edges],
                key=lambda v: len([w for w in adj[v] if w not in visited and (min(v, w), max(v, w)) not in used_edges])
            )
            
            for next_v in neighbors:
                edge = (min(current, next_v), max(current, next_v))
                if edge in used_edges:
                    continue
                visited.add(next_v)
                path.append(next_v)
                dfs(next_v, path, visited)
                path.pop()
                visited.remove(next_v)
        
        dfs(start, [start], {start})
        return best_path

    def decompose_greedy_with_merge(edges, adj, time_budget=20.0):
        """
        Decompose edges into paths using greedy longest-path extraction with iterative merging.
        """
        total_edges = len(edges)
        if total_edges == 0:
            return []
        
        used_edges = set()
        paths = []
        
        # Phase 1: Extract longest paths greedily
        vertices = sorted(adj.keys(), key=lambda v: len(adj[v]))  # low-degree first
        
        for start in vertices:
            if time_left() < 2.0:
                break
            if len(used_edges) >= total_edges:
                break
            
            # Find an unused edge incident to start
            incident = [(u, v) for u, v in edges 
                       if (u == start or v == start) and (u, v) not in used_edges]
            if not incident:
                continue
            
            path = find_longest_path(adj, start, used_edges, max_len=50)
            if len(path) > 1:
                paths.append(path)
                used_edges |= path_edges(path)
        
        # Check if all edges covered
        if len(used_edges) < total_edges:
            # Fallback: add remaining edges as single-edge paths
            for e in edges:
                if e not in used_edges:
                    paths.append(list(e))
                    used_edges.add(e)
        
        # Phase 2: Iterative merging
        improved = True
        max_iterations = 100
        iteration = 0
        while improved and iteration < max_iterations and time_left() > 2.0:
            improved = False
            iteration += 1
            for i in range(len(paths)):
                if time_left() < 2.0:
                    break
                for j in range(i + 1, len(paths)):
                    p1, p2 = paths[i], paths[j]
                    
                    # Try all 4 merge combinations
                    merged = None
                    # p1 + p2 (end of p1 = start of p2)
                    if p1[-1] == p2[0]:
                        m = p1 + p2[1:]
                        if is_simple(m):
                            merged = m
                    # p2 + p1 (end of p2 = start of p1)
                    if merged is None and p2[-1] == p1[0]:
                        m = p2 + p1[1:]
                        if is_simple(m):
                            merged = m
                    # reverse p2 + p1 (end of p2 = start of p1, p2 reversed)
                    if merged is None and p2[0] == p1[0]:
                        m = p2[::-1] + p1[1:]
                        if is_simple(m):
                            merged = m
                    # p1 + reverse p2 (end of p1 = end of p2, p2 reversed)
                    if merged is None and p1[-1] == p2[-1]:
                        m = p1 + p2[::-1][1:]
                        if is_simple(m):
                            merged = m
                    
                    if merged is not None:
                        # Verify all edges in merged are valid and in used_edges
                        m_edges = path_edges(merged)
                        if m_edges.issubset(used_edges) and len(m_edges) == len(merged) - 1:
                            paths[i] = merged
                            del paths[j]
                            improved = True
                            break
                if improved:
                    break
        
        return paths

    def decompose_local_search(edges, adj, time_budget=30.0):
        """
        Start with greedy decomposition, then use local search to reduce path count.
        Moves: 
        1. Move an endpoint vertex from one path to another (if it creates a valid merge)
        2. Swap segments between paths
        3. Reorder paths
        """
        # First get a greedy solution
        paths = decompose_greedy_with_merge(edges, adj, time_budget=time_budget * 0.3)
        
        if not paths:
            return paths
        
        best_paths = [list(p) for p in paths]
        best_count = len(paths)
        
        # Local search: try to reduce number of paths by reassigning edges
        # Key insight: if two paths share an endpoint, we might be able to merge them
        # if we can find a different routing for one of the paths
        
        # Try: for each pair of paths that share an endpoint, try to find a longer path
        # that replaces one of them and merges with the other
        for _ in range(50):
            if time_left() < 2.0:
                break
            
            current_paths = [list(p) for p in best_paths]
            improved = False
            
            for i in range(len(current_paths)):
                if time_left() < 2.0:
                    break
                for j in range(i + 1, len(current_paths)):
                    if time_left() < 2.0:
                        break
                    
                    p1, p2 = current_paths[i], current_paths[j]
                    
                    # Check if they can be merged (share endpoint)
                    mergeable = False
                    merged = None
                    
                    if p1[-1] == p2[0]:
                        m = p1 + p2[1:]
                        if is_simple(m):
                            merged = m
                            mergeable = True
                    elif p2[-1] == p1[0]:
                        m = p2 + p1[1:]
                        if is_simple(m):
                            merged = m
                            mergeable = True
                    elif p1[0] == p2[0]:
                        # Both start at same vertex - try to find a longer path from that vertex
                        start_v = p1[0]
                        # Remove both paths' edges, find a new longer path
                        edges_to_remove = path_edges(p1) | path_edges(p2)
                        remaining_used = set()
                        for k, p in enumerate(current_paths):
                            if k != i and k != j:
                                remaining_used |= path_edges(p)
                        
                        new_path = find_longest_path(adj, start_v, remaining_used, max_len=len(p1) + len(p2) + 5)
                        if len(new_path) > 1:
                            new_edges = path_edges(new_path)
                            # Check if new path covers all edges from p1 and p2
                            if new_edges == edges_to_remove:
                                merged = new_path
                                mergeable = True
                    
                    if mergeable and merged is not None:
                        # Verify merged path is valid
                        m_edges = path_edges(merged)
                        all_edges_in_graph = set(edges)
                        if m_edges.issubset(all_edges_in_graph):
                            # Replace p1 and p2 with merged
                            new_paths = []
                            for k, p in enumerate(current_paths):
                                if k == i:
                                    new_paths.append(merged)
                                elif k != j:
                                    new_paths.append(p)
                            
                            # Verify all edges are covered exactly once
                            total_covered = set()
                            valid = True
                            for p in new_paths:
                                pe = path_edges(p)
                                if pe & total_covered:
                                    valid = False
                                    break
                                total_covered |= pe
                            
                            if valid and total_covered == set(edges) and len(new_paths) < len(current_paths):
                                best_paths = new_paths
                                best_count = len(new_paths)
                                improved = True
                                break
                if improved:
                    break
            
            if not improved:
                break
        
        return best_paths

    def decompose_annealing(edges, adj, time_budget=40.0):
        """
        Simulated annealing approach:
        State: assignment of each edge to a path (represented as ordered list of paths)
        Moves: 
        - Move an edge from one path to another (if it maintains validity)
        - Reverse a path
        - Merge two paths if possible
        - Split a path if it helps
        """
        # Initialize with greedy
        paths = decompose_greedy_with_merge(edges, adj, time_budget=time_budget * 0.2)
        
        if not paths:
            return paths
        
        best_paths = [list(p) for p in paths]
        best_count = len(paths)
        
        # Represent state as list of paths
        current_paths = [list(p) for p in best_paths]
        current_count = len(current_paths)
        
        # Temperature schedule
        T = 1.0
        cooling = 0.995
        T_min = 0.01
        
        rng = random.Random(42)
        
        while time_left() > 2.0 and T > T_min:
            T *= cooling
            
            # Pick a random move
            move_type = rng.random()
            
            if move_type < 0.4:
                # Try to merge two random paths
                if len(current_paths) < 2:
                    continue
                i, j = rng.sample(range(len(current_paths)), 2)
                p1, p2 = current_paths[i], current_paths[j]
                
                merged = None
                if p1[-1] == p2[0]:
                    m = p1 + p2[1:]
                    if is_simple(m):
                        merged = m
                elif p2[-1] == p1[0]:
                    m = p2 + p1[1:]
                    if is_simple(m):
                        merged = m
                elif p1[0] == p2[0]:
                    m = p2[::-1] + p1[1:]
                    if is_simple(m):
                        merged = m
                elif p1[-1] == p2[-1]:
                    m = p1 + p2[::-1][1:]
                    if is_simple(m):
                        merged = m
                
                if merged is not None:
                    m_edges = path_edges(merged)
                    if m_edges.issubset(set(edges)):
                        new_paths = []
                        for k, p in enumerate(current_paths):
                            if k == i:
                                new_paths.append(merged)
                            elif k != j:
                                new_paths.append(p)
                        
                        # Verify
                        total_covered = set()
                        valid = True
                        for p in new_paths:
                            pe = path_edges(p)
                            if pe & total_covered:
                                valid = False
                                break
                            total_covered |= pe
                        
                        if valid and total_covered == set(edges):
                            delta = len(new_paths) - len(current_paths)
                            if delta < 0 or rng.random() < (1 if delta == 0 else 0.1):
                                current_paths = new_paths
                                current_count = len(new_paths)
                                if current_count < best_count:
                                    best_count = current_count
                                    best_paths = [list(p) for p in current_paths]
            
            elif move_type < 0.7:
                # Try to extend a path by moving an edge from another path
                if len(current_paths) < 2:
                    continue
                i, j = rng.sample(range(len(current_paths)), 2)
                p1, p2 = current_paths[i], current_paths[j]
                
                # Try to prepend or append an edge from p2 to p1
                # This is complex; skip for now and rely on merge moves
            
            else:
                # Try to find a longer path replacing a short one
                if len(current_paths) < 1:
                    continue
                i = rng.randint(0, len(current_paths) - 1)
                p = current_paths[i]
                if len(p) <= 2:
                    continue
                
                # Remove this path's edges, find a longer path
                p_edges = path_edges(p)
                remaining_used = set()
                for k, pp in enumerate(current_paths):
                    if k != i:
                        remaining_used |= path_edges(pp)
                
                # Try from each endpoint
                for start_v in [p[0], p[-1]]:
                    new_path = find_longest_path(adj, start_v, remaining_used, max_len=len(p) + 10)
                    if len(new_path) > 1:
                        new_edges = path_edges(new_path)
                        if new_edges == p_edges:
                            # Same edges, different routing - no benefit
                            pass
                        elif new_edges.issubset(p_edges):
                            # Subset - not useful
                            pass
                        # For now, only accept if it covers exactly the same edges
                        # (different routing doesn't help reduce count)
        
        return best_paths

    def decompose_graph(edges, time_budget=40.0):
        """Main decomposition function combining multiple strategies."""
        if not edges:
            return []
        
        adj = build_adjacency(edges)
        total_edges = len(edges)
        
        best_paths = None
        best_count = total_edges
        
        # Strategy 1: Greedy with merge
        if time_left() > 5.0:
            paths = decompose_greedy_with_merge(edges, adj, time_budget=min(10.0, time_budget * 0.3))
            if paths and len(paths) < best_count:
                best_count = len(paths)
                best_paths = [list(p) for p in paths]
        
        # Strategy 2: Local search
        if time_left() > 10.0:
            paths = decompose_local_search(edges, adj, time_budget=min(15.0, time_budget * 0.4))
            if paths and len(paths) < best_count:
                best_count = len(paths)
                best_paths = [list(p) for p in paths]
        
        # Strategy 3: Simulated annealing
        if time_left() > 15.0:
            paths = decompose_annealing(edges, adj, time_budget=min(20.0, time_budget * 0.5))
            if paths and len(paths) < best_count:
                best_count = len(paths)
                best_paths = [list(p) for p in paths]
        
        # Fallback: if no valid solution, create one
        if best_paths is None:
            used_edges = set()
            paths = []
            for e in edges:
                u, v = e
                extended = False
                for i, path in enumerate(paths):
                    if path[0] == v:
                        new_path = [u] + path
                        if is_simple(new_path):
                            paths[i] = new_path
                            used_edges.add(e)
                            extended = True
                            break
                    if path[-1] == v:
                        new_path = path + [u]
                        if is_simple(new_path):
                            paths[i] = new_path
                            used_edges.add(e)
                            extended = True
                            break
                    if path[0] == u:
                        new_path = [v] + path
                        if is_simple(new_path):
                            paths[i] = new_path
                            used_edges.add(e)
                            extended = True
                            break
                    if path[-1] == u:
                        new_path = path + [v]
                        if is_simple(new_path):
                            paths[i] = new_path
                            used_edges.add(e)
                            extended = True
                            break
                if not extended:
                    paths.append([u, v])
                    used_edges.add(e)
            best_paths = paths
        
        return best_paths

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for spec in specs:
        if time_left() < 5.0:
            # Not enough time, use simple fallback
            edges = instance(*spec)
            adj = build_adjacency(edges)
            used_edges = set()
            paths = []
            for e in edges:
                u, v = e
                extended = False
                for i, path in enumerate(paths):
                    if path[0] == v:
                        new_path = [u] + path
                        if is_simple(new_path):
                            paths[i] = new_path
                            used_edges.add(e)
                            extended = True
                            break
                    if path[-1] == v:
                        new_path = path + [u]
                        if is_simple(new_path):
                            paths[i] = new_path
                            used_edges.add(e)
                            extended = True
                            break
                    if path[0] == u:
                        new_path = [v] + path
                        if is_simple(new_path):
                            paths[i] = new_path
                            used_edges.add(e)
                            extended = True
                            break
                    if path[-1] == u:
                        new_path = path + [v]
                        if is_simple(new_path):
                            paths[i] = new_path
                            used_edges.add(e)
                            extended = True
                            break
                if not extended:
                    paths.append([u, v])
                    used_edges.add(e)
            results.append(paths)
            continue
        
        edges = instance(*spec)
        paths = decompose_graph(edges, time_budget=40.0)
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
