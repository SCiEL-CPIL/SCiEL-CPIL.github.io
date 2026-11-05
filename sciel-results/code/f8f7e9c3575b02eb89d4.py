import time
import random
from collections import defaultdict
from itertools import combinations

def construct():
    """
    Decomposes the edges of three fixed instance graphs into the minimum number of
    edge-disjoint simple paths.
    
    Returns:
        A list of three lists of paths. Each path is a list of vertices.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def validate_paths(paths, edges):
        """Validate that paths are simple, edge-disjoint, and cover all edges."""
        if not paths:
            return len(edges) == 0
        
        edge_set = set(edges)
        used_edges = set()
        
        for path in paths:
            if len(path) < 2:
                continue
            # Check simple path
            if len(set(path)) != len(path):
                return False
            
            # Check edges exist and are not reused
            for j in range(len(path) - 1):
                u, v = path[j], path[j+1]
                e = tuple(sorted((u, v)))
                if e not in edge_set:
                    return False
                if e in used_edges:
                    return False
                used_edges.add(e)
        
        # Check all edges are covered
        if used_edges != edge_set:
            return False
        
        return True

    def max_matching_bipartite(left_nodes, right_nodes, adjacency):
        """
        Find maximum bipartite matching using Hopcroft-Karp algorithm.
        left_nodes, right_nodes: lists of node IDs
        adjacency: dict mapping left node -> set of right nodes
        Returns: list of (left, right) pairs in the matching
        """
        if not left_nodes or not right_nodes:
            return []
        
        # Build node index maps
        left_index = {node: i for i, node in enumerate(left_nodes)}
        right_index = {node: i for i, node in enumerate(right_nodes)}
        
        n_left = len(left_nodes)
        n_right = len(right_nodes)
        
        # Build adjacency list in indexed form
        adj = [[] for _ in range(n_left)]
        for u in left_nodes:
            ui = left_index[u]
            for v in adjacency.get(u, set()):
                if v in right_index:
                    adj[ui].append(right_index[v])
        
        # Hopcroft-Karp
        match_left = [-1] * n_left
        match_right = [-1] * n_right
        dist = [0] * n_left
        
        def bfs():
            queue = []
            for u in range(n_left):
                if match_left[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = float('inf')
            
            found = False
            while queue:
                u = queue.pop(0)
                for v in adj[u]:
                    if match_right[v] == -1:
                        found = True
                    elif dist[match_right[v]] == float('inf'):
                        dist[match_right[v]] = dist[u] + 1
                        queue.append(match_right[v])
            return found
        
        def dfs(u):
            for v in adj[u]:
                if match_right[v] == -1 or (dist[match_right[v]] == dist[u] + 1 and dfs(match_right[v])):
                    match_left[u] = v
                    match_right[v] = u
                    return True
            dist[u] = float('inf')
            return False
        
        matching = []
        while bfs():
            for u in range(n_left):
                if match_left[u] == -1:
                    if dfs(u):
                        pass
        
        for u in range(n_left):
            if match_left[u] != -1:
                matching.append((left_nodes[u], right_nodes[match_left[u]]))
        
        return matching

    def greedy_initial_decomposition(edges, seed):
        """
        Create an initial path decomposition using greedy approach.
        Returns list of paths.
        """
        rng = random.Random(seed)
        remaining = set(edges)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        paths = []
        
        while remaining:
            # Pick a random edge to start
            edge = rng.choice(list(remaining))
            remaining.remove(edge)
            u, v = edge
            path = [u, v]
            
            # Extend the path greedily
            changed = True
            while changed:
                changed = False
                # Try extending from both ends
                for end in [0, 1]:  # 0 = front, 1 = back
                    if end == 0:
                        current = path[0]
                        candidates = [nb for nb in adj[current] 
                                      if nb not in path and ((current, nb) in remaining or (nb, current) in remaining)]
                        if candidates:
                            nb = rng.choice(candidates)
                            if (current, nb) in remaining:
                                remaining.remove((current, nb))
                            else:
                                remaining.remove((nb, current))
                            path.insert(0, nb)
                            changed = True
                    else:
                        current = path[-1]
                        candidates = [nb for nb in adj[current] 
                                      if nb not in path and ((current, nb) in remaining or (nb, current) in remaining)]
                        if candidates:
                            nb = rng.choice(candidates)
                            if (current, nb) in remaining:
                                remaining.remove((current, nb))
                            else:
                                remaining.remove((nb, current))
                            path.append(nb)
                            changed = True
                
                if not changed:
                    break
            
            paths.append(path)
        
        return paths

    def merge_paths_with_matching(paths, edges, adj, used_edges, max_iterations=200):
        """
        Iteratively merge path endpoints using maximum matching.
        paths: list of paths (each is a list of vertices)
        edges: set of all edges
        adj: adjacency dict
        used_edges: set of edges already used in paths
        Returns: merged list of paths
        """
        if not paths:
            return []
        
        current_paths = [list(p) for p in paths]
        current_used = set(used_edges)
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 5:
                break
            
            # Build auxiliary graph for matching
            # Each path has two endpoints: first and last vertex
            # We can match endpoint i of path A with endpoint j of path B if:
            # 1. There's an unused edge between them
            # 2. They're from different paths
            # 3. The merged path is simple
            
            node_to_path = {}
            for i, path in enumerate(current_paths):
                if len(path) >= 2:
                    node_to_path[(i, 0)] = i
                    node_to_path[(i, 1)] = i
            
            nodes = list(node_to_path.keys())
            
            if len(nodes) < 4:  # Need at least 2 paths with 2 endpoints each
                break
            
            # Build adjacency for matching - only include edges that result in simple merged paths
            match_adj = defaultdict(set)
            
            for a, b in combinations(nodes, 2):
                path_a, pos_a = a
                path_b, pos_b = b
                
                # Skip if same path
                if path_a == path_b:
                    continue
                
                vertex_a = current_paths[path_a][0] if pos_a == 0 else current_paths[path_a][-1]
                vertex_b = current_paths[path_b][0] if pos_b == 0 else current_paths[path_b][-1]
                
                # Check if there's an unused edge between them
                e = tuple(sorted((vertex_a, vertex_b)))
                if e in current_used:
                    continue
                if e not in edges:
                    continue
                
                # Check if connecting would create a cycle (i.e., if the two paths share any vertices)
                set_a = set(current_paths[path_a])
                set_b = set(current_paths[path_b])
                if set_a & set_b:
                    continue
                
                match_adj[a].add(b)
                match_adj[b].add(a)
            
            # Find maximum matching using greedy approach with prioritization
            matched = set()
            matches = []
            
            # Sort nodes by degree (descending) for better matching
            nodes_sorted = sorted(nodes, key=lambda x: len(match_adj[x]), reverse=True)
            
            for node in nodes_sorted:
                if node in matched:
                    continue
                # Sort neighbors by degree (descending)
                for neighbor in sorted(match_adj[node], key=lambda x: len(match_adj[x]), reverse=True):
                    if neighbor in matched:
                        continue
                    # Check if this match is valid
                    path_a, pos_a = node
                    path_b, pos_b = neighbor
                    
                    vertex_a = current_paths[path_a][0] if pos_a == 0 else current_paths[path_a][-1]
                    vertex_b = current_paths[path_b][0] if pos_b == 0 else current_paths[path_b][-1]
                    
                    e = tuple(sorted((vertex_a, vertex_b)))
                    if e in current_used:
                        continue
                    if e not in edges:
                        continue
                    
                    # Check simplicity
                    set_a = set(current_paths[path_a])
                    set_b = set(current_paths[path_b])
                    if set_a & set_b:
                        continue
                    
                    # Perform the merge
                    matched.add(node)
                    matched.add(neighbor)
                    matches.append((node, neighbor, e))
                    break
            
            if not matches:
                break
            
            # Apply all matches
            new_paths = {}
            for path_idx, path in enumerate(current_paths):
                new_paths[path_idx] = list(path)
            
            removed_paths = set()
            
            for (path_a, pos_a), (path_b, pos_b), edge in matches:
                if path_a in removed_paths or path_b in removed_paths:
                    continue
                
                vertex_a = new_paths[path_a][0] if pos_a == 0 else new_paths[path_a][-1]
                vertex_b = new_paths[path_b][0] if pos_b == 0 else new_paths[path_b][-1]
                
                # Determine how to merge
                # We need to connect the two paths using the edge (vertex_a, vertex_b)
                
                # Get the two paths
                path1 = new_paths[path_a]
                path2 = new_paths[path_b]
                
                # Determine orientation
                # We want to connect endpoint pos_a of path1 to endpoint pos_b of path2
                
                # Reverse path1 if pos_a == 1 (we want to connect from the front)
                if pos_a == 1:
                    path1 = list(reversed(path1))
                
                # Reverse path2 if pos_b == 1 (we want to connect from the back)
                if pos_b == 1:
                    path2 = list(reversed(path2))
                
                # Now we're connecting path1[-1] to path2[0]
                # Verify they are the correct vertices
                if path1[-1] != vertex_a or path2[0] != vertex_b:
                    # This shouldn't happen, but skip if it does
                    continue
                
                # Merge
                merged = path1 + path2[1:]
                
                # Check if the merged path is simple
                if len(set(merged)) != len(merged):
                    # Skip this merge
                    continue
                
                # Check if all edges in the merged path are valid and unused
                valid = True
                for j in range(len(merged) - 1):
                    e = tuple(sorted((merged[j], merged[j+1])))
                    if e not in edges:
                        valid = False
                        break
                    if e in current_used and e != edge:
                        # This edge was already used, which is a problem
                        # Actually, we need to check if this edge is part of one of the original paths
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Update
                new_paths[path_a] = merged
                del new_paths[path_b]
                removed_paths.add(path_b)
                current_used.add(edge)
            
            # Rebuild current_paths
            current_paths = [new_paths[i] for i in sorted(new_paths.keys())]
            
            if len(current_paths) == len(paths):
                break
        
        return current_paths

    def local_search_improve(paths, edges, adj, used_edges, max_iterations=50):
        """
        Try to reduce path count by splitting and recombining paths.
        """
        if not paths:
            return []
        
        current_paths = [list(p) for p in paths]
        current_used = set(used_edges)
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 5:
                break
            
            improved = False
            
            # Try to find two paths that can be split and recombined to reduce total count
            for i in range(len(current_paths)):
                for j in range(i + 1, len(current_paths)):
                    path_i = current_paths[i]
                    path_j = current_paths[j]
                    
                    if len(path_i) < 3 or len(path_j) < 3:
                        continue
                    
                    # Try splitting path_i at different positions
                    for split_i in range(1, len(path_i) - 1):
                        part1 = path_i[:split_i + 1]
                        part2 = path_i[split_i:]
                        
                        # Try splitting path_j at different positions
                        for split_j in range(1, len(path_j) - 1):
                            part3 = path_j[:split_j + 1]
                            part4 = path_j[split_j:]
                            
                            # Try all four combinations of merging
                            combinations_to_try = [
                                (part1, part3, part2, part4),
                                (part1, part4, part2, part3),
                                (part2, part3, part1, part4),
                                (part2, part4, part1, part3),
                            ]
                            
                            for a, b, c, d in combinations_to_try:
                                # Check if we can merge a and b, and c and d
                                if len(a) < 2 or len(b) < 2 or len(c) < 2 or len(d) < 2:
                                    continue
                                
                                # Check edge between a[-1] and b[0]
                                e1 = tuple(sorted((a[-1], b[0])))
                                if e1 not in edges or e1 in current_used:
                                    continue
                                
                                # Check edge between c[-1] and d[0]
                                e2 = tuple(sorted((c[-1], d[0])))
                                if e2 not in edges or e2 in current_used:
                                    continue
                                
                                # Check simplicity
                                merged1 = a + b[1:]
                                merged2 = c + d[1:]
                                
                                if len(set(merged1)) != len(merged1) or len(set(merged2)) != len(merged2):
                                    continue
                                
                                # Check that all edges are valid
                                valid = True
                                for merged in [merged1, merged2]:
                                    for k in range(len(merged) - 1):
                                        e = tuple(sorted((merged[k], merged[k+1])))
                                        if e not in edges:
                                            valid = False
                                            break
                                if not valid:
                                    continue
                                
                                # This is a valid improvement - we're replacing 2 paths with 2 paths
                                # but potentially with different structure that allows further merges
                                # For now, just accept if it doesn't increase path count
                                improved = True
                                break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                
                if improved:
                    break
            
            if not improved:
                break
        
        return current_paths

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        if not edges:
            return []
        
        # Build adjacency and degrees
        adj = defaultdict(set)
        degrees = defaultdict(int)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            degrees[u] += 1
            degrees[v] += 1
            
        odd_vertices = [v for v in degrees if degrees[v] % 2 != 0]
        num_odd = len(odd_vertices)
        
        # Theoretical Lower Bound
        if num_odd == 0:
            lb = 1 if len(edges) == 1 else 2
        else:
            lb = num_odd // 2
        
        best_paths = None
        best_count = float('inf')
        
        # Strategy: Try multiple random initial decompositions, then merge with matching
        max_attempts = 300
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            seed_val = seed * 10000 + attempt
            initial_paths = greedy_initial_decomposition(edges, seed_val)
            
            if not initial_paths:
                continue
            
            # Track used edges
            used_edges = set()
            for path in initial_paths:
                for j in range(len(path) - 1):
                    u, v = path[j], path[j+1]
                    e = tuple(sorted((u, v)))
                    used_edges.add(e)
            
            # Merge with matching
            merged_paths = merge_paths_with_matching(initial_paths, edges, adj, used_edges)
            
            # Validate
            if validate_paths(merged_paths, edges):
                if len(merged_paths) < best_count:
                    best_count = len(merged_paths)
                    best_paths = merged_paths
                    
                    # If we reached the lower bound, we can stop
                    if len(merged_paths) <= lb:
                        break
        
        # Fallback to simple greedy if matching approach didn't work well
        if best_paths is None:
            for i in range(100):
                if time.time() > deadline - 5:
                    break
                paths = greedy_initial_decomposition(edges, seed * 10000 + 10000 + i)
                if validate_paths(paths, edges):
                    if len(paths) < best_count:
                        best_count = len(paths)
                        best_paths = paths
            
            if best_paths is None:
                best_paths = [[u, v] for u, v in edges]
                best_count = len(best_paths)

        return best_paths

    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    results = []
    
    for spec in specs:
        n, p, seed = spec
        # Check time before starting instance
        if time.time() > deadline - 10:
            # Not enough time, use trivial decomposition for remaining
            edges = instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
            
        paths = solve_instance(n, p, seed)
        results.append(paths)
        
    # Fallback
    while len(results) < 3:
        n, p, seed = specs[len(results)]
        edges = instance(n, p, seed)
        results.append([[u, v] for u, v in edges])

    return results

if __name__ == "__main__":
    res = construct()
    total = sum(len(r) for r in res)
    print(f"Total paths: {total}")
    for i, r in enumerate(res):
        print(f"Instance {i}: {len(r)} paths")
