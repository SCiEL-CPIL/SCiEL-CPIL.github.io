import time
import random
from collections import defaultdict
from itertools import combinations
from collections import deque

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

    def merge_paths_with_matching(paths, edges, adj, used_edges, max_iterations=100):
        """
        Iteratively merge path endpoints using maximum matching on a bipartite representation.
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
            # 3. The merge would result in a simple path
            
            endpoints = []  # (path_idx, endpoint_pos, vertex)
            for i, path in enumerate(current_paths):
                if len(path) >= 2:
                    endpoints.append((i, 0, path[0]))
                    endpoints.append((i, 1, path[-1]))
            
            if len(endpoints) < 4:  # Need at least 2 paths with 2 endpoints each
                break
            
            # Build bipartite graph: left = endpoints from even-indexed paths, right = from odd-indexed
            # This ensures we don't match endpoints from the same path
            left_nodes = []
            right_nodes = []
            left_to_endpoint = {}
            right_to_endpoint = {}
            
            for i, path in enumerate(current_paths):
                if len(path) < 2:
                    continue
                for pos in [0, 1]:
                    endpoint = (i, pos)
                    vertex = path[0] if pos == 0 else path[-1]
                    if i % 2 == 0:
                        left_nodes.append((i, pos))
                        left_to_endpoint[(i, pos)] = vertex
                    else:
                        right_nodes.append((i, pos))
                        right_to_endpoint[(i, pos)] = vertex
            
            if not left_nodes or not right_nodes:
                break
            
            # Build adjacency: left endpoint -> set of right endpoints
            adjacency = defaultdict(set)
            for (i1, pos1) in left_nodes:
                vertex1 = current_paths[i1][0] if pos1 == 0 else current_paths[i1][-1]
                for (i2, pos2) in right_nodes:
                    if i1 == i2:
                        continue
                    vertex2 = current_paths[i2][0] if pos2 == 0 else current_paths[i2][-1]
                    e = tuple(sorted((vertex1, vertex2)))
                    if e not in edges or e in current_used:
                        continue
                    # Check if merge would be simple
                    path1_set = set(current_paths[i1])
                    path2_set = set(current_paths[i2])
                    if path1_set & path2_set:
                        continue
                    adjacency[(i1, pos1)].add((i2, pos2))
            
            if not adjacency:
                break
            
            # Find maximum matching
            matching = max_matching_bipartite(left_nodes, right_nodes, adjacency)
            
            if not matching:
                break
            
            # Apply matches one by one, checking validity
            new_paths = {i: list(p) for i, p in enumerate(current_paths)}
            removed_paths = set()
            applied_count = 0
            
            for (left_ep, right_ep) in matching:
                i1, pos1 = left_ep
                i2, pos2 = right_ep
                
                if i1 in removed_paths or i2 in removed_paths:
                    continue
                
                path1 = new_paths[i1]
                path2 = new_paths[i2]
                
                vertex1 = path1[0] if pos1 == 0 else path1[-1]
                vertex2 = path2[0] if pos2 == 0 else path2[-1]
                
                e = tuple(sorted((vertex1, vertex2)))
                if e not in edges or e in current_used:
                    continue
                
                # Determine orientation for merge
                # We want to connect vertex1 (at pos1 of path1) to vertex2 (at pos2 of path2)
                
                # Reverse path1 if pos1 == 1 (so vertex1 is at the end)
                p1 = list(reversed(path1)) if pos1 == 1 else list(path1)
                # Reverse path2 if pos2 == 1 (so vertex2 is at the start)
                p2 = list(reversed(path2)) if pos2 == 1 else list(path2)
                
                # Now p1[-1] should be vertex1 and p2[0] should be vertex2
                if p1[-1] != vertex1 or p2[0] != vertex2:
                    continue
                
                merged = p1 + p2[1:]
                
                # Check simplicity
                if len(set(merged)) != len(merged):
                    continue
                
                # Check all edges are valid
                valid = True
                for j in range(len(merged) - 1):
                    ej = tuple(sorted((merged[j], merged[j+1])))
                    if ej not in edges:
                        valid = False
                        break
                    if ej in current_used and ej != e:
                        valid = False
                        break
                if not valid:
                    continue
                
                new_paths[i1] = merged
                del new_paths[i2]
                removed_paths.add(i2)
                current_used.add(e)
                applied_count += 1
            
            if applied_count == 0:
                break
            
            current_paths = [new_paths[i] for i in sorted(new_paths.keys())]
            
            if len(current_paths) >= len(paths):
                break
        
        return current_paths

    def local_search_improve(paths, edges, adj, used_edges, max_iterations=50):
        """
        Try to further reduce path count by local search:
        - Try swapping segments between paths
        - Try re-routing path endpoints
        """
        if not paths:
            return []
        
        current_paths = [list(p) for p in paths]
        current_used = set(used_edges)
        best_count = len(current_paths)
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 5:
                break
            
            improved = False
            
            # Try to find two paths that can be merged via a different connection
            # by removing an edge from one path and adding it to another
            for i in range(len(current_paths)):
                for j in range(i + 1, len(current_paths)):
                    if time.time() > deadline - 5:
                        break
                    
                    path_i = current_paths[i]
                    path_j = current_paths[j]
                    
                    # Try removing an edge from path_i and reconnecting endpoints
                    # This is complex; skip for now and rely on the matching approach
            
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
        max_attempts = 500
        
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
