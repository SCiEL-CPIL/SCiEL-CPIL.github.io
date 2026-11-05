import time
import random
from collections import deque

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}


def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)

    # Find largest connected component
    visited = [False] * n
    components = []
    for i in range(n):
        if not visited[i]:
            stack = [i]
            visited[i] = True
            comp = []
            while stack:
                node = stack.pop()
                comp.append(node)
                for nei in adj[node]:
                    if not visited[nei]:
                        visited[nei] = True
                        stack.append(nei)
            components.append(comp)

    if not components:
        return [0]

    largest_comp = max(components, key=len)
    comp_set = set(largest_comp)
    comp_size = len(largest_comp)

    start_time = time.time()
    time_limit = 85.0

    def is_valid_path(path):
        if not path:
            return False
        s = set(path)
        if len(s) != len(path):
            return False
        for i in range(len(path) - 1):
            if path[i + 1] not in adj[path[i]]:
                return False
        return True

    def get_tree_diameter(comp_nodes):
        """
        If the component is a tree, return the diameter path.
        Otherwise, return None.
        """
        if len(comp_nodes) <= 1:
            return list(comp_nodes)
        
        # Check if it's a tree: edges = nodes - 1
        edge_count = 0
        for u in comp_nodes:
            for v in adj[u]:
                if v in comp_set and u < v:
                    edge_count += 1
        
        if edge_count != len(comp_nodes) - 1:
            return None

        # BFS to find farthest node
        def bfs_farthest(start):
            dist = {start: 0}
            parent = {start: None}
            queue = deque([start])
            farthest = start
            max_dist = 0
            while queue:
                u = queue.popleft()
                if dist[u] > max_dist:
                    max_dist = dist[u]
                    farthest = u
                for v in adj[u]:
                    if v not in dist and v in comp_set:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
            # Reconstruct path
            path = []
            cur = farthest
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return path[::-1]

        # Pick arbitrary start
        start = comp_nodes[0]
        path1 = bfs_farthest(start)
        far1 = path1[-1]
        path2 = bfs_farthest(far1)
        return path2

    # Phase 1: Check if largest component is a tree
    tree_path = get_tree_diameter(largest_comp)
    if tree_path:
        # If it's a tree, this is the optimal longest path
        return tree_path

    # Phase 2: Heuristic Search for Non-Tree Component
    # Use DFS with Warnsdorff's rule and backtracking
    # We allow deeper search now, guarded by time
    
    def dfs_path(start, seen_init):
        """
        DFS to find a long path starting from start.
        """
        best_path = [start]
        seen = set(seen_init)
        seen.add(start)
        current_path = [start]
        
        # Pre-compute neighbors sorted by degree (static heuristic)
        # But we need dynamic "unvisited neighbors" count
        
        def dfs(current, depth):
            nonlocal best_path
            if len(current_path) > len(best_path):
                best_path = list(current_path)
            
            if time.time() - start_time > time_limit - 5:
                return

            # Get unvisited neighbors
            neighbors = [x for x in adj[current] if x not in seen and x in comp_set]
            if not neighbors:
                return
            
            # Warnsdorff: sort by number of unvisited neighbors
            neighbors.sort(key=lambda x: (len([y for y in adj[x] if y not in seen and y in comp_set]), x))
            
            for v in neighbors:
                seen.add(v)
                current_path.append(v)
                dfs(v, depth + 1)
                current_path.pop()
                seen.remove(v)
                
                if time.time() - start_time > time_limit - 5:
                    return

        dfs(start, 1)
        return best_path

    best_path = []
    
    # Try DFS from several promising starts
    # Promising starts: leaves or high degree?
    # In sparse graphs, starting from leaves of the "core" might be good.
    # Let's try all nodes in the component, but limit the number of DFS calls based on time.
    
    # Sort nodes by degree ascending (leaves first)
    nodes_sorted = sorted(largest_comp, key=lambda x: len(adj[x]))
    
    # Try DFS from the first 20 "leaf-like" nodes
    for start in nodes_sorted[:20]:
        if time.time() - start_time > time_limit - 10:
            break
        p = dfs_path(start, set())
        if len(p) > len(best_path):
            best_path = list(p)

    # If we haven't found a path covering most of the component, try more starts
    if len(best_path) < comp_size * 0.8:
        for start in nodes_sorted[20:50]:
            if time.time() - start_time > time_limit - 5:
                break
            p = dfs_path(start, set())
            if len(p) > len(best_path):
                best_path = list(p)

    # Local Refinement: 2-Opt and Insertion
    def try_insert_node(path, seen):
        path_set = set(path)
        candidates = [x for x in comp_set if x not in path_set]
        for u in candidates:
            adj_u = adj[u]
            if not (adj_u & path_set):
                continue
            for i in range(len(path) - 1):
                if path[i] in adj_u and path[i + 1] in adj_u:
                    new_path = path[:i + 1] + [u] + path[i + 1:]
                    return new_path
        return None

    def greedy_extend(path, seen, strategy='min'):
        path = list(path)
        seen = set(seen)
        improved = True
        while improved:
            improved = False
            cur = path[-1]
            neighbors = [x for x in adj[cur] if x not in seen and x in comp_set]
            if neighbors:
                if strategy == 'min':
                    v = min(neighbors, key=lambda x: (len([y for y in adj[x] if y not in seen and y in comp_set]), x))
                else:
                    v = max(neighbors, key=lambda x: (len([y for y in adj[x] if y not in seen and y in comp_set]), x))
                path.append(v)
                seen.add(v)
                improved = True
            cur = path[0]
            neighbors = [x for x in adj[cur] if x not in seen and x in comp_set]
            if neighbors:
                if strategy == 'min':
                    v = min(neighbors, key=lambda x: (len([y for y in adj[x] if y not in seen and y in comp_set]), x))
                else:
                    v = max(neighbors, key=lambda x: (len([y for y in adj[x] if y not in seen and y in comp_set]), x))
                path.insert(0, v)
                seen.add(v)
                improved = True
        return path, seen

    def full_refine(path):
        path = list(path)
        seen = set(path)
        improved = True
        iteration = 0
        while improved and iteration < 50:
            improved = False
            iteration += 1
            
            # Extend
            for strat in ['min', 'max']:
                ext, new_seen = greedy_extend(path, seen, strat)
                if len(ext) > len(path):
                    path = ext
                    seen = new_seen
                    improved = True
                    break

            # Insert
            inserted = True
            while inserted:
                inserted = try_insert_node(path, seen)
                if inserted is not None:
                    path = inserted
                    seen = set(path)
                    improved = True

        return path

    if best_path:
        best_path = full_refine(best_path)

    # If still not good, try random restarts
    rng = random.Random(748)
    for _ in range(100):
        if time.time() - start_time > time_limit - 2:
            break
        start = rng.choice(largest_comp)
        p = dfs_path(start, set())
        p = full_refine(p)
        if len(p) > len(best_path):
            best_path = list(p)

    if not best_path:
        return [0]

    if not is_valid_path(best_path):
        return [largest_comp[0]]

    return best_path
