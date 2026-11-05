import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find largest connected component
    visited_comp = [False] * n
    components = []
    for i in range(n):
        if not visited_comp[i]:
            comp = []
            stack = [i]
            visited_comp[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for neighbor in adj[node]:
                    if not visited_comp[neighbor]:
                        visited_comp[neighbor] = True
                        stack.append(neighbor)
            components.append(comp)
    
    components.sort(key=len, reverse=True)
    comp = components[0]
    comp_set = set(comp)
    comp_size = len(comp)
    
    # Build adjacency restricted to component
    adj_comp = [adj[u] & comp_set for u in range(n)]
    
    # Check if tree - use BFS diameter
    edge_count = sum(len(adj_comp[u]) for u in comp) // 2
    if edge_count == comp_size - 1:
        def bfs_diameter(start_node):
            dist = {start_node: 0}
            parent = {start_node: None}
            q = [start_node]
            farthest = start_node
            max_dist = 0
            while q:
                u = q.pop(0)
                if dist[u] > max_dist:
                    max_dist = dist[u]
                    farthest = u
                for v in adj_comp[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
            return farthest, parent
        
        u1, parent1 = bfs_diameter(comp[0])
        u2, parent2 = bfs_diameter(u1)
        path = []
        cur = u2
        while cur is not None:
            path.append(cur)
            cur = parent2[cur]
        path.reverse()
        return path
    
    # Use branch-and-bound with connectivity pruning
    best_path = []
    deadline = time.time() + 55.0 # Use more of the 60s budget
    
    # Precompute degrees for Warnsdorff ordering
    deg = [len(adj_comp[u]) for u in range(n)]
    
    # Helper to count reachable unvisited nodes from current
    def count_reachable(current, seen_mask):
        # BFS from current, only visiting unvisited nodes
        visited = set()
        stack = [current]
        visited.add(current)
        while stack:
            node = stack.pop()
            for nb in adj_comp[node]:
                if nb not in seen_mask and nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        return len(visited) - 1 # Exclude current
    
    # Greedy initialization to get a good lower bound
    # Try starting from each node, pick neighbor with min degree (Warnsdorff)
    greedy_best = []
    for start in comp:
        if time.time() > deadline:
            break
        path = [start]
        seen = {start}
        curr = start
        while True:
            nexts = [nb for nb in adj_comp[curr] if nb not in seen]
            if not nexts:
                break
            # Sort by degree (Warnsdorff)
            nexts.sort(key=lambda x: len([nn for nn in adj_comp[x] if nn not in seen]))
            curr = nexts[0]
            path.append(curr)
            seen.add(curr)
        if len(path) > len(greedy_best):
            greedy_best = path[:]
    
    best_path = greedy_best
    
    # DFS with backtracking
    def dfs(current, path, seen_mask, depth):
        nonlocal best_path, deadline
        
        if time.time() > deadline:
            return
        
        if depth > len(best_path):
            best_path = path[:]
            
        if depth == comp_size:
            return
            
        # Pruning: Check max possible extension
        # Reachable unvisited nodes from current
        # If depth + reachable <= len(best_path), prune
        # Note: reachable includes nodes we can potentially visit.
        # We must be able to reach them from current.
        
        # Optimization: Only do full BFS if it might help prune
        # If depth + (comp_size - depth) <= len(best_path), i.e., comp_size <= len(best_path), we already returned or are close.
        # But we need to know if *all* remaining are reachable.
        
        # Quick check: if number of unvisited nodes is small, do BFS.
        # Actually, let's just do BFS if depth + (comp_size - depth) > len(best_path)
        # which is always true unless we found a Hamiltonian path.
        
        # To save time, we can check if the unvisited graph is connected.
        # If not, we can only visit the component connected to current.
        
        # Let's calculate reachable unvisited nodes.
        # This is O(N+E).
        
        # Optimization: If we have visited most nodes, this is fast.
        # If we have visited few, it's slower but prunes more.
        
        # Let's try to optimize by checking if any unvisited node has degree 0 in unvisited graph?
        # That's hard to check quickly without building the subgraph.
        
        # Let's stick to BFS for reachability.
        
        # Get unvisited neighbors
        unvisited_neighbors = []
        for nb in adj_comp[current]:
            if not (seen_mask & (1 << nb)):
                unvisited_neighbors.append(nb)
        
        if not unvisited_neighbors:
            return
        
        # If we can't even reach all unvisited, we might be stuck.
        # But we can still extend the path.
        
        # Pruning check:
        # Max path length = depth + (size of component of unvisited nodes connected to current)
        # Let's compute this size.
        
        # To avoid overhead, only do this if depth + (comp_size - depth) > len(best_path)
        # which is true.
        
        # However, BFS is expensive.
        # Let's only do BFS if we are close to the best or if depth is large.
        # Or always?
        
        # Let's try always, but optimized.
        
        # We can use a bitmask for seen.
        # BFS on unvisited nodes.
        
        # Actually, let's just use the neighbor list for Warnsdorff first, and do the pruning check.
        
        # Pruning:
        # If depth + count_reachable(current, seen_mask) <= len(best_path):
        #     return
        
        # Let's implement count_reachable efficiently.
        
        # Instead of BFS every time, we can just check if the unvisited nodes form a single component?
        # No, we need the size of the component connected to current.
        
        # Let's do the BFS.
        
        # Optimization: If len(best_path) is already comp_size, we are done.
        
        # Let's refine the BFS to stop early if it finds enough nodes?
        # No, we need the total count.
        
        # Let's try to use the fact that if the graph is sparse, BFS is fast.
        
        # Let's do the BFS.
        
        # But wait, if we do BFS at every node, it's O(N * (N+E)) which is 100 * 100 = 10000 ops per path step?
        # Path length ~ 30. 30 * 10000 = 300k ops. This is fine.
        
        # Let's do it.
        
        # Calculate reachable
        reachable_count = 0
        visited_reach = set()
        stack = [current]
        visited_reach.add(current)
        while stack:
            node = stack.pop()
            for nb in adj_comp[node]:
                if not (seen_mask & (1 << nb)) and nb not in visited_reach:
                    visited_reach.add(nb)
                    stack.append(nb)
                    reachable_count += 1
        
        if depth + reachable_count <= len(best_path):
            return
        
        # Warnsdorff ordering
        def warnsdorff_key(nb):
            # Count unvisited neighbors of nb
            options = 0
            for nn in adj_comp[nb]:
                if not (seen_mask & (1 << nn)):
                    options += 1
            return (options, nb)
        
        unvisited_neighbors.sort(key=warnsdorff_key)
        
        for nb in unvisited_neighbors:
            if time.time() > deadline:
                return
            
            new_seen = seen_mask | (1 << nb)
            new_depth = depth + 1
            
            path.append(nb)
            dfs(nb, path, new_seen, new_depth)
            path.pop()
    
    # Try multiple starting points
    # Start with high-degree nodes first? Or low?
    # Warnsdorff usually works well starting from anywhere, but high degree gives more options.
    # Let's sort by degree descending.
    start_order = sorted(comp, key=lambda x: -deg[x])
    
    for start in start_order:
        if time.time() > deadline:
            break
        if comp_size <= len(best_path):
            break
        
        # Reset seen_mask
        seen_mask = 1 << start
        dfs(start, [start], seen_mask, 1)
        
    return best_path if best_path else comp[:1]
# EVOLVE-BLOCK-END
