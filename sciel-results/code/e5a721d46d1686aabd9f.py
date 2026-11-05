import time
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Find connected components
    visited = [False] * n
    components = []
    for i in range(n):
        if not visited[i]:
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for neighbor in adj[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)
            components.append(comp)
            
    # Sort components by size, largest first
    components.sort(key=len, reverse=True)
    
    best_path = []
    deadline = time.time() + 85.0  # Safety margin for 90s limit
    
    for comp in components:
        if time.time() > deadline:
            break
            
        comp_set = set(comp)
        m_comp = 0
        for u in comp:
            m_comp += len(adj[u] & comp_set)
        m_comp //= 2
        
        if m_comp == len(comp) - 1:
            # Tree case: find diameter
            def bfs_farthest(start, comp_set, adj):
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
                        if v in comp_set and v not in dist:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)
                path = []
                curr = farthest
                while curr is not None:
                    path.append(curr)
                    curr = parent[curr]
                return path[::-1]
            
            start_node = comp[0]
            path1 = bfs_farthest(start_node, comp_set, adj)
            end_node = path1[-1]
            path2 = bfs_farthest(end_node, comp_set, adj)
            
            if len(path2) > len(best_path):
                best_path = path2
                
        elif m_comp == len(comp):
            # Unicyclic graph case
            # Find the cycle
            in_cycle = set()
            parent = {v: -1 for v in comp}
            
            # Find cycle using DFS
            def find_cycle(start, adj, comp_set):
                color = {v: 0 for v in comp_set}  # 0=white, 1=gray, 2=black
                cycle = []
                
                def dfs(u):
                    color[u] = 1
                    for v in sorted(adj[u]):
                        if v not in comp_set:
                            continue
                        if color[v] == 0:
                            parent[v] = u
                            if dfs(v):
                                return True
                        elif color[v] == 1:
                            # Found cycle
                            cycle.append(v)
                            curr = u
                            while curr != v:
                                cycle.append(curr)
                                curr = parent[curr]
                            cycle.append(v)
                            return True
                    color[u] = 2
                    return False
                
                dfs(start)
                return cycle
            
            cycle = find_cycle(comp[0], adj, comp_set)
            
            if not cycle:
                # Fallback to DFS
                pass
            else:
                in_cycle = set(cycle)
                
                # For each cycle node, compute height of tree hanging off it
                heights = {}
                for c in in_cycle:
                    # BFS from c, but don't go into other cycle nodes
                    height = 0
                    parent2 = {c: -1}
                    queue = deque([c])
                    while queue:
                        u = queue.popleft()
                        for v in adj[u]:
                            if v in comp_set and v not in in_cycle and v not in parent2:
                                parent2[v] = u
                                queue.append(v)
                        # Update height
                    # Recompute height properly
                    h = 0
                    for v in parent2:
                        if v == c:
                            continue
                        # Count depth
                        depth = 0
                        curr = v
                        while curr != c:
                            depth += 1
                            curr = parent2[curr]
                        h = max(h, depth)
                    heights[c] = h
                
                # Now try all pairs of cycle nodes
                # The path goes: tree from i -> along cycle -> tree from j
                # Cycle order: let's get it ordered
                # cycle list might not be in order, let's fix that
                cycle_ordered = []
                if len(cycle) > 0:
                    # Find the cycle in order
                    start_c = cycle[0]
                    cycle_ordered = [start_c]
                    curr = start_c
                    for _ in range(len(cycle) - 1):
                        found = False
                        for v in sorted(adj[curr]):
                            if v in in_cycle and v not in cycle_ordered:
                                cycle_ordered.append(v)
                                curr = v
                                found = True
                                break
                        if not found:
                            break
                
                # Try all pairs (i, j) on the cycle
                # The path length = height[i] + (distance along cycle from i to j) + height[j] + 1
                # Actually, the number of vertices = height[i] + dist_cycle(i,j) + height[j] + 1
                # where dist_cycle is the number of edges along the cycle
                
                # Get cycle in a clean order
                if len(cycle_ordered) < 2:
                    cycle_ordered = list(in_cycle)
                
                k = len(cycle_ordered)
                best_len = -1
                best_i, best_j = -1, -1
                best_dir = 1  # 1 or -1 direction along cycle
                
                for i in range(k):
                    for j in range(k):
                        if i == j:
                            continue
                        # Distance from i to j going forward
                        dist_f = (j - i) % k
                        # Distance from i to j going backward
                        dist_b = k - dist_f
                        
                        # Try forward
                        len_f = heights[cycle_ordered[i]] + dist_f + heights[cycle_ordered[j]] + 1
                        if len_f > best_len:
                            best_len = len_f
                            best_i, best_j = i, j
                            best_dir = 1
                        
                        # Try backward
                        len_b = heights[cycle_ordered[i]] + dist_b + heights[cycle_ordered[j]] + 1
                        if len_b > best_len:
                            best_len = len_b
                            best_i, best_j = i, j
                            best_dir = -1
                
                if best_i >= 0:
                    # Reconstruct path
                    ci = cycle_ordered[best_i]
                    cj = cycle_ordered[best_j]
                    
                    # Get the tree path from ci downward
                    def get_tree_path(c, adj, comp_set, in_cycle):
                        # Return list of nodes from c down to the deepest leaf
                        # We need to find the deepest node in the tree rooted at c
                        best_leaf = c
                        best_depth = 0
                        parent3 = {c: -1}
                        queue = deque([c])
                        while queue:
                            u = queue.popleft()
                            for v in adj[u]:
                                if v in comp_set and v not in in_cycle and v not in parent3:
                                    parent3[v] = u
                                    queue.append(v)
                        for v in parent3:
                            if v == c:
                                continue
                            depth = 0
                            curr = v
                            while curr != c:
                                depth += 1
                                curr = parent3[curr]
                            if depth > best_depth:
                                best_depth = depth
                                best_leaf = v
                        
                        # Reconstruct path from c to best_leaf
                        path = []
                        curr = best_leaf
                        while curr != -1:
                            path.append(curr)
                            if curr == c:
                                break
                            curr = parent3[curr]
                        path.reverse()
                        return path
                    
                    tree_i = get_tree_path(ci, adj, comp_set, in_cycle)
                    tree_j = get_tree_path(cj, adj, comp_set, in_cycle)
                    
                    # Get cycle segment
                    if best_dir == 1:
                        # Forward from ci to cj
                        seg = []
                        idx = best_i
                        while True:
                            seg.append(cycle_ordered[idx])
                            if idx == best_j:
                                break
                            idx = (idx + 1) % k
                    else:
                        # Backward from ci to cj
                        seg = []
                        idx = best_i
                        while True:
                            seg.append(cycle_ordered[idx])
                            if idx == best_j:
                                break
                            idx = (idx - 1) % k
                    
                    # Combine: tree_i (excluding ci, which is start of seg) + seg + tree_j (excluding cj, which is end of seg)
                    # tree_i ends with ci, seg starts with ci, so merge
                    # tree_j starts with cj, seg ends with cj, so merge
                    full_path = tree_i[:-1] + seg + tree_j[1:]
                    
                    if len(full_path) > len(best_path):
                        best_path = full_path
                        
        else:
            # More cycles: use multi-strategy search
            # Strategy 1: Beam search with local optimization
            def beam_search(start, adj, comp_set, beam_width=50, max_iter=200):
                """Beam search to find long paths, then local optimization."""
                if time.time() > deadline:
                    return [start]
                    
                # Initialize beam with paths of length 1
                beam = [([start], {start})]
                best_path = [start]
                
                for iteration in range(max_iter):
                    if time.time() > deadline:
                        break
                    
                    new_beam = []
                    for path, visited_set in beam:
                        # Try to extend from both ends
                        for end in [0, -1]:
                            node = path[end]
                            neighbors = [v for v in adj[node] if v in comp_set and v not in visited_set]
                            
                            if not neighbors:
                                continue
                            
                            # Score neighbors by degree (prefer lower degree first - Warnsdorff)
                            neighbors.sort(key=lambda v: len(adj[v] & comp_set))
                            
                            for neighbor in neighbors[:3]:  # Try top 3
                                new_visited = visited_set | {neighbor}
                                if end == 0:
                                    new_path = [neighbor] + path
                                else:
                                    new_path = path + [neighbor]
                                
                                new_beam.append((new_path, new_visited))
                                if len(new_path) > len(best_path):
                                    best_path = new_path[:]
                    
                    if not new_beam:
                        break
                    
                    # Keep top beam_width paths by length
                    new_beam.sort(key=lambda x: len(x[0]), reverse=True)
                    beam = new_beam[:beam_width]
                    
                    # If we can't extend any path, stop
                    if all(len(p) == len(best_path) for p, _ in beam):
                        break
                
                return best_path
            
            # Strategy 2: 2-opt improvement
            def two_opt_improve(path, adj, comp_set):
                """Try to improve path by reversing segments."""
                if time.time() > deadline:
                    return path
                
                improved = True
                best = path[:]
                visited_best = set(best)
                
                while improved and time.time() < deadline:
                    improved = False
                    n_path = len(best)
                    
                    for i in range(n_path - 1):
                        if time.time() > deadline:
                            return best
                        for j in range(i + 2, n_path):
                            # Try reversing segment [i+1..j]
                            # Check if new connections are valid
                            new_path = best[:i+1] + best[i+1:j+1][::-1] + best[j+1:]
                            
                            # Verify it's still a valid path
                            valid = True
                            for k in range(len(new_path) - 1):
                                if new_path[k+1] not in adj[new_path[k]]:
                                    valid = False
                                    break
                            
                            if valid and len(set(new_path)) == len(new_path):
                                # Try to extend from new endpoints
                                extended = False
                                for end_idx in [0, -1]:
                                    node = new_path[end_idx]
                                    for v in adj[node]:
                                        if v in comp_set and v not in set(new_path):
                                            if end_idx == 0:
                                                new_path = [v] + new_path
                                            else:
                                                new_path = new_path + [v]
                                            extended = True
                                            break
                                    if extended:
                                        break
                                
                                if len(new_path) > len(best):
                                    best = new_path
                                    improved = True
                                    break
                        if improved:
                            break
                
                return best
            
            # Strategy 3: Swap improvement (reverse subpath to expose new endpoints)
            def swap_improve(path, adj, comp_set):
                """Try swapping segments to create new extendable endpoints."""
                if time.time() > deadline:
                    return path
                
                best = path[:]
                n_path = len(best)
                improved = True
                
                while improved and time.time() < deadline:
                    improved = False
                    
                    for i in range(n_path - 2):
                        if time.time() > deadline:
                            return best
                        for j in range(i + 2, n_path - 1):
                            # Reverse segment [i+1..j]
                            new_path = best[:i+1] + best[i+1:j+1][::-1] + best[j+1:]
                            
                            # Verify validity
                            valid = True
                            for k in range(len(new_path) - 1):
                                if new_path[k+1] not in adj[new_path[k]]:
                                    valid = False
                                    break
                            
                            if not valid or len(set(new_path)) != len(new_path):
                                continue
                            
                            # Try extending from new endpoints
                            for end_idx in [0, -1]:
                                node = new_path[end_idx]
                                for v in adj[node]:
                                    if v in comp_set and v not in set(new_path):
                                        if end_idx == 0:
                                            new_path = [v] + new_path
                                        else:
                                            new_path = new_path + [v]
                                        if len(new_path) > len(best):
                                            best = new_path
                                            improved = True
                                        break
                                if improved:
                                    break
                        if improved:
                            break
                
                return best
            
            # Run multi-strategy search from multiple starting points
            # Prioritize starting from low-degree vertices (Warnsdorff heuristic)
            sorted_starts = sorted(comp, key=lambda v: len(adj[v] & comp_set))
            
            # First pass: beam search from top candidates
            for start in sorted_starts[:20]:
                if time.time() > deadline:
                    break
                p = beam_search(start, adj, comp_set, beam_width=50, max_iter=100)
                if len(p) > len(best_path):
                    best_path = p
            
            # Second pass: 2-opt and swap improvements on best found
            if len(best_path) > 1:
                improved_path = two_opt_improve(best_path, adj, comp_set)
                if len(improved_path) > len(best_path):
                    best_path = improved_path
                
                improved_path = swap_improve(best_path, adj, comp_set)
                if len(improved_path) > len(best_path):
                    best_path = improved_path
            
            # Third pass: try more starts if time remains
            if time.time() < deadline - 10:
                for start in sorted_starts[20:]:
                    if time.time() > deadline:
                        break
                    p = beam_search(start, adj, comp_set, beam_width=30, max_iter=50)
                    if len(p) > len(best_path):
                        best_path = p
                        # Try to improve
                        improved_path = two_opt_improve(best_path, adj, comp_set)
                        if len(improved_path) > len(best_path):
                            best_path = improved_path
                        improved_path = swap_improve(best_path, adj, comp_set)
                        if len(improved_path) > len(best_path):
                            best_path = improved_path

    if not best_path:
        return [0]
    
    return best_path
# EVOLVE-BLOCK-END
