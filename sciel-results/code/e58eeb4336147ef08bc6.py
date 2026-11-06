import time
import sys
from collections import deque

# Pinned graph data
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]]}

def construct():
    n = P['n']
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    
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
                for nei in adj[node]:
                    if not visited[nei]:
                        visited[nei] = True
                        stack.append(nei)
            components.append(comp)
    
    largest_comp = max(components, key=len)
    if not largest_comp:
        return [0]
        
    comp_set = set(largest_comp)
    comp_size = len(largest_comp)
    
    # Check if tree
    e_count = 0
    for u in largest_comp:
        for v in adj[u]:
            if v in comp_set and u < v:
                e_count += 1
                
    is_tree = (e_count == comp_size - 1)
    
    best_path = []
    best_len = 0
    
    if is_tree:
        # Diameter
        def bfs_farthest(start):
            dist = {start: 0}
            parent = {start: None}
            q = deque([start])
            farthest = start
            max_d = 0
            while q:
                u = q.popleft()
                if dist[u] > max_d:
                    max_d = dist[u]
                    farthest = u
                for v in adj[u]:
                    if v in comp_set and v not in dist:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
            return farthest, dist, parent
        
        u, _, _ = bfs_farthest(largest_comp[0])
        v, _, parent_v = bfs_farthest(u)
        
        path = []
        curr = v
        while curr is not None:
            path.append(curr)
            curr = parent_v[curr]
        path.reverse()
        return path
    else:
        # Dynamic Warnsdorff DFS
        # Precompute static degrees for initial sorting?
        # We will use dynamic unvisited degree.
        
        start_nodes = sorted(largest_comp, key=lambda x: len(adj[x]))
        
        deadline = time.time() + 85
        
        # Use a list for visited to allow O(1) check and modification
        visited_set = set()
        path = []
        
        def dfs(start):
            nonlocal best_path, best_len, deadline
            
            # Reset for this start? No, we pass state.
            # But we want to try multiple starts.
            # We will implement iterative restarts.
            pass

        # Iterative DFS with backtracking is hard to implement recursively with restarts.
        # Let's stick to recursive but call it for multiple starts.
        
        def dfs_recursive(current, path, visited_set):
            nonlocal best_path, best_len
            
            if time.time() > deadline:
                return False # Signal timeout
            
            if len(path) > best_len:
                best_len = len(path)
                best_path = path[:]
                
            # Pruning
            remaining = comp_size - len(visited_set)
            if len(path) + remaining <= best_len:
                return False
            
            curr = path[-1]
            
            # Get neighbors
            neighbors = [v for v in adj[curr] if v in comp_set and v not in visited_set]
            
            if not neighbors:
                return False
            
            # Sort by dynamic degree (number of unvisited neighbors)
            # This is the key improvement
            def unvisited_deg(node):
                count = 0
                for nb in adj[node]:
                    if nb in comp_set and nb not in visited_set:
                        count += 1
                return count
            
            neighbors.sort(key=unvisited_deg)
            
            for v in neighbors:
                path.append(v)
                visited_set.add(v)
                # Recurse
                # We don't need to return True/False for success, just continue
                dfs_recursive(v, path, visited_set)
                path.pop()
                visited_set.remove(v)
                
                if time.time() > deadline:
                    return True # Timeout occurred
                
            return False

        # Try starts
        # We should try starts that are likely to be in the "core" or have high connectivity?
        # Or low degree to force the path through?
        # Let's try a mix.
        
        # To avoid deep recursion overhead, we might limit depth? No, 80 is fine.
        
        for start in start_nodes:
            if time.time() > deadline:
                break
            
            # We can also try to randomize the order of start nodes?
            # But deterministic is better for reproducibility if needed.
            
            # Initialize state
            path = [start]
            visited_set = {start}
            
            # Run DFS
            dfs_recursive(start, path, visited_set)
            
            if best_len == comp_size:
                break # Found Hamiltonian path
        
        if not best_path:
            # Fallback
            best_path = [largest_comp[0]]
            
        return best_path
