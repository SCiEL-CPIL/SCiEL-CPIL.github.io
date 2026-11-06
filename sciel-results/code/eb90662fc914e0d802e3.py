import time
import collections

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
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
                for nei in adj[node]:
                    if not visited[nei]:
                        visited[nei] = True
                        stack.append(nei)
            components.append(comp)
            
    # Find the largest component
    if not components:
        return [0]
        
    largest_comp = max(components, key=len)
    if not largest_comp:
        return [0]
        
    # Check if the largest component is a tree
    num_nodes = len(largest_comp)
    num_edges = sum(len(adj[u]) for u in largest_comp) // 2
    
    best_path = []
    
    # If it's a tree, the longest path is the diameter
    if num_edges == num_nodes - 1:
        # Find diameter
        # BFS from an arbitrary node to find farthest node
        start = largest_comp[0]
        dist = {start: 0}
        parent = {start: None}
        queue = collections.deque([start])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
        # Find the farthest node from start
        end = max(dist, key=dist.get)
        
        # BFS from end to find the diameter path
        dist2 = {end: 0}
        parent2 = {end: None}
        queue = collections.deque([end])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if v not in dist2:
                    dist2[v] = dist2[u] + 1
                    parent2[v] = u
                    queue.append(v)
        
        # Reconstruct path
        path = []
        curr = max(dist2, key=dist2.get)
        while curr is not None:
            path.append(curr)
            curr = parent2[curr]
        path.reverse()
        best_path = path
    else:
        # If not a tree, we need to search for a longer path
        # Use Warnsdorff's rule guided DFS
        # We will run DFS from a few starting points
        # Time limit: 50 seconds
        start_time = time.time()
        time_limit = 50.0
        
        def dfs(node, path, visited_set):
            nonlocal best_path
            if len(path) > len(best_path):
                best_path = path[:]
                
            # Warnsdorff's rule: choose neighbor with fewest unvisited neighbors
            neighbors = [nei for nei in adj[node] if nei not in visited_set]
            if not neighbors:
                return
                
            # Sort neighbors by their number of unvisited neighbors
            neighbors.sort(key=lambda x: len([y for y in adj[x] if y not in visited_set]))
            
            for nei in neighbors:
                if time.time() - start_time > time_limit:
                    return
                visited_set.add(nei)
                path.append(nei)
                dfs(nei, path, visited_set)
                path.pop()
                visited_set.remove(nei)
                
        # Try starting from a few nodes in the largest component
        # To avoid exponential blowup, we limit the depth or use the time limit
        # Since the component is likely small, this should be fast
        # But to be safe, we only try a few starts
        for start_node in largest_comp[:10]:
            if time.time() - start_time > time_limit:
                break
            dfs(start_node, [start_node], {start_node})
            
    return best_path
