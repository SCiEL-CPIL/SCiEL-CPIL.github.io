import time
from collections import defaultdict

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    start_time = time.time()
    deadline = 55.0 # seconds
    
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
            
    # Sort components by size descending
    components.sort(key=len, reverse=True)
    
    best_path = []
    
    for comp in components:
        if time.time() > deadline:
            break
            
        comp_set = set(comp)
        # Count edges in component
        edge_count = 0
        for u in comp:
            for v in adj[u]:
                if v in comp_set and u < v:
                    edge_count += 1
                    
        if len(comp) <= 20:
            # If it's a tree, find diameter
            if edge_count == len(comp) - 1:
                # Find diameter
                if len(comp) == 1:
                    path = [comp[0]]
                else:
                    # BFS from arbitrary node
                    start = comp[0]
                    dist = {start: 0}
                    parent = {start: None}
                    queue = [start]
                    while queue:
                        node = queue.pop(0)
                        for nei in adj[node]:
                            if nei in comp_set and nei not in dist:
                                dist[nei] = dist[node] + 1
                                parent[nei] = node
                                queue.append(nei)
                    # Find farthest
                    far1 = max(dist, key=dist.get)
                    # BFS from far1
                    dist2 = {far1: 0}
                    parent2 = {far1: None}
                    queue = [far1]
                    while queue:
                        node = queue.pop(0)
                        for nei in adj[node]:
                            if nei in comp_set and nei not in dist2:
                                dist2[nei] = dist2[node] + 1
                                parent2[nei] = node
                                queue.append(nei)
                    far2 = max(dist2, key=dist2.get)
                    # Reconstruct path
                    path = []
                    cur = far2
                    while cur is not None:
                        path.append(cur)
                        cur = parent2[cur]
                    path.reverse()
                
                if len(path) > len(best_path):
                    best_path = path
            else:
                # Small non-tree, use DFS
                def dfs_small(path, seen):
                    nonlocal best_path
                    if len(path) > len(best_path):
                        best_path = path.copy()
                    for nei in adj[path[-1]]:
                        if nei in comp_set and nei not in seen:
                            seen.add(nei)
                            path.append(nei)
                            dfs_small(path, seen)
                            path.pop()
                            seen.remove(nei)
                            
                for start in comp:
                    if time.time() > deadline: break
                    dfs_small([start], {start})
                    
        else:
            # Heuristic search for large component
            deg_in_comp = {v: len([u for u in adj[v] if u in comp_set]) for v in comp}
            
            def check_connectivity(current, unvisited, adj, comp_set):
                if not unvisited:
                    return True
                visited = set()
                stack = [current]
                visited.add(current)
                while stack:
                    node = stack.pop()
                    for nei in adj[node]:
                        if nei in unvisited and nei not in visited:
                            visited.add(nei)
                            stack.append(nei)
                for u in unvisited:
                    if u not in visited:
                        return False
                return True

            def dfs_heuristic(path, seen):
                nonlocal best_path
                if len(path) > len(best_path):
                    best_path = path.copy()
                
                current = path[-1]
                unvisited = comp_set - seen
                if not unvisited:
                    return
                    
                # Get unvisited neighbors
                unvisited_neis = [u for u in adj[current] if u in unvisited]
                
                if not unvisited_neis:
                    return
                    
                # Sort by degree in unvisited graph (Warnsdorff)
                def sort_key(node):
                    cnt = 0
                    for u in adj[node]:
                        if u in unvisited:
                            cnt += 1
                    return cnt
                
                unvisited_neis.sort(key=sort_key)
                
                for next_node in unvisited_neis:
                    if time.time() > deadline:
                        return
                    seen.add(next_node)
                    path.append(next_node)
                    
                    # Check connectivity
                    if check_connectivity(next_node, comp_set - seen, adj, comp_set):
                        dfs_heuristic(path, seen)
                        
                    path.pop()
                    seen.remove(next_node)
                    
            start_nodes = sorted(comp, key=lambda x: deg_in_comp[x])
            
            for start in start_nodes:
                if time.time() > deadline:
                    break
                dfs_heuristic([start], {start})
                
    return best_path
