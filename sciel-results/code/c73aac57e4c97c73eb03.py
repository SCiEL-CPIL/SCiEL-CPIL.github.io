"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import time

sys.setrecursionlimit(2000)

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # 1. Find Largest Connected Component
    visited_comp = [False] * n
    lcc = []
    lcc_size = 0
    for i in range(n):
        if not visited_comp[i]:
            stack = [i]
            visited_comp[i] = True
            comp = []
            while stack:
                node = stack.pop()
                comp.append(node)
                for nei in adj[node]:
                    if not visited_comp[nei]:
                        visited_comp[nei] = True
                        stack.append(nei)
            if len(comp) > lcc_size:
                lcc_size = len(comp)
                lcc = comp[:]
                
    if not lcc:
        return [0]
        
    lcc_set = set(lcc)
    
    # 2. Check if LCC is a tree
    edges_in_lcc = 0
    for u in lcc:
        for v in adj[u]:
            if v in lcc_set and u < v:
                edges_in_lcc += 1
                
    if edges_in_lcc == lcc_size - 1:
        # It's a tree. Longest path is the diameter.
        # Find diameter via BFS
        def bfs_farthest(start, allowed_set):
            dist = {start: 0}
            parent = {start: -1}
            q = [start]
            farthest = start
            max_d = 0
            while q:
                curr = q.pop(0)
                for nei in adj[curr]:
                    if nei in allowed_set and nei not in dist:
                        dist[nei] = dist[curr] + 1
                        parent[nei] = curr
                        q.append(nei)
                        if dist[nei] > max_d:
                            max_d = dist[nei]
                            farthest = nei
            return farthest, max_d, parent
            
        start_node = lcc[0]
        mid, _, _ = bfs_farthest(start_node, lcc_set)
        end, _, parent_map = bfs_farthest(mid, lcc_set)
        
        # Reconstruct path from end to mid
        path = []
        curr = end
        while curr != -1:
            path.append(curr)
            if curr == mid:
                break
            curr = parent_map[curr]
            
        # Now we have path from end to mid. We need path from mid to start_node?
        # Actually, the diameter is between two leaves. 
        # The BFS from mid gives the farthest node 'end'.
        # The other end of the diameter is the farthest from 'end'.
        # Let's just do a DFS on the tree to find the longest path, simpler and less error prone for reconstruction.
        
        def find_diameter_path(root, parent, allowed_set):
            max_len = 0
            max_path = [root]
            
            def dfs(node, par, path):
                nonlocal max_len, max_path
                if len(path) > max_len:
                    max_len = len(path)
                    max_path = path[:]
                    
                for nei in adj[node]:
                    if nei in allowed_set and nei != par:
                        dfs(nei, node, path + [nei])
                        
            dfs(root, -1, [root])
            return max_path
            
        return find_diameter_path(lcc[0], -1, lcc_set)

    # 3. General Case: Use Warnsdorff's Rule with Backtracking
    best_path = []
    best_len = 0
    start_time = time.time()
    time_limit = 50.0
    
    visited = [False] * n
    
    def dfs(node, path):
        nonlocal best_path, best_len
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
            
        if time.time() - start_time > time_limit:
            return
            
        if len(path) == lcc_size:
            return
            
        unvisited_neighbors = []
        for nei in adj[node]:
            if nei in lcc_set and not visited[nei]:
                unvisited_count = 0
                for nn in adj[nei]:
                    if nn in lcc_set and not visited[nn]:
                        unvisited_count += 1
                unvisited_neighbors.append((unvisited_count, nei))
                
        if not unvisited_neighbors:
            return
            
        unvisited_neighbors.sort()
        
        for _, nei in unvisited_neighbors:
            visited[nei] = True
            path.append(nei)
            dfs(nei, path)
            path.pop()
            visited[nei] = False
            if time.time() - start_time > time_limit:
                return

    starts = lcc[:]
    starts.sort(key=lambda x: len([y for y in adj[x] if y in lcc_set]))
    
    for start_node in starts:
        if time.time() - start_time > time_limit:
            break
            
        for i in range(n):
            visited[i] = False
            
        visited[start_node] = True
        dfs(start_node, [start_node])
        
        if best_len == lcc_size:
            break

    return best_path if best_path else [lcc[0]]

# EVOLVE-BLOCK-END
