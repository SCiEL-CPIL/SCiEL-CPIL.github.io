import time
import sys
from collections import deque

# Increase recursion limit
sys.setrecursionlimit(10000)

# P is the pinned graph data
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    deadline = time.time() + 80  # Safety margin
    
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find connected components
    visited_comp = [False] * n
    components = []
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
            components.append(comp)
    
    # The longest path must be in the largest component
    # Sort components by size descending
    components.sort(key=len, reverse=True)
    
    best_path = [components[0][0]] if components else [0]
    
    # We will search in the largest component
    # If the largest component is small, we can do exhaustive search.
    # If it's large (e.g., 70+ nodes), we use DFS with pruning.
    
    def dfs(path, current, visited):
        nonlocal best_path
        if len(path) > len(best_path):
            best_path = path[:]
        
        # Pruning: Check reachable unvisited nodes from current
        # If current + reachable <= best, prune
        # We do a quick BFS/DFS to count reachable unvisited nodes
        # This is O(N) per node. 
        # Optimization: If we have many unvisited nodes but they are in separate components
        # not connected to current, we can only extend into the one connected to current.
        
        reachable_count = 0
        # Quick check: just count neighbors? No, need transitive closure of unvisited.
        # For efficiency, we only do this full check if we are close to the best 
        # or if the tree is deep.
        
        # Actually, a simpler bound: 
        # Max extension = number of unvisited nodes in the component of 'current' in the graph induced by unvisited nodes.
        
        # Let's implement the reachable count.
        # Use a set to avoid global state changes if possible, or just a list.
        # Since N=100, a simple list-based BFS is fast.
        
        # Heuristic: Only do expensive reachable check if len(path) is close to best 
        # or if we are at a high degree node? 
        # Let's do it always if remaining potential > best.
        
        # Potential = len(path) + (number of unvisited nodes reachable from current)
        # If potential <= len(best_path), return
        
        # Count reachable
        # We can optimize by noting that if all unvisited nodes are connected, 
        # then reachable = total_unvisited.
        
        total_unvisited = n - len(visited)
        if len(path) + total_unvisited <= len(best_path):
            return
        
        # Check if we need to calculate reachable
        # If the graph is connected (which the giant component is), 
        # the reachable set is the unvisited nodes in that component.
        # Since we started in the largest component, and we only traverse within it,
        # 'visited' contains nodes from this component.
        # The unvisited nodes from this component are `component_nodes - visited`.
        # Are they all connected to current?
        # If the remaining unvisited subgraph is disconnected, we can only take one part.
        
        # For N=100, let's just do the BFS.
        stack = [current]
        seen_reach = {current}
        # We only care about unvisited nodes
        # But we must traverse through visited nodes? NO.
        # The path is simple. We cannot go through visited nodes.
        # So we only traverse unvisited nodes.
        
        # Wait, if I am at 'current', I can only go to unvisited neighbors.
        # So the reachable set is the unvisited nodes connected to 'current' via unvisited nodes.
        
        # Optimization: If total_unvisited is small, BFS is fast.
        # If total_unvisited is large, BFS is O(N+E).
        
        # Let's add a fast path: 
        # If we just started, or if we haven't visited many, 
        # the graph is likely still connected.
        
        # To be safe and fast, we do BFS.
        count = 0
        bstack = [current]
        bseen = {current}
        while bstack:
            u = bstack.pop()
            for v in adj[u]:
                if v in visited:
                    continue
                if v not in bseen:
                    bseen.add(v)
                    count += 1
                    bstack.append(v)
        
        if len(path) + count <= len(best_path):
            return
        
        # Extend
        neighbors = [v for v in adj[current] if v not in visited]
        
        # Heuristic: Sort by number of unvisited neighbors (Warnsdorff)
        # This helps find long paths faster
        def heuristic(v):
            if v in visited:
                return 999
            unvis_nei = 0
            for w in adj[v]:
                if w not in visited:
                    unvis_nei += 1
            return unvis_nei
        
        neighbors.sort(key=heuristic)
        
        for v in neighbors:
            if time.time() >= deadline:
                return
            path.append(v)
            visited.add(v)
            dfs(path, v, visited)
            path.pop()
            visited.discard(v)
            if time.time() >= deadline:
                return

    # Start DFS from each node in the largest component
    # But we can be smarter: 
    # The longest path in a tree is between two leaves.
    # In a general graph, it's harder.
    # We iterate all starts.
    
    largest_comp = components[0]
    
    # Optimization: If the component is a tree, we can find the diameter in O(N).
    # Let's check if it's a tree.
    # Edges in comp:
    e_count = 0
    for u in largest_comp:
        for v in adj[u]:
            if v in largest_comp and u < v:
                e_count += 1
                
    if e_count == len(largest_comp) - 1:
        # It's a tree! Find diameter.
        # Step 1: Pick any node, find farthest node A.
        # Step 2: Pick A, find farthest node B and path.
        def bfs_farthest(start, comp_set):
            dist = {start: 0}
            parent = {start: None}
            queue = [start]
            farthest = start
            max_dist = 0
            while queue:
                u = queue.pop(0)
                if dist[u] > max_dist:
                    max_dist = dist[u]
                    farthest = u
                for v in adj[u]:
                    if v in comp_set and v not in dist:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
            # Reconstruct path
            path = [farthest]
            while parent[farthest] is not None:
                farthest = parent[farthest]
                path.append(farthest)
            path.reverse()
            return path
        
        # Step 1
        start_node = largest_comp[0]
        path1 = bfs_farthest(start_node, set(largest_comp))
        end_node_a = path1[-1]
        
        # Step 2
        path2 = bfs_farthest(end_node_a, set(largest_comp))
        
        if len(path2) > len(best_path):
            best_path = path2
    else:
        # Not a tree, use DFS with pruning
        # To reduce starts, we can just start from all nodes in the largest component
        # But we can also try to start from leaves of the "core"
        
        # Simple approach: iterate all nodes in largest component
        for start in largest_comp:
            if time.time() >= deadline:
                break
            path = [start]
            visited = {start}
            dfs(path, start, visited)
            if time.time() >= deadline:
                break

    return best_path

# EVOLVE-BLOCK-END
