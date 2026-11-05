import time
import sys
from collections import defaultdict

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
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
            comp = []
            stack = [i]
            visited_comp[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for nei in adj[node]:
                    if not visited_comp[nei]:
                        visited_comp[nei] = True
                        stack.append(nei)
            components.append(comp)
            
    # Sort components by size descending
    components.sort(key=len, reverse=True)
    
    best_path = []
    best_len = 0
    deadline = time.time() + 50.0  # Safety margin for 60s limit
    
    # Helper to check if a path is valid (should be by construction, but good for sanity)
    def is_valid_path(path):
        if not path:
            return False
        for i in range(len(path) - 1):
            if path[i+1] not in adj[path[i]]:
                return False
        return True

    # Greedy initial guess to get a decent lower bound quickly
    def greedy_path(start):
        path = [start]
        visited = {start}
        while True:
            curr = path[-1]
            candidates = [nei for nei in adj[curr] if nei not in visited]
            if not candidates:
                break
            # Heuristic: pick neighbor with highest degree in unvisited
            best_nei = max(candidates, key=lambda x: len(adj[x] - visited))
            path.append(best_nei)
            visited.add(best_nei)
        return path

    # Try greedy from each node in the largest component to initialize best
    if components:
        largest_comp = components[0]
        for start in largest_comp:
            if time.time() > deadline:
                break
            gp = greedy_path(start)
            if len(gp) > best_len:
                best_len = len(gp)
                best_path = gp

    # Branch and Bound on each component
    for comp in components:
        if len(comp) <= best_len:
            continue
            
        comp_set = set(comp)
        comp_nodes = comp
        
        # We will run B&B starting from each node in the component? 
        # Or just one start? 
        # Starting from all nodes is more thorough but slower.
        # Given N=100, let's try all starts in the top few components.
        
        # To optimize, we can just run a single B&B that can start anywhere?
        # No, B&B usually fixes the start.
        # Let's iterate starts.
        
        for start in comp_nodes:
            if time.time() > deadline:
                break
            
            # Initial state
            path = [start]
            visited = {start}
            
            # Recursive B&B
            def dfs(curr_path, visited_set):
                nonlocal best_path, best_len, deadline
                
                if time.time() > deadline:
                    return

                curr_len = len(curr_path)
                
                # Update best
                if curr_len > best_len:
                    best_len = curr_len
                    best_path = curr_path[:]
                    
                # Pruning: Even if we visit all remaining nodes in the component, can we beat best_len?
                # Max possible length = curr_len + (len(comp) - curr_len) = len(comp)
                # If len(comp) <= best_len, we would have skipped this component entirely.
                # So this check is redundant if we skip small components.
                
                # Stronger Pruning:
                # Let's see how many unvisited nodes are reachable from curr_path[-1]
                # in the subgraph induced by (comp - visited_set).
                # If curr_len + 1 + reachable_count <= best_len, prune.
                
                last_node = curr_path[-1]
                unvisited_in_comp = comp_set - visited_set
                
                if not unvisited_in_comp:
                    return

                # BFS to find reachable unvisited nodes from last_node
                # Note: last_node is in visited_set, so we start BFS from its neighbors
                # that are in unvisited_in_comp.
                
                reachable = set()
                stack = []
                for nei in adj[last_node]:
                    if nei in unvisited_in_comp:
                        stack.append(nei)
                        reachable.add(nei)
                
                while stack:
                    node = stack.pop()
                    for nei in adj[node]:
                        if nei in unvisited_in_comp and nei not in reachable:
                            reachable.add(nei)
                            stack.append(nei)
                            
                # If we add 1 node (the next step), the max extension is len(reachable)
                # Total max path length = curr_len + 1 + len(reachable)
                if curr_len + 1 + len(reachable) <= best_len:
                    return

                # Generate candidates: neighbors of last_node in unvisited_in_comp
                candidates = [nei for nei in adj[last_node] if nei in unvisited_in_comp]
                
                if not candidates:
                    return

                # Sort candidates to find good solutions earlier
                # Heuristic: prefer nodes with higher degree in the unvisited subgraph
                def score_candidate(node):
                    # Count unvisited neighbors
                    unv_neis = [n for n in adj[node] if n in unvisited_in_comp]
                    return len(unv_neis)
                
                candidates.sort(key=score_candidate, reverse=True)
                
                for cand in candidates:
                    if time.time() > deadline:
                        return
                    new_path = curr_path + [cand]
                    new_visited = visited_set | {cand}
                    dfs(new_path, new_visited)
            
            dfs(path, visited)
            
            # If we found a path covering the whole component, we can stop searching this component
            if best_len == len(comp):
                break

    if not best_path:
        # Fallback: return any valid edge or single node
        if P['edges']:
            return [P['edges'][0][0], P['edges'][0][1]]
        return [0]
        
    return best_path
# EVOLVE-BLOCK-END
