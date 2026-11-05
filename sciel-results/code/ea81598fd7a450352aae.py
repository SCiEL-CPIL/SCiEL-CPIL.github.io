"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}


def _find_components(n, adj):
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
                for nb in adj[node]:
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)
            components.append(comp)
    return components


def _count_reachable(start, adj, blocked):
    """Count nodes reachable from start avoiding blocked nodes."""
    if start in blocked:
        return 0
    visited = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for nb in adj[node]:
            if nb not in visited and nb not in blocked:
                visited.add(nb)
                stack.append(nb)
    return len(visited)


def _bfs_diameter(comp, adj):
    """Find diameter of a tree-like component via BFS from two endpoints."""
    if len(comp) <= 1:
        return 0, []
    
    comp_set = set(comp)
    comp_adj = {v: [u for u in adj[v] if u in comp_set] for v in comp}
    
    # BFS from arbitrary node to find farthest
    def bfs_farthest(start):
        dist = {start: 0}
        parent = {start: None}
        queue = [start]
        farthest = start
        while queue:
            node = queue.pop(0)
            for nb in comp_adj[node]:
                if nb not in dist:
                    dist[nb] = dist[node] + 1
                    parent[nb] = node
                    queue.append(nb)
                    if dist[nb] > dist[farthest]:
                        farthest = nb
        return farthest, dist, parent
    
    # First BFS
    start = comp[0]
    far1, dist1, parent1 = bfs_farthest(start)
    
    # Second BFS from far1
    far2, dist2, parent2 = bfs_farthest(far1)
    
    # Reconstruct path
    path = []
    node = far2
    while node is not None:
        path.append(node)
        node = parent2[node]
    path.reverse()
    
    return len(path) - 1, path


def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)

    # Find connected components
    components = _find_components(n, adj)
    
    # Sort components by size descending - focus on largest first
    components.sort(key=len, reverse=True)

    best_path = []

    # For each component, try to find longest path
    for comp in components:
        if time.time() > deadline:
            break
        comp_set = set(comp)
        comp_size = len(comp)
        
        # Quick check: if component is a tree, use diameter
        edge_count = sum(len(adj[v] & comp_set) for v in comp) // 2
        if edge_count == comp_size - 1:
            # It's a tree - diameter is optimal
            _, diameter_path = _bfs_diameter(comp, adj)
            if len(diameter_path) > len(best_path):
                best_path = diameter_path
            continue
        
        # Build adjacency within component
        comp_adj = {v: [u for u in adj[v] if u in comp_set] for v in comp}
        
        # Try multiple strategies
        strategies = [
            'warnsdorff',    # prefer nodes with fewer unvisited neighbors
            'anti_warnsdorff',  # prefer nodes with more unvisited neighbors  
            'degree_asc',    # prefer lower degree
            'degree_desc'    # prefer higher degree
        ]
        
        best_in_comp = []
        
        # Order starts: try lower-degree vertices first (likely endpoints)
        start_order = sorted(comp, key=lambda v: len(comp_adj[v]))
        
        for strategy in strategies:
            if time.time() > deadline:
                break
            
            for start in start_order:
                if time.time() > deadline:
                    break
                
                # If we already have a path as long as the component, stop
                if len(best_in_comp) >= comp_size:
                    break
                
                path = [start]
                path_set = {start}
                
                def dfs():
                    nonlocal best_in_comp, path, path_set
                    if time.time() > deadline:
                        return
                    
                    if len(path) > len(best_in_comp):
                        best_in_comp = list(path)
                        if len(best_in_comp) == comp_size:
                            return  # Hamiltonian path found
                    
                    current = path[-1]
                    candidates = [nb for nb in comp_adj[current] if nb not in path_set]
                    if not candidates:
                        return
                    
                    # Branch and bound: check if remaining nodes can beat best
                    remaining = comp_size - len(path)
                    if len(path) + remaining <= len(best_in_comp):
                        return
                    
                    # Heuristic ordering based on strategy
                    if strategy == 'warnsdorff':
                        # Prefer nodes with fewer unvisited neighbors (to avoid dead ends)
                        candidates.sort(key=lambda x: len([y for y in comp_adj[x] if y not in path_set]))
                    elif strategy == 'anti_warnsdorff':
                        # Prefer nodes with more unvisited neighbors
                        candidates.sort(key=lambda x: -len([y for y in comp_adj[x] if y not in path_set]))
                    elif strategy == 'degree_asc':
                        # Prefer lower total degree
                        candidates.sort(key=lambda x: len(comp_adj[x]))
                    elif strategy == 'degree_desc':
                        # Prefer higher total degree
                        candidates.sort(key=lambda x: -len(comp_adj[x]))
                    
                    for nb in candidates:
                        if time.time() > deadline:
                            return
                        
                        # Additional pruning: if taking this node disconnects the rest
                        # in a way that can't beat our best, skip
                        path.append(nb)
                        path_set.add(nb)
                        
                        # Quick check: count reachable from nb avoiding path_set minus {nb}
                        # But this is expensive, so only do it for deeper searches
                        if len(path) > len(best_in_comp) - 2:
                            # Near the end, do a quick reachability check
                            reachable = _count_reachable(nb, comp_adj, path_set - {nb})
                            if len(path) + reachable <= len(best_in_comp):
                                path.pop()
                                path_set.remove(nb)
                                continue
                        
                        dfs()
                        
                        path.pop()
                        path_set.remove(nb)
                        if time.time() > deadline:
                            return
                    return
                
                dfs()
                if time.time() > deadline:
                    break
        
        if len(best_in_comp) > len(best_path):
            best_path = best_in_comp

    # If no path found (shouldn't happen), return a single vertex
    if not best_path:
        # Find any edge
        for u, v in P['edges']:
            return [u, v]
        return [0]

    return best_path

# EVOLVE-BLOCK-END
