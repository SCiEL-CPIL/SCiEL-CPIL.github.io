"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

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
                for nb in adj[node]:
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)
            components.append(comp)
    
    # Sort components by size, work on the largest first
    components.sort(key=len, reverse=True)
    
    deadline = time.time() + 55  # safety margin before 60s limit
    best_path = []
    
    # For each component (starting with largest), try to find longest path
    for comp in components:
        if time.time() > deadline:
            break
        comp_set = set(comp)
        comp_size = len(comp)
        
        if comp_size <= len(best_path):
            continue
        
        # Try multiple starting vertices within this component
        # Prioritize vertices with higher degree
        sorted_starts = sorted(comp, key=lambda x: (-len(adj[x] & comp_set), x))
        
        for start in sorted_starts:
            if time.time() > deadline:
                break
            if len(best_path) >= comp_size:
                break
            
            # DFS with backtracking, guided by slack heuristic
            # Use iterative approach with explicit stack
            # Each stack entry: (path, seen, next_neighbor_index)
            # But that's complex. Let's use recursive with a call limit.
            
            # Recursive DFS with backtracking
            best_local = [start]
            seen = {start}
            node_count = [0]  # mutable counter for node visits
            
            def dfs(path, seen):
                if time.time() > deadline:
                    return
                if node_count[0] > 500000:  # limit total work
                    return
                node_count[0] += 1
                
                if len(path) > len(best_local):
                    best_local[:] = path[:]
                
                cur = path[-1]
                unvisited = [nb for nb in adj[cur] if nb not in seen]
                
                if not unvisited:
                    return
                
                # Sort by slack: prefer neighbors with more unvisited neighbors
                unvisited.sort(key=lambda nb: (-len(adj[nb] - seen), nb))
                
                for nb in unvisited:
                    if time.time() > deadline or node_count[0] > 500000:
                        return
                    path.append(nb)
                    seen.add(nb)
                    dfs(path, seen)
                    path.pop()
                    seen.discard(nb)
            
            dfs([start], {start})
            
            if len(best_local) > len(best_path):
                best_path = best_local[:]
        
        # Early exit if we found a Hamiltonian path in this component
        if len(best_path) >= comp_size:
            break
    
    # Fallback: if no path found (shouldn't happen), return a single vertex
    if not best_path:
        best_path = [0]
    
    return best_path
# EVOLVE-BLOCK-END
