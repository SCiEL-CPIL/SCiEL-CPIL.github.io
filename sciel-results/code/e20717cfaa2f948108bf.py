"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    deadline = time.time() + 55  # 55s safety margin
    rng = random.Random(748)
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Convert to list for faster iteration
    adj_list = [list(s) for s in adj]
    
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
                for nb in adj_list[node]:
                    if not visited_comp[nb]:
                        visited_comp[nb] = True
                        stack.append(nb)
            components.append(comp)
    
    # Sort components by size descending (search largest first)
    components.sort(key=len, reverse=True)
    
    best_path = []
    best_len = 0
    
    # Greedy initial solution: extend path greedily from each vertex
    def greedy_extend(start, adj_list, n, rng=None):
        path = [start]
        visited = {start}
        while True:
            last = path[-1]
            candidates = [nb for nb in adj_list[last] if nb not in visited]
            if not candidates:
                break
            if rng is not None and rng.random() < 0.3:
                # Random choice with some bias toward low-degree
                weights = [1.0 / (1 + len([y for y in adj_list[x] if y not in visited])) for x in candidates]
                total_w = sum(weights)
                r = rng.random() * total_w
                cumulative = 0
                chosen = candidates[0]
                for c, w in zip(candidates, weights):
                    cumulative += w
                    if r <= cumulative:
                        chosen = c
                        break
            else:
                # Deterministic: pick candidate with fewest unvisited neighbors
                candidates.sort(key=lambda x: len([y for y in adj_list[x] if y not in visited]))
                chosen = candidates[0]
            path.append(chosen)
            visited.add(chosen)
        return path
    
    # Get initial greedy solutions
    for start in range(n):
        gpath = greedy_extend(start, adj_list, n)
        if len(gpath) > best_len:
            best_path = gpath[:]
            best_len = len(gpath)
    
    # Compute matching-based slack for each vertex to guide search
    # For the largest component, compute a maximum matching to identify
    # vertices with low slack (fewer alternative matches)
    def compute_matching_slack(comp, adj_list, n):
        """Compute slack for each vertex using a simple greedy matching."""
        comp_set = set(comp)
        # Build bipartite-like structure: each vertex can be matched to at most one neighbor
        # Use a simple greedy approach to find a near-maximum matching
        matched_to = [-1] * n  # matched_to[v] = u if v is matched to u
        matched_from = [-1] * n  # matched_from[u] = v if u is matched to v
        
        # Sort vertices by degree ascending (match low-degree first)
        sorted_comp = sorted(comp, key=lambda x: len([y for y in adj_list[x] if y in comp_set]))
        
        for v in sorted_comp:
            # Find an unmatched neighbor
            for nb in adj_list[v]:
                if nb in comp_set and matched_to[nb] == -1 and matched_from[nb] == -1:
                    matched_to[v] = nb
                    matched_from[nb] = v
                    break
        
        # Compute slack: number of potential matches minus 1 (if matched)
        # Low slack means the vertex has few alternatives
        slack = [0] * n
        for v in comp:
            potential = len([y for y in adj_list[v] if y in comp_set])
            if matched_to[v] != -1:
                slack[v] = potential - 1  # One match used, rest are slack
            else:
                slack[v] = potential
        
        return slack
    
    # Iterative deepening DFS with matching-guided heuristics
    def dfs_guided(start, comp, comp_size, max_depth, deadline, rng, slack):
        """DFS guided by matching slack to prioritize constrained vertices."""
        nonlocal best_path, best_len
        
        comp_set = set(comp)
        
        def dfs(path, visited):
            nonlocal best_path, best_len
            if time.time() > deadline:
                return False
            
            cur_len = len(path)
            if cur_len > best_len:
                best_path = path[:]
                best_len = cur_len
            
            if cur_len == max_depth:
                return True
            
            last = path[-1]
            # Get unvisited neighbors in component
            unvisited_neighbors = [nb for nb in adj_list[last] if nb not in visited and nb in comp_set]
            if not unvisited_neighbors:
                return False
            
            # Sort by matching slack (ascending) - prefer vertices with fewer alternatives
            # This helps avoid trapping the path by prioritizing constrained choices
            unvisited_neighbors.sort(key=lambda x: slack[x])
            
            for neighbor in unvisited_neighbors:
                path.append(neighbor)
                visited.add(neighbor)
                if dfs(path, visited):
                    path.pop()
                    visited.remove(neighbor)
                    return True
                path.pop()
                visited.remove(neighbor)
                if time.time() > deadline:
                    return False
            return False
        
        return dfs([start], {start})
    
    # Try each component, starting with largest
    for comp in components:
        if time.time() > deadline:
            break
        comp_size = len(comp)
        if comp_size <= best_len:
            break
        
        # Compute matching slack for this component
        slack = compute_matching_slack(comp, adj_list, n)
        
        # Try vertices in this component as start, ordered by slack ascending (most constrained first)
        comp_vertices = sorted(comp, key=lambda x: slack[x])
        
        # For each start, try increasing path lengths
        for start in comp_vertices:
            if time.time() > deadline:
                break
            
            # Try to find paths of increasing length
            # Start from best_len+1 and go up to comp_size
            target_len = best_len + 1
            while target_len <= comp_size and time.time() < deadline:
                found = dfs_guided(start, comp, comp_size, target_len, deadline, rng, slack)
                if found:
                    target_len += 1
                else:
                    # If we couldn't find a path of this length, break
                    break
        
        # Randomized restarts: try random starting points and greedy extensions
        restarts = 0
        while time.time() < deadline and restarts < 100:
            start = rng.choice(comp)
            gpath = greedy_extend(start, adj_list, n, rng)
            if len(gpath) > best_len:
                best_path = gpath[:]
                best_len = len(gpath)
            restarts += 1
    
    return best_path if best_path else [0]
# EVOLVE-BLOCK-END
