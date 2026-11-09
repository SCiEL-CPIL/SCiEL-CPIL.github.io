import time
import random
import sys
from collections import defaultdict

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # 45s safety margin out of 90s
    
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find connected components to limit search space
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
                for nb in adj[node]:
                    if not visited_comp[nb]:
                        visited_comp[nb] = True
                        stack.append(nb)
            components.append(comp)
    
    # Sort components by size (largest first)
    components.sort(key=len, reverse=True)
    
    best_path = []
    best_len = 0
    
    # Quick greedy to get initial bound
    def greedy_warnsdorff(start_node):
        visited = {start_node}
        path = [start_node]
        while True:
            last = path[-1]
            unvisited_neighbors = [x for x in adj[last] if x not in visited]
            if not unvisited_neighbors:
                break
            # Warnsdorff: pick neighbor with fewest unvisited neighbors
            next_node = min(unvisited_neighbors, key=lambda x: len(adj[x] - visited))
            path.append(next_node)
            visited.add(next_node)
        return path
    
    # Get initial best from greedy on all nodes
    for i in range(n):
        path = greedy_warnsdorff(i)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path
    
    # Iterative Deepening DFS with pruning
    # We'll search for paths of increasing length, starting from best_len+1
    # Use bitmasks for visited set (n=100 fits in 2 ints or we can use a frozenset)
    
    def dfs_find_path(start_node, target_len, visited_mask, current_path, time_budget):
        """
        Try to find a path of length target_len starting from start_node.
        visited_mask: tuple of two ints representing visited set (bits 0-63 and 64-99)
        Returns path if found, None otherwise.
        """
        if len(current_path) >= target_len:
            return current_path[:]
        
        if time.time() > time_budget:
            return None
        
        last = current_path[-1]
        # Sort neighbors by some heuristic (fewer unvisited neighbors first for pruning)
        neighbors = list(adj[last])
        # Filter to unvisited
        unvisited_neighbors = []
        for nb in neighbors:
            if nb < 64:
                if not (visited_mask[0] & (1 << nb)):
                    unvisited_neighbors.append(nb)
            else:
                if not (visited_mask[1] & (1 << (nb - 64))):
                    unvisited_neighbors.append(nb)
        
        if not unvisited_neighbors:
            return None
        
        # Heuristic: try neighbors with fewer unvisited neighbors first (dead-end avoidance)
        def neighbor_score(nb):
            count = 0
            for x in adj[nb]:
                if x < 64:
                    if not (visited_mask[0] & (1 << x)):
                        count += 1
                else:
                    if not (visited_mask[1] & (1 << (x - 64))):
                        count += 1
            return count
        
        unvisited_neighbors.sort(key=neighbor_score)
        
        for nb in unvisited_neighbors:
            # Update visited mask
            if nb < 64:
                new_mask = (visited_mask[0] | (1 << nb), visited_mask[1])
            else:
                new_mask = (visited_mask[0], visited_mask[1] | (1 << (nb - 64)))
            
            current_path.append(nb)
            result = dfs_find_path(nb, target_len, new_mask, current_path, time_budget)
            if result is not None:
                return result
            current_path.pop()
        
        return None
    
    # Try to find longer paths using iterative deepening
    # Start from best_len + 1 and go up
    # For each target length, try different starting nodes
    
    # Limit the search: we'll try up to a few more lengths
    max_extra = 5  # Try up to best_len + 5
    for target_len in range(best_len + 1, min(best_len + max_extra + 1, n + 1)):
        if time.time() > deadline:
            break
        
        # Try starting from different nodes
        # Focus on nodes in the largest component
        if components:
            start_nodes = components[0][:]
            # Add some nodes from other components if they're large enough
            for comp in components[1:]:
                if len(comp) >= target_len:
                    start_nodes.extend(comp)
        else:
            start_nodes = list(range(n))
        
        # Shuffle start nodes for diversity
        rng = random.Random(748 + target_len)
        rng.shuffle(start_nodes)
        
        found = False
        for start_node in start_nodes:
            if time.time() > deadline:
                break
            
            # Quick check: can we even reach target_len from this node?
            # Use a simple BFS/DFS to estimate max possible path length
            # For now, just try the DFS
            
            initial_mask = (1 << start_node, 0) if start_node < 64 else (0, 1 << (start_node - 64))
            path = dfs_find_path(start_node, target_len, initial_mask, [start_node], deadline)
            
            if path is not None and len(path) >= target_len:
                best_len = len(path)
                best_path = path
                found = True
                # We found a path of this length, try longer
                break
        
        if not found:
            # If we couldn't find a path of this length, we likely can't find longer ones
            # But let's try one more level to be safe
            continue
    
    # Final verification and return
    if not best_path:
        # Fallback: return a single node
        best_path = [0]
    
    # Verify the path is valid
    for i in range(len(best_path) - 1):
        if best_path[i + 1] not in adj[best_path[i]]:
            # Invalid path, fallback
            best_path = [0]
            break
    
    return best_path

# EVOLVE-BLOCK-END
