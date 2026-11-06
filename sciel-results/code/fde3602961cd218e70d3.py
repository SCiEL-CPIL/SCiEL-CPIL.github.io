import time
import random
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit (5s buffer)

    P_data = {
        'n': 100,
        'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]]
    }

    n = P_data['n']
    adj = [set() for _ in range(n)]
    adj_list = [[] for _ in range(n)]
    for u, v in P_data['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    for i in range(n):
        adj_list[i] = list(adj[i])
    
    # Find largest component
    visited_comp = [False] * n
    largest_comp = []
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
            if len(comp) > len(largest_comp):
                largest_comp = comp
    
    best_path = []
    
    def time_check():
        return time.time() < deadline

    def extend_path_greedy(path):
        """
        Attempt to extend the path at both ends greedily using Warnsdorff's rule.
        Returns the extended path.
        """
        if not path:
            return path
        path = path[:]
        visited_set = set(path)
        changed = True
        while changed:
            changed = False
            # Extend start
            start_node = path[0]
            candidates = [v for v in adj_list[start_node] if v not in visited_set]
            if candidates:
                # Warnsdorff: pick neighbor with fewest unvisited neighbors
                candidates.sort(key=lambda v: len(adj[v] - visited_set))
                next_v = candidates[0]
                path.insert(0, next_v)
                visited_set.add(next_v)
                changed = True
            
            # Extend end
            if not time_check():
                break
            end_node = path[-1]
            candidates = [v for v in adj_list[end_node] if v not in visited_set]
            if candidates:
                candidates.sort(key=lambda v: len(adj[v] - visited_set))
                next_v = candidates[0]
                path.append(next_v)
                visited_set.add(next_v)
                changed = True
                
        return path

    def greedy_walk(start, strategy='warnsdorff', rng=None):
        """
        Generate a path using a greedy strategy.
        """
        if rng is None:
            rng = random.Random()
            
        path = [start]
        visited_set = {start}
        
        while time_check():
            current = path[-1]
            neighbors = [v for v in adj_list[current] if v not in visited_set]
            if not neighbors:
                break
            
            if strategy == 'warnsdorff':
                # Prefer neighbors with fewer unvisited neighbors
                neighbors.sort(key=lambda v: len(adj[v] - visited_set))
                next_node = neighbors[0]
            elif strategy == 'random':
                next_node = rng.choice(neighbors)
            elif strategy == 'max_deg':
                # Prefer neighbors with more unvisited neighbors
                neighbors.sort(key=lambda v: len(adj[v] - visited_set), reverse=True)
                next_node = neighbors[0]
            else:
                next_node = rng.choice(neighbors)
                
            path.append(next_node)
            visited_set.add(next_node)
            
        return path

    def deep_dfs(start, current_best_len):
        """
        Perform a DFS from 'start' to find a long path.
        Uses Warnsdorff's rule for ordering.
        Updates best_path non-locally.
        """
        nonlocal best_path
        
        visited = [False] * n
        path = []
        
        def dfs(u):
            nonlocal best_path, time_check, deadline
            
            if not time_check():
                return False
            
            path.append(u)
            visited[u] = True
            
            if len(path) > len(best_path):
                best_path = path[:]
                
            if len(path) == n:
                # Found a Hamiltonian path (unlikely but possible)
                path.pop()
                visited[u] = False
                return True
            
            curr = u
            candidates = [v for v in adj_list[curr] if not visited[v]]
            
            if not candidates:
                path.pop()
                visited[u] = False
                return False
            
            # Sort candidates using Warnsdorff's rule
            # We want to visit vertices that have fewer unvisited neighbors first
            # to avoid isolating them later
            candidates.sort(key=lambda v: len(adj[v] - {x for x in range(n) if visited[x]}))
            
            # Optional: If a candidate has only 1 unvisited neighbor (the current one), 
            # it is a forced move. The sort handles this naturally (score 1 or 0).
            
            found = False
            for v in candidates:
                if not time_check():
                    break
                if dfs(v):
                    found = True
                    # If we found a full path, we could stop, but let's just return 
                    # to unwind. The best_path is already updated.
                    break
            
            path.pop()
            visited[u] = False
            return found
            
        dfs(start)

    # Phase 1: Greedy Initial Paths
    # Run multiple greedy walks from different starts with different strategies
    # to get a good initial best_path.
    
    # Start from all vertices in largest component with Warnsdorff
    for start in largest_comp:
        if not time_check():
            break
        path = greedy_walk(start, 'warnsdorff')
        path = extend_path_greedy(path)
        if len(path) > len(best_path):
            best_path = path[:]
            
    # If we haven't found a very long path, try random walks
    if len(best_path) < 50:
        rng = random.Random(42)
        for _ in range(20):
            if not time_check():
                break
            start = rng.choice(largest_comp)
            path = greedy_walk(start, 'random', rng)
            path = extend_path_greedy(path)
            if len(path) > len(best_path):
                best_path = path[:]
                
    # Phase 2: Deep DFS from promising starts
    # The greedy path gives us a good idea of where to start.
    # We will run DFS from the start vertex of the best path found so far, 
    # and maybe a few others.
    
    # Start DFS from the first vertex of the best path
    if best_path:
        deep_dfs(best_path[0], len(best_path))
        
    if len(best_path) == n:
        return best_path

    # If still not optimal, try DFS from a few other vertices
    # Prioritize vertices that are in the best path or have high degree
    # Let's try the first 5 vertices of the best path
    if best_path:
        starts_to_try = list(set(best_path[:5]))
        # Add a few random ones
        for _ in range(5):
            if not time_check():
                break
            starts_to_try.append(random.choice(largest_comp))
            
        for start in starts_to_try:
            if not time_check():
                break
            if len(best_path) == n:
                break
            deep_dfs(start, len(best_path))
            
    if len(best_path) == n:
        return best_path

    # Phase 3: Final Polish / Extension
    # Try to extend the best path found by DFS using greedy extension again.
    # Sometimes DFS gets stuck in a local minimum where greedy could extend.
    final_path = extend_path_greedy(best_path)
    if len(final_path) > len(best_path):
        best_path = final_path

    # Validation
    if not best_path:
        best_path = [largest_comp[0]]
        
    # Ensure distinct vertices and valid edges
    seen = set()
    valid_path = []
    for node in best_path:
        if node not in seen:
            valid_path.append(node)
            seen.add(node)
        else:
            break
    best_path = valid_path
    
    for i in range(len(best_path) - 1):
        if best_path[i+1] not in adj[best_path[i]]:
            best_path = best_path[:i+1]
            break
            
    return best_path
