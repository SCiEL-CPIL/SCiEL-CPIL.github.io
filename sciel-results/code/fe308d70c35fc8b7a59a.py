import time
import random
import sys
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # 75s safety margin out of 90s
    
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        
    degrees = [len(adj[i]) for i in range(n)]
    
    best_path = [0]
    best_len = 1
    
    # Helper: time check
    def time_ok():
        return time.time() < deadline
    
    # Strategy 1: Warnsdorff greedy from all nodes
    def greedy_warnsdorff(start_node):
        visited = set([start_node])
        path = [start_node]
        while True:
            if not time_ok():
                break
            last = path[-1]
            neighbors = [x for x in adj[last] if x not in visited]
            if not neighbors:
                break
            next_node = min(neighbors, key=lambda x: (degrees[x], x))
            path.append(next_node)
            visited.add(next_node)
        return path
    
    for i in range(n):
        if not time_ok():
            break
        path = greedy_warnsdorff(i)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
    
    # Strategy 2: Max-degree greedy from all nodes
    def greedy_max_degree(start_node):
        visited = set([start_node])
        path = [start_node]
        while True:
            if not time_ok():
                break
            last = path[-1]
            neighbors = [x for x in adj[last] if x not in visited]
            if not neighbors:
                break
            next_node = max(neighbors, key=lambda x: (len(adj[x] - visited), x))
            path.append(next_node)
            visited.add(next_node)
        return path
    
    for i in range(n):
        if not time_ok():
            break
        path = greedy_max_degree(i)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
    
    # Strategy 3: Forced-move greedy with random tie-breaking
    rng = random.Random(748)
    
    def greedy_forced_random(start_node):
        visited = set([start_node])
        path = [start_node]
        while True:
            if not time_ok():
                break
            last = path[-1]
            neighbors = [x for x in adj[last] if x not in visited]
            if not neighbors:
                break
            forced = [x for x in neighbors if len(adj[x] - visited) == 1]
            if forced:
                next_node = forced[0]
            else:
                next_node = rng.choice(neighbors)
            path.append(next_node)
            visited.add(next_node)
        return path
    
    for i in range(n):
        if not time_ok():
            break
        path = greedy_forced_random(i)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
    
    # More random restarts of forced greedy
    for _ in range(200):
        if not time_ok():
            break
        start_node = rng.randint(0, n - 1)
        path = greedy_forced_random(start_node)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
    
    # Strategy 4: Iterative Deepening DFS to try to extend beyond best_len
    # We try to find a path of length best_len + 1, best_len + 2, etc.
    # Use DFS with pruning and time limits per target length
    
    # For IDDFS, we search from each start node
    # We use a stack-based DFS to avoid recursion depth issues
    
    def find_path_iddfs(target_len, start_node):
        """Try to find a simple path of exactly target_len vertices starting from start_node."""
        if not time_ok():
            return None
        
        # Stack entries: (current_node, path_tuple, visited_frozenset)
        # Use a list as stack
        stack = [(start_node, (start_node,), frozenset([start_node]))]
        visited_global = set()  # to avoid revisiting the same state in this DFS call
        
        while stack:
            if not time_ok():
                return None
            
            current, path, visited = stack.pop()
            
            if len(path) == target_len:
                return list(path)
            
            # Check if we've already processed this state
            state = (current, path)  # This is expensive; use a different approach
            # Actually, for a path problem, the state is determined by the last node and the visited set
            # But that's too large. Let's just proceed with the DFS and rely on the time limit.
            
            neighbors = [x for x in adj[current] if x not in visited]
            
            # Order neighbors by degree (Warnsdorff) for better pruning
            neighbors.sort(key=lambda x: (degrees[x], x))
            
            for nb in neighbors:
                if not time_ok():
                    return None
                new_path = path + (nb,)
                new_visited = visited | {nb}
                # Add to stack in reverse order so that lower degree is explored first (LIFO)
                stack.append((nb, new_path, new_visited))
        
        return None
    
    # Try to improve by finding paths of length best_len + 1, best_len + 2, etc.
    for target_len in range(best_len + 1, n + 1):
        if not time_ok():
            break
        
        found = False
        for start_node in range(n):
            if not time_ok():
                break
            
            path = find_path_iddfs(target_len, start_node)
            if path is not None:
                best_path = path
                best_len = target_len
                found = True
                break
        
        if not found:
            # If we can't find a path of this length, stop trying longer ones
            # (since we're searching for exact length, if target_len fails, longer ones are harder)
            # But actually, it's possible that target_len fails but target_len+1 succeeds from a different start?
            # No, if a path of length L exists, a path of length L-1 also exists (just truncate).
            # So if no path of length target_len exists, no path of length > target_len exists.
            break
    
    return best_path

# EVOLVE-BLOCK-END
