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
    
    # Find connected components
    def find_components():
        visited = set()
        components = []
        for i in range(n):
            if i not in visited:
                comp = set()
                stack = [i]
                visited.add(i)
                while stack:
                    node = stack.pop()
                    comp.add(node)
                    for nb in adj[node]:
                        if nb not in visited:
                            visited.add(nb)
                            stack.append(nb)
                components.append(comp)
        return components
    
    components = find_components()
    # Sort components by size descending
    components.sort(key=len, reverse=True)
    
    # Get the largest component's vertices
    if components:
        largest_comp = components[0]
    else:
        largest_comp = set(range(n))
    
    best_path = [0]
    best_len = 1
    
    # Helper: time check
    def time_ok():
        return time.time() < deadline
    
    # Strategy 1: Warnsdorff greedy from all nodes in largest component
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
    
    for i in sorted(largest_comp, key=lambda x: degrees[x], reverse=True):
        if not time_ok():
            break
        path = greedy_warnsdorff(i)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
    
    # Strategy 2: Max-degree greedy from all nodes in largest component
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
    
    for i in sorted(largest_comp, key=lambda x: degrees[x], reverse=True):
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
    
    for i in sorted(largest_comp, key=lambda x: degrees[x], reverse=True):
        if not time_ok():
            break
        path = greedy_forced_random(i)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
    
    # More random restarts of forced greedy
    for _ in range(500):
        if not time_ok():
            break
        start_node = rng.choice(list(largest_comp))
        path = greedy_forced_random(start_node)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
    
    # Strategy 4: Two-end extension - try to extend both ends of the current best path
    def extend_both_ends(path):
        """Try to extend a path from both ends."""
        if not path:
            return path
        visited = set(path)
        extended = list(path)
        
        # Try extending from the start
        while True:
            if not time_ok():
                break
            first = extended[0]
            neighbors = [x for x in adj[first] if x not in visited]
            if not neighbors:
                break
            # Choose neighbor with minimum remaining degree (Warnsdorff)
            next_node = min(neighbors, key=lambda x: (degrees[x], x))
            extended.insert(0, next_node)
            visited.add(next_node)
        
        # Try extending from the end
        while True:
            if not time_ok():
                break
            last = extended[-1]
            neighbors = [x for x in adj[last] if x not in visited]
            if not neighbors:
                break
            next_node = min(neighbors, key=lambda x: (degrees[x], x))
            extended.append(next_node)
            visited.add(next_node)
        
        return extended
    
    # Apply two-end extension to best path
    if len(best_path) > 1:
        extended = extend_both_ends(best_path)
        if len(extended) > best_len:
            best_len = len(extended)
            best_path = extended
    
    # Strategy 5: Backtracking search to find longer paths
    # Use a limited-depth DFS with pruning
    def backtracking_search(start_node, max_depth, current_path, visited):
        """Try to find a path of length max_depth starting from start_node."""
        if not time_ok():
            return None
        if len(current_path) == max_depth:
            return list(current_path)
        
        last = current_path[-1]
        neighbors = [x for x in adj[last] if x not in visited]
        
        # Sort by Warnsdorff score
        neighbors.sort(key=lambda x: (degrees[x], x))
        
        for nb in neighbors:
            if not time_ok():
                return None
            result = backtracking_search(nb, max_depth, current_path + [nb], visited | {nb})
            if result is not None:
                return result
        
        return None
    
    # Try to find paths of increasing length using backtracking
    # Start from best_len + 1 and go up
    for target_len in range(best_len + 1, min(best_len + 10, n + 1)):
        if not time_ok():
            break
        
        found = False
        # Try from high-degree nodes in the largest component
        for start_node in sorted(largest_comp, key=lambda x: degrees[x], reverse=True):
            if not time_ok():
                break
            
            # Use a time-limited backtracking search
            path = backtracking_search(start_node, target_len, [start_node], {start_node})
            if path is not None:
                best_path = path
                best_len = target_len
                found = True
                break
        
        if not found:
            # If we can't find a path of this length, try a few more starts before giving up
            for _ in range(5):
                if not time_ok():
                    break
                start_node = rng.choice(list(largest_comp))
                path = backtracking_search(start_node, target_len, [start_node], {start_node})
                if path is not None:
                    best_path = path
                    best_len = target_len
                    found = True
                    break
        
        if not found:
            # If we can't find a path of this length, stop trying longer ones
            break
    
    # Final polish: try to extend the best path from both ends one more time
    if len(best_path) > 1:
        extended = extend_both_ends(best_path)
        if len(extended) > best_len:
            best_len = len(extended)
            best_path = extended
    
    return best_path

# EVOLVE-BLOCK-END
