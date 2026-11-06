import time
import sys

sys.setrecursionlimit(10000)

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute adjacency as bitmasks for fast operations
    adj_mask = [0] * n
    for u, v in P['edges']:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    def time_check():
        return time.time() > deadline
    
    # Strategy 1: Greedy Warnsdorff from all starts to get a good baseline
    best_path = []
    for start in range(n):
        if time_check():
            break
        path = [start]
        visited_mask = (1 << start)
        while True:
            curr = path[-1]
            candidates = [v for v in adj[curr] if not (visited_mask >> v) & 1]
            if not candidates:
                break
            # Warnsdorff: prefer neighbors with fewest unvisited neighbors
            candidates.sort(key=lambda v: bin(adj_mask[v] & ~visited_mask).count('1'))
            next_v = candidates[0]
            path.append(next_v)
            visited_mask |= (1 << next_v)
        
        if len(path) > len(best_path):
            best_path = path[:]
    
    if len(best_path) == n:
        return best_path
    
    best_len = len(best_path)
    
    # Strategy 2: Iterative deepening DFS with bitmasks
    # Try to find paths longer than best_len
    
    visited = [False] * n
    path = []
    found_full = [False]
    
    def dfs(u, depth, target_len):
        nonlocal best_path, best_len, found_full
        if time_check() or found_full[0]:
            return False
        
        path.append(u)
        visited[u] = True
        
        if len(path) > best_len:
            best_path = path[:]
            best_len = len(path)
        
        if len(path) == n:
            found_full[0] = True
            return True
        
        if len(path) == target_len:
            path.pop()
            visited[u] = False
            return False
        
        curr = u
        # Get unvisited neighbors using bitmask
        unvisited = ~((1 << u) | 0)  # placeholder
        # Actually, let's just iterate adj[curr] and check visited
        candidates = []
        for v in adj[curr]:
            if not visited[v]:
                # Count unvisited neighbors of v for Warnsdorff
                cnt = 0
                for w in adj[v]:
                    if not visited[w]:
                        cnt += 1
                candidates.append((cnt, v))
        
        if not candidates:
            path.pop()
            visited[u] = False
            return False
        
        # Sort by remaining degree (Warnsdorff)
        candidates.sort()
        
        for _, v in candidates:
            if time_check() or found_full[0]:
                break
            if dfs(v, depth + 1, target_len):
                # Don't pop on success - let the path stay
                # But we need to clean up if we're going to return
                # Actually, we return True and the caller handles it
                return True
        
        path.pop()
        visited[u] = False
        return False
    
    # Try iterative deepening from best_len+1 up to n
    # Start from vertices that were in the best path first
    start_order = list(best_path) + [i for i in range(n) if i not in set(best_path)]
    
    for target in range(best_len + 1, n + 1):
        if time_check() or found_full[0]:
            break
        for start in start_order[:20]:  # Limit starts to save time
            if time_check() or found_full[0]:
                break
            # Reset state
            for i in range(n):
                visited[i] = False
            path.clear()
            found_full[0] = False
            
            if dfs(start, 0, target):
                if found_full[0]:
                    break
                # If we found a path of length target, try longer
                break
    
    if len(best_path) == n:
        return best_path
    
    # Strategy 3: Try to extend the best path by rearranging
    # Use local search: try to insert unvisited vertices into the path
    if best_path and len(best_path) < n:
        unvisited = set(range(n)) - set(best_path)
        
        # Try to extend from both ends
        improved = True
        while improved and not time_check():
            improved = False
            curr_path = best_path[:]
            
            # Try extending from the end
            for _ in range(10):
                if time_check():
                    break
                curr = curr_path[-1]
                candidates = [v for v in adj[curr] if v in unvisited]
                if not candidates:
                    break
                candidates.sort(key=lambda v: bin(adj_mask[v] & ~((1 << 100) - 1)).count('1'))
                # Actually, count unvisited neighbors
                candidates.sort(key=lambda v: sum(1 for w in adj[v] if w in unvisited))
                next_v = candidates[0]
                curr_path.append(next_v)
                unvisited.discard(next_v)
                improved = True
            
            # Try extending from the front
            for _ in range(10):
                if time_check():
                    break
                curr = curr_path[0]
                candidates = [v for v in adj[curr] if v in unvisited]
                if not candidates:
                    break
                candidates.sort(key=lambda v: sum(1 for w in adj[v] if w in unvisited))
                next_v = candidates[0]
                curr_path.insert(0, next_v)
                unvisited.discard(next_v)
                improved = True
            
            if len(curr_path) > len(best_path):
                best_path = curr_path
        
        # Try to insert unvisited vertices in the middle
        for _ in range(5):
            if time_check():
                break
            unvisited = set(range(n)) - set(best_path)
            if not unvisited:
                break
            
            improved_insert = True
            while improved_insert and not time_check():
                improved_insert = False
                for i in range(len(best_path) - 1):
                    if time_check():
                        break
                    a, b = best_path[i], best_path[i+1]
                    # Check if any unvisited vertex can be inserted between a and b
                    for v in unvisited:
                        if v in adj[a] and v in adj[b]:
                            # Insert v between a and b
                            new_path = best_path[:i+1] + [v] + best_path[i+1:]
                            # Verify it's still valid (it should be since we only inserted)
                            best_path = new_path
                            unvisited.discard(v)
                            improved_insert = True
                            break
    
    return best_path if best_path else [0]
