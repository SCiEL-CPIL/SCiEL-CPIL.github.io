import time
import sys

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    adj_list = [sorted(adj[i]) for i in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    adj_list = [sorted(adj[i]) for i in range(n)]
    
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
                for nei in adj[node]:
                    if not visited[nei]:
                        visited[nei] = True
                        stack.append(nei)
            components.append(comp)
    
    # Sort by size descending
    components.sort(key=len, reverse=True)
    
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin
    
    best_path = []
    best_len = 0
    
    def is_valid_path(path):
        if not path:
            return False
        if len(set(path)) != len(path):
            return False
        for i in range(len(path) - 1):
            if path[i+1] not in adj[path[i]]:
                return False
        return True
    
    def update_best(path):
        nonlocal best_path, best_len
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
    
    def greedy_extend(path, visited_set, max_depth, time_limit):
        """Greedy extension using Warnsdorff heuristic."""
        current_path = list(path)
        current_visited = set(visited_set)
        
        while len(current_path) < max_depth:
            if time.time() > time_limit:
                break
            last = current_path[-1]
            candidates = [v for v in adj_list[last] if v not in current_visited]
            if not candidates:
                break
            # Warnsdorff: pick neighbor with fewest unvisited neighbors
            candidates.sort(key=lambda v: (len([u for u in adj_list[v] if u not in current_visited]), v))
            next_v = candidates[0]
            current_path.append(next_v)
            current_visited.add(next_v)
            update_best(current_path)
        
        return current_path
    
    def extend_both_ends(path, visited_set, max_depth, time_limit):
        """Try to extend the path from both ends repeatedly."""
        current_path = list(path)
        current_visited = set(visited_set)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Try extending from the left end
            if len(current_path) >= 1:
                left_end = current_path[0]
                candidates = [v for v in adj_list[left_end] if v not in current_visited]
                if candidates:
                    candidates.sort(key=lambda v: (len([u for u in adj_list[v] if u not in current_visited]), v))
                    next_v = candidates[0]
                    current_path.insert(0, next_v)
                    current_visited.add(next_v)
                    improved = True
                    update_best(current_path)
            
            # Try extending from the right end
            if len(current_path) >= 1:
                right_end = current_path[-1]
                candidates = [v for v in adj_list[right_end] if v not in current_visited]
                if candidates:
                    candidates.sort(key=lambda v: (len([u for u in adj_list[v] if u not in current_visited]), v))
                    next_v = candidates[0]
                    current_path.append(next_v)
                    current_visited.add(next_v)
                    improved = True
                    update_best(current_path)
        
        return current_path
    
    def two_opt_improve(path, time_limit):
        """Apply 2-opt local search to rearrange the path for better extension opportunities."""
        current_path = list(path)
        improved = True
        
        while improved and time.time() < time_limit:
            improved = False
            plen = len(current_path)
            
            for i in range(plen - 2):
                if time.time() > time_limit:
                    return current_path
                for j in range(i + 2, plen):
                    # Check if reversing segment [i+1..j] creates valid edges
                    if j + 1 < plen:
                        # Need edge current_path[i] -> current_path[j] and current_path[i+1] -> current_path[j+1]
                        if current_path[j] in adj[current_path[i]] and current_path[i+1] in adj[current_path[j+1]]:
                            new_path = current_path[:i+1] + current_path[i+1:j+1][::-1] + current_path[j+1:]
                            # Verify all consecutive pairs are edges
                            valid = True
                            for k in range(len(new_path) - 1):
                                if new_path[k+1] not in adj[new_path[k]]:
                                    valid = False
                                    break
                            if valid:
                                current_path = new_path
                                improved = True
                                update_best(current_path)
                                break
                    else:
                        # j is the last element, need edge current_path[i] -> current_path[j]
                        if current_path[j] in adj[current_path[i]]:
                            new_path = current_path[:i+1] + current_path[i+1:j+1][::-1]
                            valid = True
                            for k in range(len(new_path) - 1):
                                if new_path[k+1] not in adj[new_path[k]]:
                                    valid = False
                                    break
                            if valid:
                                current_path = new_path
                                improved = True
                                update_best(current_path)
                                break
                if improved:
                    break
        
        return current_path
    
    def dfs_warnsdorff(start, max_depth, time_limit):
        """DFS with Warnsdorff heuristic and backtracking to find longest path."""
        nonlocal best_path, best_len
        
        path = [start]
        visited_set = {start}
        
        def extend_from(current_path, current_visited):
            nonlocal best_path, best_len
            
            if time.time() > time_limit:
                return False
            
            last = current_path[-1]
            candidates = [v for v in adj_list[last] if v not in current_visited]
            
            if not candidates:
                return False
            
            # Warnsdorff ordering
            candidates.sort(key=lambda v: (len([u for u in adj_list[v] if u not in current_visited]), v))
            
            for v in candidates:
                if time.time() > time_limit:
                    return False
                
                current_path.append(v)
                current_visited.add(v)
                
                if len(current_path) > best_len:
                    best_len = len(current_path)
                    best_path = list(current_path)
                
                if len(current_path) < max_depth:
                    extend_from(current_path, current_visited)
                
                # Backtrack
                current_path.pop()
                current_visited.discard(v)
            
            return True
        
        # Start the search
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
        
        if len(path) < max_depth:
            extend_from(path, visited_set)
        
        return best_path
    
    def full_search_cycle(start, time_budget):
        """Complete search cycle: DFS, extend both ends, 2-opt, extend both ends again."""
        if time.time() > deadline:
            return
        
        # Phase 1: DFS from start
        time_limit1 = min(time.time() + time_budget * 0.4, deadline - 5.0)
        dfs_warnsdorff(start, n, time_limit1)
        
        if time.time() > deadline:
            return
        
        # Phase 2: Extend both ends of best path
        if best_path:
            visited_set = set(best_path)
            time_limit2 = min(time.time() + time_budget * 0.2, deadline - 3.0)
            extended = extend_both_ends(best_path, visited_set, n, time_limit2)
        
        if time.time() > deadline:
            return
        
        # Phase 3: 2-opt improvement
        if best_path and len(best_path) > 3:
            time_limit3 = min(time.time() + time_budget * 0.3, deadline - 2.0)
            improved = two_opt_improve(best_path, time_limit3)
        
        if time.time() > deadline:
            return
        
        # Phase 4: Final extension from both ends
        if best_path:
            visited_set = set(best_path)
            time_limit4 = min(time.time() + time_budget * 0.1, deadline - 1.0)
            extended = extend_both_ends(best_path, visited_set, n, time_limit4)
    
    # Main search strategy
    largest_comp = components[0] if components else []
    
    if not largest_comp:
        return [0]
    
    # Phase 1: Quick greedy from all vertices in largest component to get initial best
    for start in largest_comp:
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        if remaining < 0.1:
            break
        time_budget = min(remaining, 0.05)
        greedy_extend([start], {start}, n, time_budget)
    
    # Phase 2: Full search cycles from promising starts
    # Start with vertices that are endpoints of current best path, then others
    if best_path:
        endpoints = [best_path[0], best_path[-1]]
        other_starts = [v for v in largest_comp if v not in endpoints]
    else:
        endpoints = []
        other_starts = largest_comp
    
    # Try endpoints first
    for start in endpoints:
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        if remaining < 1.0:
            break
        time_budget = min(remaining, 5.0)
        full_search_cycle(start, time_budget)
    
    # Try other starts
    for start in other_starts:
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        if remaining < 1.0:
            break
        time_budget = min(remaining, 3.0)
        full_search_cycle(start, time_budget)
    
    # Phase 3: Final polish - if we have time, do more 2-opt and extension passes
    if time.time() < deadline and best_path:
        remaining = deadline - time.time()
        if remaining > 2.0:
            # Multiple 2-opt passes
            for _ in range(3):
                if time.time() > deadline:
                    break
                time_budget = min(remaining / 3, deadline - time.time() - 1.0)
                if time_budget < 0.5:
                    break
                if best_path and len(best_path) > 3:
                    two_opt_improve(best_path, time.time() + time_budget)
                if best_path:
                    visited_set = set(best_path)
                    extend_both_ends(best_path, visited_set, n, time.time() + time_budget)
    
    # Ensure we return a valid non-empty path
    if not best_path:
        # Fallback: return any single vertex
        if components:
            best_path = [components[0][0]]
        else:
            best_path = [0]
    
    # Final validation
    if not is_valid_path(best_path):
        # Try to fix by finding a valid path
        for v in largest_comp:
            test_path = [v]
            if is_valid_path(test_path):
                best_path = test_path
                break
    
    return best_path
