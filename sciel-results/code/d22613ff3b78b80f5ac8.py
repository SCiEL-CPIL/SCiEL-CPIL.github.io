"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import heapq
from collections import deque

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find connected components
    visited = [False] * n
    components = []
    for i in range(n):
        if not visited[i]:
            comp = []
            queue = deque([i])
            visited[i] = True
            while queue:
                node = queue.popleft()
                comp.append(node)
                for neighbor in adj[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            components.append(comp)
    
    # Find the largest component
    largest_comp = max(components, key=len)
    comp_set = set(largest_comp)
    comp_size = len(largest_comp)
    
    if comp_size <= 2:
        return largest_comp
    
    # Helper to calculate slack (number of unvisited neighbors in the component)
    def get_slack(vertex, used):
        return len(adj[vertex] & comp_set - used)
    
    best_path = []
    
    # Strategy: Multiple randomized DFS attempts with backtracking limited by depth/time
    # The "matching slack" heuristic: prioritize neighbors with low slack (more constrained)
    # This is similar to the "Warnsdorff's rule" for Knight's tour but adapted for general graphs.
    
    rng = random.Random(748) # Use the task's seed for reproducibility of the search strategy
    
    # We will run several attempts. Each attempt is a DFS with limited backtracking.
    # If we get stuck, we backtrack a few steps and try a different branch.
    
    max_attempts = 20
    backtrack_limit = 100 # Total backtracks allowed per attempt
    
    for attempt in range(max_attempts):
        # Choose a starting vertex.
        # Vertices with low initial degree are good candidates for endpoints.
        # Let's pick a random vertex from the component, but bias towards low degree.
        # Or just random for diversity.
        start = rng.choice(largest_comp)
        
        path = [start]
        used = {start}
        backtracks = 0
        
        # DFS with limited backtracking
        # We maintain a stack of states to allow backtracking
        # State: (current_vertex, path_copy, used_copy, next_index_to_try)
        # For efficiency, we simulate this with recursion or a manual stack.
        # Given N=100, recursion depth is safe.
        
        def dfs_with_backtrack(current, path, used, backtracks_remaining):
            if len(path) == comp_size:
                return path, True, backtracks_remaining
            
            current = path[-1]
            # Get candidates
            candidates = [v for v in adj[current] if v in comp_set and v not in used]
            
            if not candidates:
                # Dead end
                return None, False, backtracks_remaining
            
            # Sort candidates by slack (lowest slack first)
            # Tie-break randomly to explore different paths
            candidates.sort(key=lambda v: (get_slack(v, used), rng.random()))
            
            for next_v in candidates:
                # Try extending
                path.append(next_v)
                used.add(next_v)
                
                result_path, success, remaining_backtracks = dfs_with_backtrack(next_v, path, used, backtracks_remaining)
                
                if success:
                    return result_path, True, remaining_backtracks
                
                # Backtrack
                path.pop()
                used.remove(next_v)
                
                # We only "spend" a backtrack if we explored a branch that failed
                # But we want to limit total backtracks.
                # If we just tried one branch and it failed, we spent 1 backtrack.
                # If we have no backtracks left, we must accept the current path as a local optimum
                # and stop this attempt.
                backtracks_remaining -= 1
                if backtracks_remaining <= 0:
                    # We cannot backtrack further. 
                    # The current path (after popping the failed branch) is the best we can do in this line.
                    # Actually, if we pop and then run out of backtracks, we are stuck at the current node with no way to try other branches.
                    # So we return the current path as the result for this attempt.
                    return path, False, 0
            
            # If we tried all candidates and none succeeded, and we have backtracks, 
            # we should return failure to the caller so it can backtrack further.
            return None, False, backtracks_remaining

        # To avoid recursion depth issues or just to manage the "limited backtracking" logic more cleanly:
        # The above recursion might be too deep or not strictly limiting backtracks globally in the way intended.
        # Let's use a simpler iterative approach with a "retry" mechanism.
        
        # Simpler approach: 
        # 1. Greedy path with heuristic.
        # 2. If stuck, try to "repair" by local search or restart.
        
        # Let's stick to a robust greedy with randomized tie-breaking and multiple restarts.
        # The "backtracking" in the previous thought was complex. 
        # A "Limited Backtracking DFS" is better implemented as:
        # Try to find a path. If we hit a dead end, check if we can backtrack 
        # to a previous node and try a different neighbor. Limit total backtracks.
        
        path = [start]
        used = {start}
        # Stack of (node, list_of_untried_neighbors)
        # We need to track which neighbors have been tried for each node in the path.
        
        # Actually, let's use a simpler heuristic search:
        # 1. Start at a vertex.
        # 2. Extend greedily using slack heuristic.
        # 3. If stuck, look for a "pivot" in the path where a different choice could have led to a longer path.
        # This is effectively local search on the path.
        
        # Let's implement a "Greedy with Backtracking (Limited)" properly.
        
        def search_with_limited_backtrack(start_vertex, max_backtracks):
            path = [start_vertex]
            used = {start_vertex}
            # stack of choices made
            # Each element: (current_node, [unvisited_neighbors])
            # We need to remember which neighbors we've already tried for each node in the path.
            
            # This is tricky to implement efficiently in Python without full recursion or heavy state copying.
            # Given N=100, copying the `used` set and `path` list for each backtrack is O(N).
            # Max backtracks 100 * N=100 = 10,000 operations. This is fast.
            
            # We will use a recursive function that returns the best path found within the backtrack budget.
            
            import sys
            sys.setrecursionlimit(200)
            
            def rec_dfs(current, path, used, budget):
                if len(path) == comp_size:
                    return path
                
                current = path[-1]
                candidates = [v for v in adj[current] if v in comp_set and v not in used]
                
                if not candidates:
                    return path # Dead end, return current path
                
                # Sort by slack
                candidates.sort(key=lambda v: (get_slack(v, used), rng.random()))
                
                best_subpath = None
                
                for v in candidates:
                    # Try v
                    new_path = path + [v]
                    new_used = used | {v}
                    
                    # Recurse
                    subpath = rec_dfs(v, new_path, new_used, budget - 1)
                    
                    if subpath is not None:
                        if best_subpath is None or len(subpath) > len(best_subpath):
                            best_subpath = subpath
                            if len(best_subpath) == comp_size:
                                return best_subpath
                    
                    # If budget is 0, we can't try other candidates for this node
                    # because we "spent" the budget on the first choice? 
                    # No, the budget should limit the total number of times we backtrack.
                    # A simpler model: 
                    # We allow `budget` backtracks total.
                    # If we try a branch and it fails, we backtrack. That costs 1.
                    # If budget is 0, we must commit to the first choice?
                    # Or we must stop exploring alternatives?
                    
                    # Let's refine: 
                    # If budget > 0, we can try the next candidate.
                    # If budget == 0, we try the first candidate and stick with its result (even if it's a dead end, we return that path).
                    
                    if budget <= 0:
                        break
                
                if best_subpath is None:
                    return path
                return best_subpath
            
            return rec_dfs(start_vertex, [start_vertex], {start_vertex}, max_backtracks)

        # Run multiple attempts with different starts and budgets
        for b in [0, 5, 10, 20, 50]:
            p = search_with_limited_backtrack(start, b)
            if len(p) > len(best_path):
                best_path = p
                if len(best_path) == comp_size:
                    return best_path
            
            # Try a different start if we haven't found a full path
            if len(best_path) < comp_size:
                 start = rng.choice(largest_comp)
        
        # Early exit if Hamiltonian path found
        if len(best_path) == comp_size:
            break
        
    # Final check and return
    # Verify validity
    for i in range(len(best_path) - 1):
        if best_path[i + 1] not in adj[best_path[i]]:
             # This should not happen if logic is correct, but just in case
             # Fallback to a simple valid path
             return [best_path[0]]
             
    return best_path
# EVOLVE-BLOCK-END
