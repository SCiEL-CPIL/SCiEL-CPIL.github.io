"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

def construct():
    start_time = time.time()
    n = P['n']
    edge_order = P['edge_order']
    num_edges = len(edge_order)
    
    # Precompute edge positions
    edge_pos = {}
    for i, (u, v) in enumerate(edge_order):
        edge_pos[(u, v)] = i
        edge_pos[(v, u)] = i
    
    def get_max_edge(cycle):
        """Get the maximum edge position in a cycle."""
        max_pos = 0
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            pos = edge_pos[(u, v)]
            if pos > max_pos:
                max_pos = pos
        return max_pos
    
    def greedy_build(threshold):
        """
        Build a Hamiltonian cycle greedily using edges with position < threshold.
        Returns the cycle or None if not found.
        """
        # Build adjacency list for edges with pos < threshold
        adj = [[] for _ in range(n)]
        for i in range(threshold):
            u, v = edge_order[i]
            adj[u].append(v)
            adj[v].append(u)
        
        # Sort adjacency lists by edge position to prioritize early edges
        for u in range(n):
            adj[u].sort(key=lambda v: edge_pos[(u, v)])
        
        start = 0
        path = [start]
        visited = [False] * n
        visited[start] = True
        
        for _ in range(n - 1):
            last = path[-1]
            found = False
            for next_v in adj[last]:
                if not visited[next_v]:
                    path.append(next_v)
                    visited[next_v] = True
                    found = True
                    break
            if not found:
                return None
        
        # Check if we can close the cycle
        last = path[-1]
        if edge_pos.get((last, start), float('inf')) < threshold:
            return path
        return None
    
    def two_opt_improve(cycle, max_iterations=1000):
        """
        Try to reduce the maximum edge position using 2-opt style moves.
        """
        if time.time() - start_time > 80:
            return cycle
        
        current_cycle = list(cycle)
        current_max = get_max_edge(current_cycle)
        improved = True
        
        while improved and time.time() - start_time < 80:
            improved = False
            for i in range(len(current_cycle) - 1):
                if time.time() - start_time > 80:
                    break
                for j in range(i + 2, len(current_cycle)):
                    if j == len(current_cycle) - 1 and i == 0:
                        continue  # Skip the closing edge pair
                    
                    # Try reversing the segment from i+1 to j
                    new_cycle = current_cycle[:i+1] + current_cycle[i+1:j+1][::-1] + current_cycle[j+1:]
                    new_max = get_max_edge(new_cycle)
                    
                    if new_max < current_max:
                        current_cycle = new_cycle
                        current_max = new_max
                        improved = True
        
        return current_cycle
    
    # Strategy: Start with a high threshold, find a cycle, then improve it
    # Use binary search to find the minimum threshold where a cycle exists
    
    lo = n
    hi = num_edges
    best_cycle = None
    best_score = num_edges + 1
    
    # First, get a valid cycle with the full set
    full_cycle = greedy_build(num_edges)
    if full_cycle:
        best_cycle = full_cycle
        best_score = get_max_edge(full_cycle) + 1
    
    # Binary search for the minimum threshold
    while lo < hi and time.time() - start_time < 70:
        mid = (lo + hi) // 2
        cycle = greedy_build(mid)
        if cycle:
            score = get_max_edge(cycle) + 1
            if score < best_score:
                best_score = score
                best_cycle = cycle
            hi = mid
        else:
            lo = mid + 1
    
    # Try to improve the best cycle found
    if best_cycle and time.time() - start_time < 80:
        improved = two_opt_improve(best_cycle)
        improved_score = get_max_edge(improved) + 1
        if improved_score < best_score:
            best_cycle = improved
            best_score = improved_score
    
    # Additional refinement: try different starting vertices and greedy orders
    if best_cycle and time.time() - start_time < 85:
        for start_v in range(n):
            if time.time() - start_time > 85:
                break
            
            # Try greedy build starting from start_v
            adj = [[] for _ in range(n)]
            for i in range(num_edges):
                u, v = edge_order[i]
                adj[u].append(v)
                adj[v].append(u)
            
            for u in range(n):
                adj[u].sort(key=lambda v: edge_pos[(u, v)])
            
            path = [start_v]
            visited = [False] * n
            visited[start_v] = True
            
            success = True
            for _ in range(n - 1):
                last = path[-1]
                found = False
                for next_v in adj[last]:
                    if not visited[next_v]:
                        path.append(next_v)
                        visited[next_v] = True
                        found = True
                        break
                if not found:
                    success = False
                    break
            
            if success:
                last = path[-1]
                if edge_pos.get((last, start_v), float('inf')) < num_edges:
                    score = get_max_edge(path) + 1
                    if score < best_score:
                        best_score = score
                        best_cycle = path
    
    if best_cycle:
        return best_cycle
    else:
        return list(range(n))

# EVOLVE-BLOCK-END
