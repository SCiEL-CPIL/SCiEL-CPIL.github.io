"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import sys

# Increase recursion limit for deep DFS
sys.setrecursionlimit(10000)

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Pre-compute degrees
    degrees = [len(adj[i]) for i in range(n)]
    
    best_path = []
    best_len = 0
    start_time = time.time()
    max_time = 55.0
    
    # Helper to check time
    def time_exceeded():
        return time.time() - start_time > max_time

    # Warnsdorff's Rule: Visit the neighbor with the fewest unvisited neighbors
    def get_warnsdorff_order(current, visited):
        neighbors = [v for v in adj[current] if v not in visited]
        # Sort by number of unvisited neighbors
        neighbors.sort(key=lambda v: len(adj[v] - visited))
        return neighbors

    def dfs(path, visited):
        nonlocal best_path, best_len
        if time_exceeded():
            return
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
            
        current = path[-1]
        # Get neighbors ordered by Warnsdorff's rule
        candidates = get_warnsdorff_order(current, visited)
        
        for next_node in candidates:
            if time_exceeded():
                return
            path.append(next_node)
            visited.add(next_node)
            dfs(path, visited)
            path.pop()
            visited.remove(next_node)
            
            # Pruning: If we have already found a path of length N, we can stop
            if best_len == n:
                return

    # Strategy:
    # 1. Try starting from each vertex, but prioritize vertices with high degree? 
    #    Actually, for paths, starting from low degree might be better to avoid trapping?
    #    Or high degree to have more options?
    #    Let's try all starts, but break early if we find a full Hamiltonian path.
    
    # To speed up, we can limit the search depth or use iterative deepening?
    # With N=100, full DFS might be too slow if the branching factor is high.
    # However, the graph is sparse.
    
    # Let's add a simple heuristic: If the graph is disconnected, we can only search within the component.
    # Find connected components
    visited_comp = set()
    components = []
    for i in range(n):
        if i not in visited_comp:
            comp = []
            stack = [i]
            visited_comp.add(i)
            while stack:
                node = stack.pop()
                comp.append(node)
                for neighbor in adj[node]:
                    if neighbor not in visited_comp:
                        visited_comp.add(neighbor)
                        stack.append(neighbor)
            components.append(comp)
    
    # Sort components by size descending
    components.sort(key=lambda x: len(x), reverse=True)
    
    # Only search in the largest component, as it will contain the longest path
    # Actually, the longest path must be in the largest component.
    if components:
        largest_comp = components[0]
        # Map local indices? No, just filter starts.
        # We will run DFS starting from each node in the largest component.
        # But to avoid redundancy, we can just pick a few promising starts.
        
        # Let's try starting from each node in the largest component, but limit the number of trials per node to avoid timeout.
        # Or, better: Run a single DFS from a "good" start?
        # A good start is often a leaf or low-degree node in the potential path structure.
        
        # Let's try all starts in the largest component, but with a tighter time check.
        for start in largest_comp:
            if time_exceeded():
                break
            
            # Reset best_path if we are starting a new search? No, keep global best.
            # But we should not explore if the maximum possible path length in this component is less than best_len.
            if len(largest_comp) <= best_len:
                break # Can't beat current best
            
            path = [start]
            visited = {start}
            
            # To prevent infinite loops or excessive time on a single start, 
            # we can limit the recursion depth or use a heuristic.
            # But Warnsdorff is usually very fast for sparse graphs.
            
            dfs(path, visited)
            
            if best_len == len(largest_comp):
                break # Found Hamiltonian path in the component

    # If no path found (shouldn't happen), return a single node
    if not best_path:
        best_path = [0]
        
    return best_path
# EVOLVE-BLOCK-END
