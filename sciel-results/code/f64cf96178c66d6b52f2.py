"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

# Increase recursion limit just in case, though 24 is small
sys.setrecursionlimit(10000)

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

def construct():
    n = P['n']
    edges = P['edge_order']
    
    # Sort edges by their arrival index (1-based in problem, 0-based in list)
    # The list is already in arrival order.
    # So edge i has index i+1.
    
    adj = [[] for _ in range(n)]
    degree = [0] * n
    
    start_time = time.time()
    deadline = start_time + 85.0 # Leave 5s margin
    
    # We want to find the minimum K such that G_K has a Hamilton cycle.
    # G_K contains edges 0..K-1.
    
    for k in range(len(edges)):
        u, v = edges[k]
        adj[u].append(v)
        adj[v].append(u)
        degree[u] += 1
        degree[v] += 1
        
        # Necessary condition: min degree >= 2
        if min(degree) < 2:
            continue
            
        # Check time
        if time.time() > deadline:
            break
            
        # Check connectivity
        if not is_connected(adj, n):
            continue
            
        # Try to find Hamiltonian cycle
        cycle = find_hamiltonian_cycle(adj, n)
        if cycle is not None:
            return cycle
            
    # Fallback: return seed (identity)
    return list(range(n))

def is_connected(adj, n):
    visited = [False] * n
    stack = [0]
    visited[0] = True
    count = 1
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                count += 1
                stack.append(v)
    return count == n

def find_hamiltonian_cycle(adj, n):
    # Use DFS with Warnsdorff's heuristic
    # State: path, visited_mask
    # Start at 0
    
    visited_mask = 1 << 0
    path = [0]
    
    # Precompute neighbor lists for faster access? No, adj is fine.
    
    def dfs(current):
        if len(path) == n:
            # Check if closes
            if 0 in adj[current]:
                return True
            return False
        
        candidates = []
        for neighbor in adj[current]:
            if not (visited_mask >> neighbor) & 1:
                # Calculate heuristic score
                # Number of unvisited neighbors
                unvisited_count = 0
                for nb in adj[neighbor]:
                    if not (visited_mask >> nb) & 1:
                        unvisited_count += 1
                # If we are at the second to last step, the last node must connect to 0
                # If len(path) == n-1, we are choosing the last node.
                # The last node must be connected to 0.
                if len(path) == n - 1:
                    if 0 not in adj[neighbor]:
                        continue
                candidates.append((unvisited_count, neighbor))
                
        if not candidates:
            return False
            
        # Sort by heuristic: fewest unvisited neighbors first
        candidates.sort()
        
        for _, neighbor in candidates:
            path.append(neighbor)
            visited_mask |= (1 << neighbor)
            
            if dfs(neighbor):
                return True
                
            path.pop()
            visited_mask &= ~(1 << neighbor)
            
        return False
        
    if dfs(0):
        return path
    return None

# EVOLVE-BLOCK-END
