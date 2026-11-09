"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
from itertools import combinations

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']

# Build edge index map: (min(u,v), max(u,v)) -> 1-based arrival index
edge_idx = {}
for i, (u, v) in enumerate(EDGE_ORDER):
    key = (min(u, v), max(u, v))
    edge_idx[key] = i + 1  # 1-based

# Build adjacency: for each vertex, list of (neighbor, arrival_index)
adj = [[] for _ in range(N)]
for i, (u, v) in enumerate(EDGE_ORDER):
    idx = i + 1
    adj[u].append((v, idx))
    adj[v].append((u, idx))
# Sort each vertex's neighbors by arrival index (early edges first)
for v in range(N):
    adj[v].sort(key=lambda x: x[1])

def get_edge_idx(u, v):
    return edge_idx[(min(u, v), max(u, v))]

def find_hamilton_cycle(deadline):
    """Find a Hamilton cycle minimizing the max edge arrival index."""
    best_cycle = None
    best_max_idx = N * (N - 1) // 2  # worst case: last edge
    
    # Try different starting vertices
    for start in range(N):
        if time.time() > deadline:
            break
        # Backtracking: build path from start
        path = [start]
        visited = [False] * N
        visited[start] = True
        current_max = 0
        
        def backtrack(pos):
            nonlocal best_cycle, best_max_idx, current_max
            if time.time() > deadline:
                return False
            
            if pos == N:
                # Check closing edge
                last = path[-1]
                first = path[0]
                close_idx = get_edge_idx(first, last)
                total_max = max(current_max, close_idx)
                if total_max < best_max_idx:
                    best_max_idx = total_max
                    best_cycle = path[:]
                return True
            
            # Try neighbors in order of arrival index
            for (nb, idx) in adj[path[pos - 1]]:
                if visited[nb]:
                    continue
                # Pruning: if this edge already exceeds best, skip
                new_max = max(current_max, idx)
                if new_max >= best_max_idx:
                    continue
                
                path.append(nb)
                visited[nb] = True
                old_max = current_max
                current_max = new_max
                
                if backtrack(pos + 1):
                    return True
                
                path.pop()
                visited[nb] = False
                current_max = old_max
                
                if time.time() > deadline:
                    return False
            
            return False
        
        backtrack(1)
        if best_cycle is not None and best_max_idx <= 15:
            break  # Good enough, stop early
    
    return best_cycle, best_max_idx

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s to be safe within 90s limit
    
    cycle, max_idx = find_hamilton_cycle(deadline)
    
    if cycle is None:
        # Fallback: return trivial cycle 0,1,2,...,23
        return list(range(N))
    
    return cycle
# EVOLVE-BLOCK-END
