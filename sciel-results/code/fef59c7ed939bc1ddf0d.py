import time
import sys
from itertools import permutations

N = 24
EDGE_ORDER = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

# Build edge position map
edge_pos = {}
for i, (u, v) in enumerate(EDGE_ORDER):
    a, b = min(u, v), max(u, v)
    edge_pos[(a, b)] = i + 1

def get_pos(u, v):
    a, b = min(u, v), max(u, v)
    return edge_pos[(a, b)]

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    n = N
    total_edges = n * (n - 1) // 2  # 276
    
    # Precompute: for each vertex, sort neighbors by edge position
    neighbors_by_pos = [[] for _ in range(n)]
    for (u, v), pos in edge_pos.items():
        neighbors_by_pos[u].append((pos, v))
        neighbors_by_pos[v].append((pos, u))
    for v in range(n):
        neighbors_by_pos[v].sort()
        neighbors_by_pos[v] = [x[1] for x in neighbors_by_pos[v]]
    
    # Also build adjacency sets for quick lookup
    adj_sets = [set() for _ in range(n)]
    for (u, v) in edge_pos:
        adj_sets[u].add(v)
        adj_sets[v].add(u)
    
    best_cycle = None
    best_score = total_edges + 1
    
    # Binary search for minimum threshold
    lo, hi = 24, total_edges
    
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        
        # Build subgraph with edges at position <= mid
        adj = [[] for _ in range(n)]
        for (u, v), pos in edge_pos.items():
            if pos <= mid:
                adj[u].append(v)
                adj[v].append(u)
        
        # Quick degree check
        feasible = True
        for v in range(n):
            if len(adj[v]) < 2:
                feasible = False
                break
        
        if not feasible:
            lo = mid + 1
            continue
        
        # Check connectivity
        visited = [False] * n
        stack = [0]
        visited[0] = True
        count = 0
        while stack:
            u = stack.pop()
            count += 1
            for w in adj[u]:
                if not visited[w]:
                    visited[w] = True
                    stack.append(w)
        if count != n:
            lo = mid + 1
            continue
        
        # Backtracking Hamilton cycle with strong pruning
        # Fix vertex 0 as start
        path = [0]
        used = [False] * n
        used[0] = True
        
        # Precompute adjacency as sets for this subgraph
        adj_set = [set(x) for x in adj]
        
        # Sort neighbors by degree (ascending) for better pruning
        adj_sorted = [sorted(adj[v], key=lambda x: len(adj[x])) for v in range(n)]
        
        found = False
        
        def backtrack():
            nonlocal found
            if time.time() > deadline:
                return False
            
            if len(path) == n:
                last = path[-1]
                if 0 in adj_set[last]:
                    return True
                return False
            
            last = path[-1]
            
            # Pruning: check if 0 still has enough available neighbors
            # 0 needs at least 1 unvisited neighbor (for the closing edge)
            # unless we're at the last step
            if len(path) < n - 1:
                # 0 needs at least one unvisited neighbor that can connect back
                # Actually 0 just needs one neighbor that will be the last vertex
                # For now, skip this complex pruning
                pass
            
            for nxt in adj_sorted[last]:
                if not used[nxt]:
                    # Pruning: if nxt is the last remaining vertex, it must connect to 0
                    if len(path) == n - 2:
                        # One vertex left after nxt
                        remaining = [v for v in range(n) if not used[v] and v != nxt]
                        if len(remaining) == 1:
                            r = remaining[0]
                            if r not in adj_set[nxt] or 0 not in adj_set[r]:
                                continue
                    
                    used[nxt] = True
                    path.append(nxt)
                    
                    # Additional pruning: check if remaining unvisited vertices 
                    # can still form a path from nxt back to 0
                    # Simple check: the last unvisited vertex must connect to 0
                    if len(path) == n - 1:
                        remaining = [v for v in range(n) if not used[v]]
                        if len(remaining) == 1:
                            r = remaining[0]
                            if 0 not in adj_set[r]:
                                path.pop()
                                used[nxt] = False
                                continue
                    
                    if backtrack():
                        return True
                    
                    path.pop()
                    used[nxt] = False
            
            return False
        
        if backtrack():
            best_cycle = path[:]
            best_score = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best_cycle is not None:
        return best_cycle
    
    # Fallback: try with full graph
    adj_full = [[] for _ in range(n)]
    for (u, v) in edge_pos:
        adj_full[u].append(v)
        adj_full[v].append(u)
    
    adj_set_full = [set(x) for x in adj_full]
    adj_sorted_full = [sorted(adj_full[v], key=lambda x: len(adj_full[x])) for v in range(n)]
    
    path = [0]
    used = [False] * n
    used[0] = True
    
    def backtrack_full():
        if time.time() > deadline:
            return False
        if len(path) == n:
            last = path[-1]
            if 0 in adj_set_full[last]:
                return True
            return False
        
        last = path[-1]
        for nxt in adj_sorted_full[last]:
            if not used[nxt]:
                if len(path) == n - 2:
                    remaining = [v for v in range(n) if not used[v] and v != nxt]
                    if len(remaining) == 1:
                        r = remaining[0]
                        if r not in adj_set_full[nxt] or 0 not in adj_set_full[r]:
                            continue
                
                used[nxt] = True
                path.append(nxt)
                
                if len(path) == n - 1:
                    remaining = [v for v in range(n) if not used[v]]
                    if len(remaining) == 1:
                        r = remaining[0]
                        if 0 not in adj_set_full[r]:
                            path.pop()
                            used[nxt] = False
                            continue
                
                if backtrack_full():
                    return True
                
                path.pop()
                used[nxt] = False
        
        return False
    
    if backtrack_full():
        return path[:]
    
    # Final fallback
    return list(range(n))
