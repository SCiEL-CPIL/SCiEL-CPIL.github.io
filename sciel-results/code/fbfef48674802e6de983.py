import time
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment

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

def find_hamilton_cycle_fast(adj, n, deadline):
    """Find Hamilton cycle using assignment-based approach with backtracking fallback."""
    # Quick degree check
    for v in range(n):
        if len(adj[v]) < 2:
            return None
    
    # Build cost matrix for assignment problem
    # Cost[i][j] = position of edge (i,j) if it exists, else INF
    INF_COST = 10**9
    cost = np.full((n, n), INF_COST, dtype=np.float64)
    
    for u in range(n):
        for v in range(n):
            if u != v and v in adj[u]:
                cost[u][v] = get_pos(u, v)
    
    # Solve assignment problem (minimum cost perfect matching)
    row_ind, col_ind = linear_sum_assignment(cost)
    
    # Check if assignment is valid (all edges exist)
    if any(cost[r, c] >= INF_COST for r, c in zip(row_ind, col_ind)):
        # Assignment found invalid edges, fall back to backtracking
        return _backtrack_hamilton(adj, n, deadline)
    
    # Extract cycles from the assignment
    # Each vertex has exactly one outgoing and one incoming edge
    # Build the permutation
    perm = [0] * n
    for r, c in zip(row_ind, col_ind):
        perm[r] = c
    
    # Find all cycles in the permutation
    visited = [False] * n
    cycles = []
    for start in range(n):
        if not visited[start]:
            cycle = []
            cur = start
            while not visited[cur]:
                visited[cur] = True
                cycle.append(cur)
                cur = perm[cur]
            cycles.append(cycle)
    
    # Check if we have a single Hamilton cycle
    if len(cycles) == 1 and len(cycles[0]) == n:
        return cycles[0]
    
    # We have multiple cycles - try to merge them using 2-opt style swaps
    # Try all pairs of cycles and see if we can merge them
    merged = False
    for attempt in range(100):
        if time.time() > deadline:
            break
        
        if len(cycles) == 1:
            break
        
        # Try to merge two cycles
        found_merge = False
        for i in range(len(cycles)):
            for j in range(i + 1, len(cycles)):
                if found_merge:
                    break
                c1 = cycles[i]
                c2 = cycles[j]
                
                # Try all pairs of edges to swap
                for a1 in range(len(c1)):
                    if found_merge:
                        break
                    v1 = c1[a1]
                    v1_next = c1[(a1 + 1) % len(c1)]
                    
                    for a2 in range(len(c2)):
                        v2 = c2[a2]
                        v2_next = c2[(a2 + 1) % len(c2)]
                        
                        # Check if we can swap: remove edges (v1,v1_next) and (v2,v2_next)
                        # Add edges (v1,v2) and (v1_next,v2_next)
                        if v2 in adj[v1] and v2_next in adj[v1_next]:
                            # Perform the merge
                            new_cycle = c1[:a1+1] + c2[a2:] + c1[a1+1:] + c2[:a2]
                            cycles = [c for idx, c in enumerate(cycles) if idx not in (i, j)]
                            cycles.append(new_cycle)
                            found_merge = True
                            break
                        
                        # Try alternative: (v1, v2_next) and (v1_next, v2)
                        if v2_next in adj[v1] and v2 in adj[v1_next]:
                            new_cycle = c1[:a1+1] + [v2] + c2[a2+1:] + c1[a1+1:] + c2[:a2+1]
                            cycles = [c for idx, c in enumerate(cycles) if idx not in (i, j)]
                            cycles.append(new_cycle)
                            found_merge = True
                            break
        
        if not found_merge:
            # Can't merge any more, try backtracking
            return _backtrack_hamilton(adj, n, deadline)
    
    if len(cycles) == 1 and len(cycles[0]) == n:
        return cycles[0]
    
    # Fallback to backtracking
    return _backtrack_hamilton(adj, n, deadline)


def _backtrack_hamilton(adj, n, deadline):
    """Backtracking Hamilton cycle finder as fallback."""
    # Fix vertex 0 as start
    path = [0]
    used = [False] * n
    used[0] = True
    
    # Pre-sort neighbors by degree for MRV heuristic
    neighbor_order = [list(adj[v]) for v in range(n)]
    for v in range(n):
        neighbor_order[v].sort(key=lambda x: len(adj[x]))
    
    def backtrack(depth):
        if time.time() > deadline:
            return None
        
        if depth == n:
            last = path[-1]
            if 0 in adj[last]:
                return path[:]
            return None
        
        last = path[-1]
        
        # Try neighbors in MRV order
        for nxt in neighbor_order[last]:
            if used[nxt]:
                continue
            
            # Pruning: if this is the second-to-last step, check closing edge feasibility
            if depth == n - 1:
                remaining = [v for v in range(n) if not used[v] and v != nxt]
                if len(remaining) == 1:
                    r = remaining[0]
                    if r not in adj[nxt] or 0 not in adj[r]:
                        continue
                elif len(remaining) == 0:
                    if 0 not in adj[nxt]:
                        continue
            
            # Degree constraint pruning
            if depth >= 2:
                unvisited_check = [v for v in range(n) if not used[v] and v != nxt]
                if unvisited_check:
                    valid = True
                    for v in unvisited_check:
                        deg = 0
                        for w in adj[v]:
                            if w == nxt:
                                continue
                            if used[w] and w != nxt:
                                deg += 1
                            elif not used[w] and w != v:
                                deg += 1
                        if nxt in adj[v]:
                            deg += 1
                        if deg < 2:
                            valid = False
                            break
                    if not valid:
                        continue
            
            used[nxt] = True
            path.append(nxt)
            
            # Connectivity pruning
            if depth < n - 1:
                unvisited = [v for v in range(n) if not used[v]]
                if len(unvisited) > 0:
                    visited_u = set()
                    stack = [unvisited[0]]
                    visited_u.add(unvisited[0])
                    while stack:
                        u = stack.pop()
                        for w in adj[u]:
                            if w in unvisited and w not in visited_u:
                                visited_u.add(w)
                                stack.append(w)
                    if len(visited_u) != len(unvisited):
                        path.pop()
                        used[nxt] = False
                        continue
                    
                    path_set = set(path)
                    has_connection = False
                    for v in unvisited:
                        for w in adj[v]:
                            if w in path_set:
                                has_connection = True
                                break
                        if has_connection:
                            break
                    if not has_connection:
                        path.pop()
                        used[nxt] = False
                        continue
            
            result = backtrack(depth + 1)
            if result is not None:
                return result
            
            path.pop()
            used[nxt] = False
        
        return None
    
    return backtrack(1)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s kill
    
    n = N
    total_edges = n * (n - 1) // 2
    
    best_cycle = None
    best_score = total_edges + 1
    
    # Pre-build sorted list of edges by position for early break
    sorted_edges = sorted([(pos, u, v) for (u, v), pos in edge_pos.items()])
    
    # Binary search for minimum threshold T
    lo, hi = 24, total_edges
    
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        
        # Build subgraph with edges at position <= mid
        adj = [[] for _ in range(n)]
        for pos, u, v in sorted_edges:
            if pos > mid:
                break
            adj[u].append(v)
            adj[v].append(u)
        
        # Check for Hamilton cycle using assignment-based method
        cycle = find_hamilton_cycle_fast(adj, n, deadline)
        
        if cycle is not None:
            best_cycle = cycle
            best_score = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best_cycle is not None:
        return best_cycle
    
    # Fallback: use full graph
    adj_full = [[] for _ in range(n)]
    for (u, v) in EDGE_ORDER:
        adj_full[u].append(v)
        adj_full[v].append(u)
    
    cycle = find_hamilton_cycle_fast(adj_full, n, deadline)
    if cycle is not None:
        return cycle
    
    # Final fallback: identity permutation
    return list(range(n))
