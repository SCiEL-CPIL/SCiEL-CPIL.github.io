import time
import sys
from collections import defaultdict

N = 24
EDGE_ORDER = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

# Precompute edge positions
edge_pos = {}
for i, (u, v) in enumerate(EDGE_ORDER):
    a, b = min(u, v), max(u, v)
    edge_pos[(a, b)] = i + 1

def find_hamilton_cycle(adj, n, deadline):
    """
    Finds a Hamilton cycle in the graph defined by adj using backtracking.
    Returns the list of vertices in the cycle or None if not found.
    """
    # Quick checks
    for v in range(n):
        if len(adj[v]) < 2:
            return None
    
    # Connectivity check
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
        return None

    # Backtracking
    # Fix 0 as start
    path = [0]
    used = [False] * n
    used[0] = True
    
    # Pre-compute adjacency sets for O(1) lookup
    adj_sets = [set(adj[v]) for v in range(n)]
    
    def backtrack():
        if time.time() > deadline:
            return None
        
        if len(path) == n:
            # Check closing edge
            last = path[-1]
            if 0 in adj_sets[last]:
                return path[:]
            return None
        
        last = path[-1]
        
        # Heuristic: If only 1 vertex left, it must connect to last and 0
        if len(path) == n - 1:
            # Find the single unvisited vertex
            remaining = -1
            for v in range(n):
                if not used[v]:
                    remaining = v
                    break
            if remaining != -1:
                if remaining in adj_sets[last] and 0 in adj_sets[remaining]:
                    path.append(remaining)
                    return path[:]
            return None

        # Try neighbors
        # Sort neighbors by dynamic degree (number of unvisited neighbors)
        # This is the Warnsdorff heuristic
        neighbors = []
        for nxt in adj[last]:
            if not used[nxt]:
                # Count unvisited neighbors of nxt
                # Exclude 'last' and '0' from the count? 
                # Warnsdorff usually counts unvisited neighbors in the remaining graph.
                # For the last step, the '0' is the target.
                # Let's count unvisited neighbors.
                deg = 0
                for w in adj[nxt]:
                    if not used[w]:
                        deg += 1
                # Special case: if nxt is the last vertex to be visited, it needs to connect to 0.
                # But we don't know if it's the last one yet.
                # Standard Warnsdorff: sort by number of unvisited neighbors.
                neighbors.append((deg, nxt))
        
        # Sort by degree (ascending)
        neighbors.sort(key=lambda x: x[0])
        
        for deg, nxt in neighbors:
            if used[nxt]:
                continue
                
            # Pruning: Check if nxt is a cut-vertex or leads to isolation?
            # Simple check: If we pick nxt, does any other unvisited vertex become isolated?
            # An unvisited vertex w (w != nxt) is isolated if it has no neighbors in 
            # (Unvisited \ {nxt}) U {last, 0}.
            # Actually, the "available" nodes for the future path are Unvisited \ {nxt} plus 
            # the entry point 'last' (which is now 'nxt') and exit point '0'.
            # So for any w in Unvisited \ {nxt}, it must have a neighbor in (Unvisited \ {nxt}) U {nxt, 0}.
            
            # To make this fast, we can just check if the number of unvisited vertices 
            # with degree 0 in the subgraph induced by (Unvisited U {nxt, 0}) is > 1.
            # Or simply: for each w in Unvisited \ {nxt}, check if it has any neighbor 
            # in (Unvisited \ {nxt}) or {nxt} or {0}.
            
            # Optimization: Only do this check if the number of unvisited vertices is small 
            # or if we are close to the end? Or always? 
            # For N=24, iterating over unvisited vertices (max 23) and their neighbors is cheap.
            
            is_dead_end = False
            unvisited_count = 0
            for w in range(n):
                if not used[w] and w != nxt:
                    unvisited_count += 1
                    # w must connect to someone in (Unvisited \ {nxt}) or {nxt} or {0}
                    has_connection = False
                    # Check neighbor nxt
                    if nxt in adj_sets[w]:
                        has_connection = True
                    # Check neighbor 0
                    elif 0 in adj_sets[w]:
                        has_connection = True
                    # Check other unvisited neighbors
                    else:
                        for nb in adj[w]:
                            if not used[nb] and nb != nxt:
                                has_connection = True
                                break
                    if not has_connection:
                        # w is isolated from the rest of the potential path
                        # It can only be saved if it's the LAST vertex?
                        # If it's the last vertex, it needs to connect to 'nxt' (current end) and '0' (start).
                        # We already checked if it connects to 'nxt' or '0' or other unvisited.
                        # If it has NO connection to any of these, it's definitely dead.
                        is_dead_end = True
                        break
            
            if is_dead_end:
                continue

            used[nxt] = True
            path.append(nxt)
            
            result = backtrack()
            if result is not None:
                return result
            
            path.pop()
            used[nxt] = False
        
        return None

    return backtrack()

def construct():
    start_time = time.time()
    # 90s limit, leave 10s margin
    deadline = start_time + 80
    
    n = N
    total_edges = n * (n - 1) // 2
    
    # Binary search for the minimum threshold T
    lo, hi = 24, total_edges
    best_cycle = None
    
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        
        # Build subgraph with edges at position <= mid
        adj = [[] for _ in range(n)]
        for (u, v), pos in edge_pos.items():
            if pos <= mid:
                adj[u].append(v)
                adj[v].append(u)
        
        # Check for Hamilton cycle
        # Use a tight deadline for this specific check to avoid wasting time
        # If we spend too long on one mid, we might not be able to try lower mids.
        # But if we fail to find a cycle, we need to know it's a "No".
        # If we timeout, we don't know. 
        # Strategy: If we timeout, assume "No" (conservative for finding *a* cycle? 
        # No, if we assume No, we increase lo. This might skip a valid low mid.
        # If we assume Yes, we decrease hi. This might lead us to invalid lows.
        # However, best_cycle is only updated on success.
        # If we assume Yes on timeout, we probe lower. If they are truly empty, we'll get fast "No"s.
        # If they are hard, we might timeout again.
        # Given N=24, backtracking should be fast. Let's give it a reasonable chunk of time.
        
        cycle = find_hamilton_cycle(adj, n, deadline)
        
        if cycle is not None:
            best_cycle = cycle
            hi = mid - 1
        else:
            # If we timed out, cycle is None. We treat it as "No" to be safe?
            # Or "Yes"? 
            # If we treat as "No", we move lo up. We might miss the minimum.
            # If we treat as "Yes", we move hi down. We might end up with hi < true_min.
            # But we still have best_cycle from a higher mid.
            # Let's assume "No" (cycle not found) to avoid chasing ghosts.
            lo = mid + 1
    
    if best_cycle is not None:
        return best_cycle
    
    # Fallback: if binary search failed due to time or no cycle found (shouldn't happen for high T)
    # Try a greedy/randomized approach on the full graph or high threshold
    # For safety, just return a valid cycle from a high threshold if we have time
    if time.time() < deadline:
        # Try T = 276 (all edges)
        adj = [[] for _ in range(n)]
        for (u, v), pos in edge_pos.items():
            adj[u].append(v)
            adj[v].append(u)
        
        # Sort neighbors by position to find a "good" cycle if possible, 
        # but we just need any valid cycle.
        # The binary search should have found one. 
        # If not, maybe the pruning was too aggressive?
        # Let's try a simpler DFS without the "dead end" heuristic which might be flawed?
        # Or just return identity? No, identity might not be a cycle in the sense of edge existence? 
        # Identity is a cycle in the vertex sequence, but edges (0,1), (1,2)... must exist.
        # In a complete graph, they do.
        
        # Let's just run the DFS on the full graph with a tighter deadline
        cycle = find_hamilton_cycle(adj, n, start_time + 5)
        if cycle:
            return cycle
            
    # Final fallback
    return list(range(n))

# Test locally
if __name__ == "__main__":
    c = construct()
    max_pos = 0
    for i in range(N):
        u = c[i]
        v = c[(i+1)%N]
        a, b = min(u, v), max(u, v)
        p = edge_pos[(a, b)]
        if p > max_pos:
            max_pos = p
    print(f"Cycle: {c}")
    print(f"Score: {max_pos}")
