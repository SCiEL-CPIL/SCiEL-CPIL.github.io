"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import sys

N = 24

# Generate the pinned edge list with random.Random(614)
def _build_edge_order():
    rng = random.Random(614)
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((i, j))
    rng.shuffle(edges)
    return edges

EDGE_ORDER = _build_edge_order()
EDGE_INDEX = {}
for idx, (u, v) in enumerate(EDGE_ORDER, 1):
    EDGE_INDEX[(u, v)] = idx
    EDGE_INDEX[(v, u)] = idx

def _edge_score(u, v):
    return EDGE_INDEX[(u, v)]

def construct():
    # We want to find a Hamiltonian cycle that minimizes the maximum edge arrival index.
    # This is the Minimum Bottleneck Hamiltonian Cycle problem.
    
    # Strategy:
    # 1. Binary search on the threshold T (max edge index allowed).
    # 2. For a given T, check if the subgraph G_T contains a Hamiltonian cycle.
    # 3. Use DFS with strong pruning (connectivity and reachability) for the check.
    
    sys.setrecursionlimit(10000)
    
    def check_hamiltonian(T):
        # Build adjacency list for edges with index <= T
        adj = [[] for _ in range(N)]
        adj_set = [set() for _ in range(N)] # For faster lookup
        for u in range(N):
            for v in range(u + 1, N):
                if _edge_score(u, v) <= T:
                    adj[u].append(v)
                    adj[v].append(u)
                    adj_set[u].add(v)
                    adj_set[v].add(u)
        
        # Necessary condition: min degree >= 2
        for u in range(N):
            if len(adj[u]) < 2:
                return None
        
        # Necessary condition: Graph must be connected
        # (Implicitly checked by DFS, but early check helps)
        
        # DFS with pruning
        visited = [False] * N
        path = [0]
        visited[0] = True
        unvisited_count = N - 1
        
        # Helper: Check if unvisited nodes are connected to each other
        # and if current node can reach 0 via unvisited nodes?
        # Actually, simpler pruning:
        # 1. If unvisited nodes are not connected in G_T, return False.
        # 2. If current node cannot reach 0 through unvisited nodes (or directly if last step), return False.
        
        def is_connected_unvisited():
            # Check if all unvisited nodes are in the same connected component
            # We only need to check if they are connected to each other.
            # Start from first unvisited node, BFS/DFS, see if we reach all others.
            unvisited = [i for i in range(N) if not visited[i]]
            if not unvisited:
                return True
            start = unvisited[0]
            stack = [start]
            seen = {start}
            while stack:
                node = stack.pop()
                for nei in adj[node]:
                    if not visited[nei] and nei not in seen:
                        seen.add(nei)
                        stack.append(nei)
            return len(seen) == len(unvisited)
        
        def can_reach_start(current):
            # Can we reach 0 from current using only unvisited nodes?
            # If current is the last node (len(path)==N), we just check edge (current, 0).
            # Otherwise, we need a path current -> ... -> 0 in G_T \ visited.
            # This is expensive. A cheaper heuristic: 
            # Check if current is connected to 0 in the graph induced by {current} U unvisited.
            
            if len(path) == N:
                return 0 in adj_set[current]
            
            # BFS from current, only visiting unvisited nodes, target 0.
            # Note: 0 is visited, so we stop when we hit a neighbor of 0? 
            # No, the path must go through unvisited nodes to reach 0.
            # So we look for a path from current to any neighbor of 0 that is unvisited?
            # Or simply, is there a path from current to 0 in G[unvisited U {current}]?
            # Since 0 is not in unvisited, we check if current can reach any neighbor of 0 that is unvisited.
            
            # Actually, simpler: 
            # If we are at step k < N-1, we must be able to reach 0 via unvisited nodes.
            # This means current must be connected to the set of unvisited nodes that are connected to 0.
            
            # Let's just do a simple BFS from current, restricted to unvisited nodes, 
            # and see if we can reach any node that is adjacent to 0.
            
            if not unvisited_count:
                return 0 in adj_set[current]
            
            # If current has no unvisited neighbors, we are stuck (unless we are done)
            has_unvisited_neighbor = False
            for nei in adj[current]:
                if not visited[nei]:
                    has_unvisited_neighbor = True
                    break
            if not has_unvisited_neighbor:
                return False
            
            # BFS
            stack = [current]
            seen = {current}
            while stack:
                node = stack.pop()
                for nei in adj[node]:
                    if nei == 0:
                        return True
                    if not visited[nei] and nei not in seen:
                        seen.add(nei)
                        stack.append(nei)
            return False

        def dfs():
            nonlocal unvisited_count
            
            if len(path) == N:
                if 0 in adj_set[path[-1]]:
                    return True
                return False
            
            last = path[-1]
            
            # Pruning 1: Unvisited nodes must be connected
            # Only check if unvisited_count > 1 to save time
            if unvisited_count > 1:
                if not is_connected_unvisited():
                    return False
            
            # Pruning 2: Must be able to reach start
            if not can_reach_start(last):
                return False
            
            for v in adj[last]:
                if not visited[v]:
                    if v == 0 and len(path) < N:
                        continue
                    
                    visited[v] = True
                    path.append(v)
                    unvisited_count -= 1
                    
                    if dfs():
                        return True
                    
                    path.pop()
                    visited[v] = False
                    unvisited_count += 1
            
            return False
        
        if dfs():
            return path[:]
        return None

    # Binary Search
    # Lower bound: N (need N edges, min index 1, but max index must be at least something)
    # Actually, the smallest possible max index is at least the N-th smallest edge? No.
    # We just start from a reasonable low value.
    # Max possible is total edges.
    
    # Initial upper bound: Greedy cycle
    def greedy_cycle():
        visited = [False] * N
        cycle = [0]
        visited[0] = True
        
        for _ in range(N - 1):
            last = cycle[-1]
            best_next = -1
            best_idx = float('inf')
            for v in range(N):
                if not visited[v]:
                    idx = _edge_score(last, v)
                    if idx < best_idx:
                        best_idx = idx
                        best_next = v
            if best_next == -1:
                return None
            cycle.append(best_next)
            visited[best_next] = True
        
        if 0 in adj_set if False else True: # Placeholder
             pass
        
        # Check close
        # We need to check if edge (last, 0) exists. It always does in complete graph.
        # But we need to check its index.
        max_idx = max(_edge_score(cycle[i], cycle[i+1]) for i in range(N-1))
        max_idx = max(max_idx, _edge_score(cycle[-1], cycle[0]))
        return (cycle, max_idx)

    result = greedy_cycle()
    if result:
        best_cycle, hi = result
    else:
        # Fallback
        best_cycle = list(range(N))
        hi = max(_edge_score(best_cycle[i], best_cycle[i+1]) for i in range(N-1))
        hi = max(hi, _edge_score(best_cycle[-1], best_cycle[0]))
        
    lo = 1
    
    # We can tighten lo. 
    # The N edges in the cycle must have distinct indices? No, but they are distinct edges.
    # The max index must be at least the N-th smallest edge in the whole graph? No.
    # But it must be at least the max of the N edges in the cycle.
    
    while lo < hi:
        mid = (lo + hi) // 2
        cycle = check_hamiltonian(mid)
        if cycle is not None:
            hi = mid
            best_cycle = cycle
        else:
            lo = mid + 1
            
    return best_cycle

# EVOLVE-BLOCK-END
