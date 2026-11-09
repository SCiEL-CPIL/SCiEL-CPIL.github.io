"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random
sys.setrecursionlimit(10000)

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj(edge_set, n):
    """Build adjacency list from a set of frozenset edges."""
    adj = [set() for _ in range(n)]
    for e in edge_set:
        u, v = tuple(e)
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_ham_cycle_fast(adj, n, deadline, seed_offset=0):
    """Find a Hamiltonian cycle using iterative deepening DFS with strong pruning."""
    start = 0
    path = [start]
    visited = [False] * n
    visited[start] = True
    
    # Precompute neighbor lists sorted by degree (most constrained first)
    sorted_neighbors = [sorted(adj[i], key=lambda x: len(adj[x])) for i in range(n)]
    
    def _check_connectivity(unvisited, current_end, target):
        """Check if target is reachable from current_end through unvisited vertices."""
        if current_end == target:
            return True
        if not unvisited:
            return False
        # BFS from current_end, can only use unvisited vertices
        visited_bfs = set()
        queue = [current_end]
        visited_bfs.add(current_end)
        while queue:
            node = queue.pop(0)
            for nb in adj[node]:
                if nb == target:
                    return True
                if nb not in visited_bfs and nb in unvisited:
                    visited_bfs.add(nb)
                    queue.append(nb)
        return False
    
    def _count_components(unvisited, adj_sub):
        """Count connected components in the subgraph induced by unvisited vertices."""
        if not unvisited:
            return 0
        unvisited_set = set(unvisited)
        visited_comp = set()
        count = 0
        for v in unvisited:
            if v not in visited_comp:
                count += 1
                # BFS
                queue = [v]
                visited_comp.add(v)
                while queue:
                    node = queue.pop(0)
                    for nb in adj_sub[node]:
                        if nb in unvisited_set and nb not in visited_comp:
                            visited_comp.add(nb)
                            queue.append(nb)
        return count
    
    def backtrack():
        if time.time() > deadline:
            return None
        if len(path) == n:
            if path[-1] in adj[0]:
                return list(path)
            return None
        cur = path[-1]
        # Try neighbors in order of fewest options first
        for nb in sorted_neighbors[cur]:
            if not visited[nb]:
                # Pruning 1: if nb is the penultimate, check if last vertex connects to both
                if len(path) == n - 2:
                    remaining = [v for v in range(n) if not visited[v] and v != nb]
                    if len(remaining) == 1:
                        r = remaining[0]
                        if r not in adj[nb] or r not in adj[0]:
                            continue
                # Pruning 2: connectivity check - after placing nb, can we still reach 0?
                elif len(path) >= 3 and len(path) <= n - 3:
                    unvisited_after = set(range(n)) - set(path) - {nb}
                    if unvisited_after and not _check_connectivity(nb, 0):
                        continue
                
                visited[nb] = True
                path.append(nb)
                result = backtrack()
                if result is not None:
                    return result
                path.pop()
                visited[nb] = False
        return None
    
    return backtrack()

def _find_cycle_with_restarts(adj, n, deadline, max_restarts=50):
    """Try multiple random orderings to find a Hamiltonian cycle."""
    # First try deterministic
    cycle = _find_ham_cycle_fast(adj, n, deadline)
    if cycle is not None:
        return cycle
    
    # Try with randomized neighbor ordering
    rng = random.Random(42)
    for attempt in range(max_restarts):
        if time.time() > deadline:
            break
        # Shuffle neighbor orderings
        shuffled_adj = [list(s) for s in adj]
        for i in range(n):
            rng.shuffle(shuffled_adj[i])
        
        # Use a simple DFS with the shuffled ordering
        path = [0]
        visited = [False] * n
        visited[0] = True
        
        def dfs_rand():
            if time.time() > deadline:
                return None
            if len(path) == n:
                if path[-1] in adj[0]:
                    return list(path)
                return None
            cur = path[-1]
            for nb in shuffled_adj[cur]:
                if not visited[nb]:
                    if len(path) == n - 2:
                        remaining = [v for v in range(n) if not visited[v] and v != nb]
                        if len(remaining) == 1:
                            r = remaining[0]
                            if r not in adj[nb] or r not in adj[0]:
                                continue
                    visited[nb] = True
                    path.append(nb)
                    result = dfs_rand()
                    if result is not None:
                        return result
                    path.pop()
                    visited[nb] = False
            return None
        
        cycle = dfs_rand()
        if cycle is not None:
            return cycle
        path = [0]
        visited = [False] * n
        visited[0] = True
    
    return None

def construct():
    deadline = time.time() + 85  # 85 second safety margin (90s kill - 5s buffer)
    
    n = P['n']
    all_edges = set()
    for u, v in P['edges']:
        all_edges.add(frozenset((u, v)))
    
    remaining_edges = set(all_edges)
    cycles = []
    
    max_cycles = 3  # floor(70/21) = 3
    
    for cycle_num in range(max_cycles):
        if time.time() > deadline:
            break
        
        # Build adjacency from remaining edges
        adj = _build_adj(remaining_edges, n)
        
        # Check if graph is still 2-regular enough (each vertex needs degree >= 2)
        valid = True
        for i in range(n):
            if len(adj[i]) < 2:
                valid = False
                break
        if not valid:
            break
        
        # Find a Hamiltonian cycle
        cycle = _find_cycle_with_restarts(adj, n, deadline)
        
        if cycle is None:
            break
        
        # Verify cycle is valid
        valid_cycle = True
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            if frozenset((u, v)) not in remaining_edges:
                valid_cycle = False
                break
        if not valid_cycle:
            break
        
        # Remove edges used by this cycle
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            remaining_edges.discard(frozenset((u, v)))
        
        cycles.append([int(v) for v in cycle])
    
    # Ensure non-empty and all types are correct
    if not cycles:
        # Fallback: find at least one cycle in the full graph
        adj_full = _build_adj(all_edges, n)
        cycle = _find_cycle_with_restarts(adj_full, n, deadline)
        if cycle is not None:
            return [[int(v) for v in cycle]]
        # Last resort: return a trivial valid structure (should not happen)
        return [[0, 1, 2]]
    
    # Ensure all vertices are Python int
    for i in range(len(cycles)):
        cycles[i] = [int(v) for v in cycles[i]]
    
    return cycles
# EVOLVE-BLOCK-END
