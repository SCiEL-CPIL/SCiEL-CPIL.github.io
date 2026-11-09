"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _edge_key(u, v):
    if u > v:
        u, v = v, u
    return (u, v)

def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _is_connected(adj, nodes, n):
    """Check if the subgraph induced by nodes is connected."""
    if not nodes:
        return True
    start = next(iter(nodes))
    visited = {start}
    stack = [start]
    while stack:
        cur = stack.pop()
        for nb in adj[cur]:
            if nb in nodes and nb not in visited:
                visited.add(nb)
                stack.append(nb)
    return len(visited) == len(nodes)

def _find_ham_cycle(adj, n, deadline):
    """Find a Hamiltonian cycle using backtracking with Warnsdorff's heuristic."""
    start = 0
    path = [start]
    visited = [False] * n
    visited[start] = True
    
    # Pre-compute neighbors lists for faster access
    neighbors = [list(adj[i]) for i in range(n)]
    
    def backtrack():
        if time.time() > deadline:
            return None
        
        if len(path) == n:
            if path[-1] in adj[0]:
                return list(path)
            return None
        
        cur = path[-1]
        unvisited_count = n - len(path)
        
        # Get unvisited neighbors
        unvisited_neighbors = [nb for nb in neighbors[cur] if not visited[nb]]
        
        if not unvisited_neighbors:
            return None
            
        # Warnsdorff's rule: sort by number of unvisited neighbors
        # We want to visit nodes with fewer options first to avoid isolating them
        def heuristic(nb):
            return len([n for n in neighbors[nb] if not visited[n]])
            
        unvisited_neighbors.sort(key=heuristic)
        
        for nb in unvisited_neighbors:
            # Pruning: Check if the remaining unvisited vertices are connected
            # We only do this if the remaining set is small enough to be fast, or always if n is small
            # For n=21, checking connectivity every step is O(N^2) which is ~400 ops, very fast.
            # However, we must include 'start' (0) in the connectivity check if we are not at the end,
            # because the cycle must close back to 0.
            # Actually, simpler: check if the set of *currently unvisited* vertices (excluding the one we are about to visit? No, including)
            # Let's check if the set of unvisited vertices (after visiting nb) is connected.
            # The set of vertices that will remain unvisited is: all unvisited except nb.
            # But we also need to be able to reach 0 from the end of the path.
            # A standard pruning: The subgraph induced by (Unvisited \ {nb}) U {0} must be connected?
            # Or simpler: The subgraph induced by Unvisited must be connected?
            # If we visit nb, the remaining path must cover Unvisited \ {nb} and end at a neighbor of 0.
            # So, (Unvisited \ {nb}) must be connected? Not necessarily, but if it's disconnected, we can't traverse it.
            # Let's check connectivity of (Unvisited \ {nb}).
            
            remaining_unvisited = [v for v in range(n) if not visited[v] and v != nb]
            
            # Optimization: Only do connectivity check if remaining_unvisited is not empty
            if remaining_unvisited:
                # Build a temporary adjacency view or just use BFS on the existing adj
                # We only care about connections within remaining_unvisited
                # Note: We don't need to connect to 'cur' or '0' yet, just among themselves
                # Actually, if remaining_unvisited is disconnected, we can't form a single path through them.
                
                # Quick check: pick a node, BFS
                start_node = remaining_unvisited[0]
                visited_conn = {start_node}
                stack = [start_node]
                while stack:
                    c = stack.pop()
                    for nb2 in adj[c]:
                        if nb2 in remaining_unvisited and nb2 not in visited_conn:
                            visited_conn.add(nb2)
                            stack.append(nb2)
                
                if len(visited_conn) != len(remaining_unvisited):
                    continue # Prune this branch

            visited[nb] = True
            path.append(nb)
            result = backtrack()
            if result is not None:
                return result
            path.pop()
            visited[nb] = False
            
        return None

    return backtrack()

def construct():
    deadline = time.time() + 50  # 50 second safety margin
    
    n = P['n']
    all_edges = set()
    for u, v in P['edges']:
        all_edges.add(_edge_key(u, v))
    
    remaining_edges = set(all_edges)
    cycles = []
    
    for _ in range(3):  # Max possible is floor(70/21)=3
        if time.time() > deadline:
            break
            
        # Build adjacency from remaining edges
        current_edges = [(u, v) for u, v in P['edges'] if _edge_key(u, v) in remaining_edges]
        adj = _build_adj(current_edges, n)
        
        # Check if graph is still 2-regular enough (each vertex needs degree >= 2)
        valid = True
        for i in range(n):
            if len(adj[i]) < 2:
                valid = False
                break
        if not valid:
            break
        
        cycle = _find_ham_cycle(adj, n, deadline)
        if cycle is None:
            break
        
        # Verify cycle is valid
        valid_cycle = True
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            if _edge_key(u, v) not in remaining_edges:
                valid_cycle = False
                break
        if not valid_cycle:
            break
        
        # Remove edges used by this cycle
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            remaining_edges.discard(_edge_key(u, v))
        
        cycles.append(cycle)
        
    # Ensure non-empty
    if not cycles:
        # Fallback: find any Hamiltonian cycle in the original graph
        adj = _build_adj(P['edges'], n)
        cycle = _find_ham_cycle(adj, n, deadline)
        if cycle:
            return [cycle]
        # Last resort: return empty or a trivial cycle if allowed? 
        # Task says "nonempty list". If we fail, we might return invalid.
        # But we expect to succeed.
        return []

    return cycles
# EVOLVE-BLOCK-END
