"""Rewritten program: find edge-disjoint Hamiltonian cycles on the 21-vertex host graph."""
import time
from collections import defaultdict

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adjacency():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _find_hamiltonian_cycle(adj, used_edges, start=0):
    """
    Find a Hamiltonian cycle starting and ending at 'start' using backtracking
    with MRV (minimum remaining values) and degree-based pruning.
    Returns a list of vertices [v0, v1, ..., v_{n-1}] where v0=start, or None.
    """
    n = P['n']
    # Precompute adjacency as sorted lists for deterministic iteration
    adj_list = [sorted(adj[i]) for i in range(n)]
    
    # Track which edges are available (not in used_edges)
    # We'll maintain a set of available edges dynamically via used_edges
    
    # State: path, set of visited vertices
    # We use iterative deepening? No, just DFS with pruning.
    
    # To make it faster, we can precompute for each vertex the neighbors
    # and filter by used_edges on the fly.
    
    # Let's use a recursive DFS with memoization? No, state space is too large.
    # Instead, use simple backtracking with strong pruning.
    
    # Pruning heuristic:
    # 1. If any unvisited vertex has no available edges to any other unvisited vertex 
    #    (and it's not the last one to connect back to start), prune.
    # 2. Degree check: for each unvisited vertex, count available edges to unvisited vertices.
    #    If count < 2, prune (unless it's the last vertex, which needs 1 edge to previous and 1 to start, 
    #    but previous is visited, so it needs 1 edge to start and 1 edge to the next unvisited? No, 
    #    in a Hamiltonian path from start, the last vertex must connect to start. 
    #    Actually, for the path, each intermediate unvisited vertex must have at least 2 available edges 
    #    to other unvisited vertices (or one to visited and one to unvisited, etc). 
    #    This is complex. Let's stick to a simpler but effective heuristic:
    #    Check if any unvisited vertex has 0 available edges to unvisited vertices AND 
    #    is not the only unvisited vertex left. If so, prune.
    #    Also, if the current vertex's next step leads to a dead end, backtrack.
    
    # Better approach: Use the "forced move" or "biconnected component" check?
    # For 21 vertices, simple backtracking with MRV might be enough if we order well.
    
    # Let's implement a DFS that picks the next vertex with the fewest available unvisited neighbors (MRV).
    
    def dfs(path, visited, current):
        if len(path) == n:
            # Check if current connects back to start
            if _edge_key(current, start) not in used_edges and start in adj[current]:
                return path[:]
            return None
        
        # Get available unvisited neighbors of current
        neighbors = []
        for nb in adj_list[current]:
            if nb not in visited:
                if _edge_key(current, nb) not in used_edges:
                    neighbors.append(nb)
        
        if not neighbors:
            return None
        
        # MRV: sort neighbors by number of available unvisited neighbors (fewest first)
        def count_available_unvisited(v):
            count = 0
            for nb in adj_list[v]:
                if nb not in visited and nb != v:
                    if _edge_key(v, nb) not in used_edges:
                        count += 1
            return count
        
        neighbors.sort(key=lambda v: count_available_unvisited(v))
        
        for nb in neighbors:
            # Pruning: Check if any other unvisited vertex becomes isolated
            # (has no available edges to other unvisited vertices)
            # This is O(n^2) per step, which might be slow. 
            # Let's try without this first, and add if needed.
            
            new_visited = visited | {nb}
            result = dfs(path + [nb], new_visited, nb)
            if result is not None:
                return result
        
        return None
    
    # Start the DFS
    path = [start]
    visited = {start}
    result = dfs(path, visited, start)
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Leave 10s margin for safety
    
    adj = _build_adjacency()
    used_edges = set()
    cycles = []
    
    # We want to find up to 3 edge-disjoint Hamiltonian cycles
    # Max possible is floor(72/21) = 3
    
    for _ in range(3):
        if time.time() > deadline:
            break
        
        # Try to find a Hamiltonian cycle avoiding used_edges
        # Try different starts to increase chance of success
        found = None
        for s in range(P['n']):
            if time.time() > deadline:
                break
            found = _find_hamiltonian_cycle(adj, used_edges, start=s)
            if found is not None:
                break
        
        if found is None:
            break
        
        # Add edges of this cycle to used_edges
        for i in range(len(found)):
            u = found[i]
            v = found[(i + 1) % len(found)]
            used_edges.add(_edge_key(u, v))
        
        cycles.append(found)
    
    # Return the list of cycles
    # Ensure types are correct: list of lists of ints
    return [list(cycle) for cycle in cycles]

# EVOLVE-BLOCK-END
