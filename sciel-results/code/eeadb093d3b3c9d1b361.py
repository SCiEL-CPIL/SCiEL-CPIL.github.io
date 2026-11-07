"""Rewritten program for packing edge-disjoint Hamiltonian cycles on 21-vertex host."""
# EVOLVE-BLOCK-START
import time
import random
import sys

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _edges_from_cycle(cycle):
    edges = set()
    n = len(cycle)
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        if u > v:
            u, v = v, u
        edges.add((u, v))
    return edges

def _remove_edges(adj, edges_to_remove):
    for u, v in edges_to_remove:
        adj[u].discard(v)
        adj[v].discard(u)
    return adj

def _find_ham_cycle_iterative(adj, n, deadline, start_override=None):
    """
    Iterative DFS with Warnsdorff heuristic to find a Hamiltonian cycle.
    Uses an explicit stack to avoid recursion limits and allow time checks.
    """
    if start_override is not None:
        start = start_override
    else:
        start = random.randint(0, n - 1)
    
    visited = [False] * n
    visited[start] = True
    path = [start]
    
    # Stack stores (current_vertex, next_neighbor_index, sorted_neighbors)
    # We'll use a stack of states for backtracking
    # Each state: (vertex, neighbors_list, next_index)
    stack = [(start, sorted(adj[start], key=lambda v: sum(1 for w in adj[v] if not visited[w])), 0)]
    
    # Actually, let's use a recursive approach but with iterative simulation
    # Better: use a proper iterative DFS
    
    # Let me rewrite this properly with iterative backtracking
    visited = [False] * n
    visited[start] = True
    path = [start]
    
    # For each vertex in path, we need to track which neighbors we've tried
    # neighbor_iter[i] = iterator over sorted neighbors of path[i]
    
    def get_sorted_neighbors(v):
        """Get unvisited neighbors sorted by Warnsdorff heuristic."""
        nbrs = [w for w in adj[v] if not visited[w]]
        if not nbrs:
            return []
        # Sort by number of unvisited neighbors (Warnsdorff)
        nbrs.sort(key=lambda w: sum(1 for x in adj[w] if not visited[x]))
        return nbrs
    
    # Iterative DFS
    # path[i] is the i-th vertex
    # For each i, we track which neighbor index we're at
    next_idx = [0] * n  # next_idx[i] = index into get_sorted_neighbors(path[i])
    nbr_lists = [None] * n  # nbr_lists[i] = sorted list of unvisited neighbors of path[i]
    
    # Initialize for the start vertex
    nbr_lists[0] = get_sorted_neighbors(start)
    
    depth = 0  # current depth in path (0-indexed, path[depth] is current)
    
    while depth >= 0:
        if time.time() > deadline:
            return None
        
        if depth == n - 1:
            # Check if we can close the cycle
            if start in adj[path[depth]]:
                return list(path)
            depth -= 1
            continue
        
        v = path[depth]
        
        if nbr_lists[depth] is None:
            nbr_lists[depth] = get_sorted_neighbors(v)
        
        if next_idx[depth] < len(nbr_lists[depth]):
            w = nbr_lists[depth][next_idx[depth]]
            next_idx[depth] += 1
            
            # Try extending to w
            visited[w] = True
            path.append(w)
            depth += 1
            nbr_lists[depth] = get_sorted_neighbors(w)
            next_idx[depth] = 0
        else:
            # No more neighbors to try, backtrack
            depth -= 1
            if depth >= 0:
                visited[path[depth + 1]] = False
                # Re-compute neighbors for the new current vertex since visited changed
                nbr_lists[depth] = get_sorted_neighbors(path[depth])
                next_idx[depth] = 0
    
    return None

def _find_ham_cycle_recursive(adj, n, deadline, start_override=None):
    """
    Recursive DFS with Warnsdorff heuristic to find a Hamiltonian cycle.
    """
    if start_override is not None:
        start = start_override
    else:
        start = random.randint(0, n - 1)
    
    visited = [False] * n
    visited[start] = True
    path = [start]
    
    sys.setrecursionlimit(10000)
    
    def dfs(current):
        if time.time() > deadline:
            return False
        
        if len(path) == n:
            if start in adj[current]:
                return True
            return False
        
        # Get unvisited neighbors
        neighbors = [v for v in adj[current] if not visited[v]]
        if not neighbors:
            return False
        
        # Warnsdorff heuristic: prefer neighbors with fewer unvisited options
        def heuristic(v):
            unvisited_count = sum(1 for w in adj[v] if not visited[w])
            return unvisited_count
        
        neighbors.sort(key=heuristic)
        
        for v in neighbors:
            visited[v] = True
            path.append(v)
            if dfs(v):
                return True
            path.pop()
            visited[v] = False
        
        return False
    
    if dfs(start):
        return path
    return None

def _is_connected(adj, n):
    """Check if the graph is connected using BFS."""
    if n == 0:
        return True
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

def _check_min_degree(adj, n):
    """Check if all vertices have degree >= 2 (necessary for Hamiltonian cycle)."""
    for i in range(n):
        if len(adj[i]) < 2:
            return False
    return True

def construct():
    n = P['n']
    edges = P['edges']
    
    # Deadline guard: 85 seconds to be safe within 90s limit
    deadline = time.time() + 85.0
    
    best_cycles = []
    
    # Try multiple random restarts to find a good set of edge-disjoint Hamiltonian cycles
    max_restarts = 100
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break
        
        # Build fresh adjacency from original edges
        adj = _build_adj(n, edges)
        cycles = []
        success = True
        
        for cycle_idx in range(10):
            if time.time() > deadline - 5:
                success = False
                break
            
            # Check connectivity and minimum degree before attempting to find a cycle
            if not _is_connected(adj, n) or not _check_min_degree(adj, n):
                success = False
                break
            
            # Try to find a Hamiltonian cycle with different starting vertices
            cycle = None
            found = False
            
            # Try several starting vertices
            start_vertices = list(range(n))
            random.shuffle(start_vertices)
            
            for start in start_vertices:
                if time.time() > deadline - 5:
                    found = False
                    break
                
                # Use recursive version (simpler and often faster for n=21)
                candidate = _find_ham_cycle_recursive(adj, n, deadline, start_override=start)
                if candidate is not None:
                    cycle = candidate
                    found = True
                    break
            
            if not found or cycle is None:
                success = False
                break
            
            cycles.append(list(cycle))
            
            # Remove the edges used in this cycle
            edge_set = _edges_from_cycle(cycle)
            _remove_edges(adj, edge_set)
        
        if len(cycles) > len(best_cycles):
            best_cycles = [list(c) for c in cycles]
        
        # If we found a lot of cycles, we can stop early
        if len(best_cycles) >= 7:
            break
    
    # Convert to proper format: list of lists of Python ints
    result = []
    for cycle in best_cycles:
        # Ensure all elements are Python ints
        clean_cycle = [int(v) for v in cycle]
        result.append(clean_cycle)
    
    # Validate the result
    if not result:
        # Fallback: return a single known Hamiltonian cycle if we found one
        # Use a simple greedy approach
        adj = _build_adj(n, edges)
        cycle = _find_ham_cycle_recursive(adj, n, deadline, start_override=0)
        if cycle is not None:
            result = [[int(v) for v in cycle]]
    
    # Final validation: check that all cycles use the same vertex set
    if result:
        vertex_set = set(result[0])
        for cycle in result:
            if set(cycle) != vertex_set:
                result = [[int(v) for v in result[0]]]  # Keep only first valid cycle
                break
    
    # Check edge-disjointness
    if len(result) > 1:
        used_edges = set()
        valid = []
        for cycle in result:
            edge_set = _edges_from_cycle(cycle)
            if edge_set & used_edges:
                break  # Not edge-disjoint
            used_edges |= edge_set
            valid.append(cycle)
        result = valid
    
    return result

# EVOLVE-BLOCK-END
