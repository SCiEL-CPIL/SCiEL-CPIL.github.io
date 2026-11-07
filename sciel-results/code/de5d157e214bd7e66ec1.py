"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj(n, edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _is_connected(vertices, adj, used_edges):
    """Check if the graph induced by vertices with edges not in used_edges is connected."""
    if not vertices:
        return True
    
    vset = set(vertices)
    start = next(iter(vset))
    visited = {start}
    stack = [start]
    
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v in vset and v not in visited:
                e = (min(u, v), max(u, v))
                if e not in used_edges:
                    visited.add(v)
                    stack.append(v)
    
    return len(visited) == len(vset)

def _two_opt_improve(cycle, adj, used_edges, deadline):
    """Try to improve a Hamiltonian cycle via 2-opt moves (reversing segments).
    A 2-opt move reverses a segment of the cycle, which changes exactly 2 edges.
    We accept a move if both new edges are available and it reduces some cost metric
    (e.g., makes edge usage more uniform). Here we just try to find a valid
    re-routing that keeps the cycle valid."""
    n = len(cycle)
    if n < 4:
        return cycle
    
    # Build a set for O(1) edge lookup
    edge_set = set()
    for u in adj:
        for v in adj[u]:
            edge_set.add((min(u, v), max(u, v)))
    
    def _edge_avail(u, v):
        e = (min(u, v), max(u, v))
        return e in edge_set and e not in used_edges
    
    # Try 2-opt: reverse segment [i..j] in the cycle
    # Original edges: (cycle[i-1], cycle[i]) and (cycle[j], cycle[(j+1)%n])
    # New edges:     (cycle[i-1], cycle[j]) and (cycle[i], cycle[(j+1)%n])
    best_cycle = list(cycle)
    best_score = 0
    
    # Compute a simple score: sum of degrees of used edges (lower = more "spread")
    def _cycle_score(cyc):
        s = 0
        for idx in range(len(cyc)):
            u = cyc[idx]
            v = cyc[(idx + 1) % len(cyc)]
            e = (min(u, v), max(u, v))
            # Count how many other edges share an endpoint with e
            s += len(adj[u]) + len(adj[v])
        return s
    
    current_score = _cycle_score(best_cycle)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(1, n - 1):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if i == 0 or j == n - 1:
                    # Avoid wrapping issues at boundaries for simplicity
                    pass
                # Check if reversing segment [i..j] gives valid new edges
                u_prev = best_cycle[(i - 1) % n]
                u_i = best_cycle[i]
                u_j = best_cycle[j]
                u_next = best_cycle[(j + 1) % n]
                
                # Skip if the new edges are the same as old (no change)
                if (min(u_prev, u_j), max(u_prev, u_j)) == (min(u_prev, u_i), max(u_prev, u_i)):
                    continue
                if (min(u_i, u_next), max(u_i, u_next)) == (min(u_j, u_next), max(u_j, u_next)):
                    continue
                
                if not _edge_avail(u_prev, u_j):
                    continue
                if not _edge_avail(u_i, u_next):
                    continue
                
                # Check the new edges aren't already in the cycle (would create duplicate)
                new_e1 = (min(u_prev, u_j), max(u_prev, u_j))
                new_e2 = (min(u_i, u_next), max(u_i, u_next))
                
                # Check they're not the same edge
                if new_e1 == new_e2:
                    continue
                
                # Check they're not already used in the cycle
                cycle_edges = set()
                for idx in range(n):
                    a = best_cycle[idx]
                    b = best_cycle[(idx + 1) % n]
                    cycle_edges.add((min(a, b), max(a, b)))
                
                if new_e1 in cycle_edges and new_e1 != (min(u_prev, u_i), max(u_prev, u_i)):
                    continue
                if new_e2 in cycle_edges and new_e2 != (min(u_j, u_next), max(u_j, u_next)):
                    continue
                
                # Apply the 2-opt move
                new_cycle = best_cycle[:i] + best_cycle[i:j+1][::-1] + best_cycle[j+1:]
                
                new_score = _cycle_score(new_cycle)
                if new_score < current_score:
                    best_cycle = new_cycle
                    current_score = new_score
                    improved = True
                    break
            if improved:
                break
    
    return best_cycle


def _find_ham_cycle(vertices, adj, used_edges, deadline, rng, max_depth=5000):
    """Find a Hamiltonian cycle on the given vertex set using DFS with backtracking."""
    vertices = list(vertices)
    n = len(vertices)
    if n < 3:
        return None
    
    vset = set(vertices)
    start = vertices[0]
    
    # Precompute adjacency as frozensets for faster lookup
    adj_frozen = {u: frozenset(v) for u, v in adj.items()}
    
    # Use the precomputed global edge set
    all_graph_edges = _ALL_EDGES
    
    # Count available edges (not in used_edges)
    available_edges = all_graph_edges - used_edges
    
    # Check if enough edges remain for a Hamiltonian cycle
    if len(available_edges) < n:
        return None
    
    def _connected_remaining(visited_set, current):
        """Check if unvisited vertices + current form a connected subgraph with available edges."""
        unvisited = vset - visited_set
        if not unvisited:
            return True
        # BFS from current through available edges to see if all unvisited are reachable
        # (plus we need to be able to return to start at the end)
        visited_bfs = set()
        stack = [current]
        visited_bfs.add(current)
        while stack:
            u = stack.pop()
            for v in adj_frozen[u]:
                if v in vset and v not in visited_bfs:
                    e = (min(u, v), max(u, v))
                    if e in available_edges:
                        visited_bfs.add(v)
                        stack.append(v)
        # All unvisited must be reachable
        return unvisited.issubset(visited_bfs)
    
    def dfs(current, visited, path):
        if time.time() > deadline:
            return None
            
        if len(visited) == n:
            # Check if we can close the cycle
            if start in adj_frozen[current]:
                e = (min(current, start), max(current, start))
                if e in available_edges:
                    return path
            return None
        
        # Get unvisited neighbors with available edges
        unvisited_neighbors = []
        for nb in adj_frozen[current]:
            if nb in vset and nb not in visited:
                e = (min(current, nb), max(current, nb))
                if e in available_edges:
                    unvisited_neighbors.append(nb)
        
        if not unvisited_neighbors:
            return None
        
        # Sort by degree in remaining graph (Warnsdorff's heuristic)
        def heuristic_key(x):
            deg = 0
            for y in adj_frozen[x]:
                if y in vset and y not in visited:
                    e2 = (min(x, y), max(x, y))
                    if e2 in available_edges:
                        deg += 1
            return (deg, rng.random())
            
        unvisited_neighbors.sort(key=heuristic_key)
        
        for nb in unvisited_neighbors:
            visited.add(nb)
            path.append(nb)
            e = (min(current, nb), max(current, nb))
            available_edges.discard(e)
            
            result = dfs(nb, visited, path)
            if result is not None:
                return result
            
            # Backtrack
            path.pop()
            visited.discard(nb)
            available_edges.add(e)
        
        return None
    
    visited = {start}
    path = [start]
    return dfs(start, visited, path)

# Precompute all graph edges once
_ALL_EDGES = set()
for _u, _v in P['edges']:
    _ALL_EDGES.add((min(_u, _v), max(_u, _v)))


def construct():
    start_time = time.time()
    total_deadline = start_time + 85  # 85 seconds to be safe (90s limit, leave margin)
    n = P['n']
    edges = P['edges']
    adj = _build_adj(n, edges)
    
    all_vertices = list(range(n))
    max_possible_cycles = 3 # Theoretical max for 21 vertices, 70 edges
    
    best_cycles = []
    best_cycle_count = 0
    
    # Try multiple random restarts to ensure we find the maximum set
    for attempt in range(30):
        if time.time() > total_deadline:
            break
        
        # Check if we already found the max
        if best_cycle_count == max_possible_cycles:
            break
        
        used_edges = set()
        cycles = []
        
        for i in range(max_possible_cycles):
            if time.time() > total_deadline:
                break
                
            # Check connectivity before expensive search
            if not _is_connected(all_vertices, adj, used_edges):
                break
                
            # Use a unique RNG for each attempt to vary the search
            rng = random.Random(attempt * 7919 + i * 104729)
            
            cycle_deadline = total_deadline
            
            cycle = _find_ham_cycle(all_vertices, adj, used_edges, cycle_deadline, rng)
            
            if cycle is None:
                break
            
            # Apply 2-opt improvement to the cycle
            if time.time() < total_deadline - 2:
                cycle = _two_opt_improve(cycle, adj, used_edges, total_deadline - 1)
            
            # Add edges of this cycle to used_edges
            for j in range(len(cycle)):
                u = cycle[j]
                v = cycle[(j + 1) % len(cycle)]
                e = (min(u, v), max(u, v))
                used_edges.add(e)
            
            cycles.append(cycle)
        
        # Update best if we found more
        if len(cycles) > best_cycle_count:
            best_cycles = cycles
            best_cycle_count = len(cycles)
            if best_cycle_count == max_possible_cycles:
                break  # Found the theoretical max, stop searching

    # Convert to proper types
    result = [list(c) for c in best_cycles]
    
    if not result:
        result = [list(range(n))]

    return result
# EVOLVE-BLOCK-END
