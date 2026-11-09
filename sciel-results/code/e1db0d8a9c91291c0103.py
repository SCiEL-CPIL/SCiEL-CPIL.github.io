"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _build_adjacency(edges, n):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_hamiltonian_cycle(adj, n, start, used_edges, deadline):
    """Find a Hamiltonian cycle starting from 'start' avoiding used_edges."""
    # Build available adjacency
    avail_adj = defaultdict(set)
    for u in range(n):
        for v in adj[u]:
            ek = _edge_key(u, v)
            if ek not in used_edges:
                avail_adj[u].add(v)
    
    # Check degree constraints: every vertex must have degree >= 2 in available graph
    for u in range(n):
        if len(avail_adj[u]) < 2:
            return None
    
    # DFS backtracking
    path = [start]
    visited = {start}
    
    # Order neighbors by degree (fewest first) for better pruning
    def _neighbors_sorted(v):
        return sorted(avail_adj[v], key=lambda x: len(avail_adj[x]))
    
    # Try to find a cycle
    def backtrack():
        if time.time() > deadline:
            return None
        if len(path) == n:
            # Check if last vertex connects to start
            last = path[-1]
            if start in avail_adj[last]:
                return list(path)
            return None
        
        current = path[-1]
        # Sort neighbors: prefer those with fewer options (most constrained)
        neighbors = _neighbors_sorted(current)
        
        for nb in neighbors:
            if nb in visited:
                continue
            # Pruning: if nb is not the last vertex to visit, it must have
            # at least one unvisited neighbor (or be able to connect to start if it's the last)
            # Simple pruning: check if removing this edge would isolate any unvisited vertex
            path.append(nb)
            visited.add(nb)
            
            result = backtrack()
            if result is not None:
                return result
            
            path.pop()
            visited.remove(nb)
        
        return None
    
    return backtrack()

def _find_hamiltonian_cycle_v2(adj, n, used_edges, deadline):
    """Try multiple starting vertices to find a Hamiltonian cycle."""
    # Try starting from each vertex, ordered by degree
    vertices_by_degree = sorted(range(n), key=lambda v: len(adj[v]))
    
    for start in vertices_by_degree:
        if time.time() > deadline:
            return None
        result = _find_hamiltonian_cycle(adj, n, start, used_edges, deadline)
        if result is not None:
            return result
    return None

def construct():
    deadline = time.time() + 50  # Leave 10s safety margin
    
    n = P['n']
    all_edges = [tuple(sorted(e)) for e in P['edges']]
    adj = _build_adjacency(all_edges, n)
    
    used_edges = set()
    cycles = []
    
    # Try to find up to 3 edge-disjoint Hamiltonian cycles
    for cycle_idx in range(3):
        if time.time() > deadline - 5:
            break
        
        # Try multiple strategies to find the next cycle
        found = None
        
        # Strategy 1: Standard backtracking with various starts
        result = _find_hamiltonian_cycle_v2(adj, n, used_edges, deadline - 5)
        if result is not None:
            found = result
        
        # Strategy 2: If not found, try with different neighbor orderings
        if found is None and time.time() < deadline - 5:
            # Try reversing the search or using different heuristics
            # We can try starting from high-degree vertices too
            for start in range(n):
                if time.time() > deadline - 5:
                    break
                avail_adj = defaultdict(set)
                for u in range(n):
                    for v in adj[u]:
                        ek = _edge_key(u, v)
                        if ek not in used_edges:
                            avail_adj[u].add(v)
                
                # Check degree constraints
                valid = True
                for u in range(n):
                    if len(avail_adj[u]) < 2:
                        valid = False
                        break
                if not valid:
                    continue
                
                path = [start]
                visited = {start}
                
                # Use a different ordering: highest degree first
                def _neighbors_sorted_h(v):
                    return sorted(avail_adj[v], key=lambda x: -len(avail_adj[x]))
                
                def backtrack2():
                    if time.time() > deadline - 5:
                        return None
                    if len(path) == n:
                        last = path[-1]
                        if start in avail_adj[last]:
                            return list(path)
                        return None
                    
                    current = path[-1]
                    neighbors = _neighbors_sorted_h(current)
                    
                    for nb in neighbors:
                        if nb in visited:
                            continue
                        path.append(nb)
                        visited.add(nb)
                        
                        result = backtrack2()
                        if result is not None:
                            return result
                        
                        path.pop()
                        visited.remove(nb)
                    
                    return None
                
                result = backtrack2()
                if result is not None:
                    found = result
                    break
        
        if found is None:
            break
        
        # Add edges of this cycle to used_edges
        for i in range(len(found)):
            u = found[i]
            v = found[(i + 1) % len(found)]
            used_edges.add(_edge_key(u, v))
        
        cycles.append(found)
    
    # Ensure we have at least one cycle (the fallback)
    if not cycles:
        # Fallback: try the trivial cycle if all edges are available
        if len(used_edges) == 0:
            test_cycle = list(range(n))
            valid = True
            for i in range(n):
                u = test_cycle[i]
                v = test_cycle[(i + 1) % n]
                if _edge_key(u, v) not in set(all_edges):
                    valid = False
                    break
            if valid:
                cycles.append(test_cycle)
    
    # Convert to proper format: list of lists of Python ints
    result = []
    for cyc in cycles:
        result.append([int(v) for v in cyc])
    
    return result
# EVOLVE-BLOCK-END
