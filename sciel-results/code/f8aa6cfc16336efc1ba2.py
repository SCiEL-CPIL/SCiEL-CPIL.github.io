"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _edge_key(u, v):
    if u > v:
        u, v = v, u
    return (u, v)

def _build_adjacency():
    n = P['n']
    adj = defaultdict(set)
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _verify_cycle(cycle, adj, used_edges):
    """Verify a cycle is valid and edge-disjoint."""
    if len(cycle) < 3:
        return False
    if len(set(cycle)) != len(cycle):
        return False
    for i in range(len(cycle)):
        u = cycle[i]
        v = cycle[(i + 1) % len(cycle)]
        if v not in adj[u]:
            return False
        e = _edge_key(u, v)
        if e in used_edges:
            return False
    return True

def _find_cycle_dfs(adj, n, used_edges, start, deadline, neighbor_order):
    """Find a Hamiltonian cycle using DFS with given start and neighbor ordering."""
    visited = {start}
    path = [start]
    
    def dfs(current, depth):
        if time.time() > deadline:
            return None
        if depth == n:
            edge = _edge_key(current, path[0])
            if edge not in used_edges:
                return path[:]
            return None
        
        for neighbor in neighbor_order[current]:
            if neighbor not in visited:
                edge = _edge_key(current, neighbor)
                if edge not in used_edges:
                    visited.add(neighbor)
                    path.append(neighbor)
                    result = dfs(neighbor, depth + 1)
                    if result is not None:
                        return result
                    path.pop()
                    visited.discard(neighbor)
        return None
    
    return dfs(start, 1)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    adj = _build_adjacency()
    n = P['n']
    
    used_edges = set()
    cycles = []
    
    # Precompute neighbor lists for all vertices
    neighbor_lists = {v: list(adj[v]) for v in range(n)}
    
    # Strategy: Try multiple orderings of neighbors to find edge-disjoint cycles
    # Use a combination of deterministic and randomized approaches
    
    # First, try to find cycles using different neighbor orderings
    orderings = []
    
    # Ordering 1: sorted
    orderings.append({v: sorted(neighbor_lists[v]) for v in range(n)})
    
    # Ordering 2: reverse sorted
    orderings.append({v: sorted(neighbor_lists[v], reverse=True) for v in range(n)})
    
    # Ordering 3-10: random shuffles
    rng = random.Random(42)
    for _ in range(7):
        orderings.append({v: list(neighbor_lists[v]) for v in range(n)})
        for v in range(n):
            rng.shuffle(orderings[-1][v])
    
    # Try each ordering to find up to 3 cycles
    for ordering in orderings:
        if time.time() > deadline - 5:
            break
        if len(cycles) >= 3:
            break
        
        for start in range(n):
            if time.time() > deadline - 5:
                break
            if len(cycles) >= 3:
                break
            
            cycle = _find_cycle_dfs(adj, n, used_edges, start, deadline, ordering)
            if cycle is not None:
                # Verify the cycle
                if _verify_cycle(cycle, adj, used_edges):
                    # Mark edges as used
                    for j in range(len(cycle)):
                        u = cycle[j]
                        v = cycle[(j + 1) % len(cycle)]
                        edge = _edge_key(u, v)
                        used_edges.add(edge)
                    cycles.append(cycle)
    
    # If we haven't found 3 cycles, try more aggressive randomized search
    if len(cycles) < 3 and time.time() < deadline - 10:
        rng2 = random.Random(123)
        attempts = 0
        max_attempts = 200
        
        while len(cycles) < 3 and attempts < max_attempts and time.time() < deadline - 5:
            attempts += 1
            
            # Random neighbor ordering
            ordering = {v: list(neighbor_lists[v]) for v in range(n)}
            for v in range(n):
                rng2.shuffle(ordering[v])
            
            # Random start
            start = rng2.randint(0, n - 1)
            
            cycle = _find_cycle_dfs(adj, n, used_edges, start, deadline, ordering)
            if cycle is not None:
                if _verify_cycle(cycle, adj, used_edges):
                    for j in range(len(cycle)):
                        u = cycle[j]
                        v = cycle[(j + 1) % len(cycle)]
                        edge = _edge_key(u, v)
                        used_edges.add(edge)
                    cycles.append(cycle)
    
    # Fallback: if we have fewer than 3 cycles, try one more targeted approach
    if len(cycles) < 3 and time.time() < deadline - 2:
        # Try to find cycles with specific structural properties
        # Use a greedy approach: at each step, prefer edges that keep more options open
        
        def find_cycle_greedy():
            if time.time() > deadline:
                return None
            
            for start in range(n):
                visited = {start}
                path = [start]
                
                def dfs_greedy(current, depth):
                    if time.time() > deadline:
                        return None
                    if depth == n:
                        edge = _edge_key(current, path[0])
                        if edge not in used_edges:
                            return path[:]
                        return None
                    
                    # Score neighbors: prefer those with fewer remaining unused edges
                    candidates = []
                    for neighbor in neighbor_lists[current]:
                        if neighbor not in visited:
                            edge = _edge_key(current, neighbor)
                            if edge not in used_edges:
                                # Count unused edges from neighbor
                                unused_count = sum(1 for nb in neighbor_lists[neighbor] 
                                                  if _edge_key(neighbor, nb) not in used_edges)
                                candidates.append((unused_count, neighbor))
                    
                    # Sort by unused count (prefer vertices with fewer options to reduce branching)
                    candidates.sort()
                    
                    for _, neighbor in candidates:
                        edge = _edge_key(current, neighbor)
                        if edge not in used_edges:
                            visited.add(neighbor)
                            path.append(neighbor)
                            result = dfs_greedy(neighbor, depth + 1)
                            if result is not None:
                                return result
                            path.pop()
                            visited.discard(neighbor)
                    return None
                
                result = dfs_greedy(start, 1)
                if result is not None:
                    return result
            
            return None
        
        cycle = find_cycle_greedy()
        if cycle is not None:
            if _verify_cycle(cycle, adj, used_edges):
                for j in range(len(cycle)):
                    u = cycle[j]
                    v = cycle[(j + 1) % len(cycle)]
                    edge = _edge_key(u, v)
                    used_edges.add(edge)
                cycles.append(cycle)
    
    # Ensure we have at least one cycle
    if not cycles:
        # Last resort: find any Hamiltonian cycle
        for start in range(n):
            visited = {start}
            path = [start]
            
            def simple_dfs(current):
                if time.time() > deadline:
                    return None
                if len(path) == n:
                    if _edge_key(current, path[0]) not in used_edges:
                        return path[:]
                    return None
                for neighbor in neighbor_lists[current]:
                    if neighbor not in visited:
                        if _edge_key(current, neighbor) not in used_edges:
                            visited.add(neighbor)
                            path.append(neighbor)
                            result = simple_dfs(neighbor)
                            if result is not None:
                                return result
                            path.pop()
                            visited.discard(neighbor)
                return None
            
            result = simple_dfs(start)
            if result is not None:
                cycles.append(result)
                break
    
    # Convert to lists of Python ints
    result = []
    for cycle in cycles:
        result.append([int(v) for v in cycle])
    
    return result
# EVOLVE-BLOCK-END
