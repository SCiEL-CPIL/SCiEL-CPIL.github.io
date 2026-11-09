"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict

sys.setrecursionlimit(10000)

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _edge_key(u, v):
    if u > v:
        u, v = v, u
    return (u, v)

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_hamiltonian_cycle(adj, n, used_edges, deadline, start_vertex=0):
    """Find a Hamiltonian cycle using backtracking with pruning."""
    
    # Precompute neighbor lists sorted by degree (ascending) for better pruning
    deg = [len(adj[v]) for v in range(n)]
    neighbor_order = {v: sorted(adj[v], key=lambda x: deg[x]) for v in range(n)}
    
    def backtrack(visited, path, current, depth):
        if time.time() > deadline:
            return None
        
        if depth == n:
            # Check if we can close the cycle
            edge = _edge_key(current, path[0])
            if edge not in used_edges:
                return path[:]
            return None
        
        # Pruning: check if remaining unvisited vertices are still reachable
        # Simple check: if any unvisited vertex has no available edges to unvisited vertices or current
        for v in range(n):
            if v not in visited:
                # Check if v can still be reached
                can_reach = False
                for nb in adj[v]:
                    if nb == current:
                        if _edge_key(v, nb) not in used_edges:
                            can_reach = True
                            break
                    elif nb not in visited:
                        if _edge_key(v, nb) not in used_edges:
                            can_reach = True
                            break
                if not can_reach:
                    return None
        
        for neighbor in neighbor_order[current]:
            if neighbor not in visited:
                edge = _edge_key(current, neighbor)
                if edge not in used_edges:
                    visited.add(neighbor)
                    path.append(neighbor)
                    result = backtrack(visited, path, neighbor, depth + 1)
                    if result is not None:
                        return result
                    path.pop()
                    visited.discard(neighbor)
        return None
    
    # Try starting from the given vertex
    visited = {start_vertex}
    path = [start_vertex]
    result = backtrack(visited, path, start_vertex, 1)
    if result is not None:
        return result
    
    # If that didn't work, try other starting vertices
    for start in range(n):
        if time.time() > deadline:
            break
        visited = {start}
        path = [start]
        result = backtrack(visited, path, start, 1)
        if result is not None:
            return result
    
    return None

def _try_circulant_steps(adj, n, used_edges, deadline):
    """Try to find Hamiltonian cycles using circulant step patterns."""
    cycles = []
    all_edges = set()
    for u, v in P['edges']:
        all_edges.add(_edge_key(u, v))
    
    # For a circulant graph, step k gives a Hamiltonian cycle if gcd(k,n)=1
    # Check if all edges {i, i+k mod n} exist in the graph
    for k in range(1, n):
        if time.time() > deadline:
            break
        # Check if gcd(k, n) == 1
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a
        if gcd(k, n) != 1:
            continue
        
        # Check if all edges exist
        valid = True
        for i in range(n):
            u = i
            v = (i + k) % n
            if _edge_key(u, v) not in all_edges:
                valid = False
                break
            if _edge_key(u, v) in used_edges:
                valid = False
                break
        
        if valid:
            # Generate the cycle
            cycle = []
            v = 0
            for _ in range(n):
                cycle.append(v)
                v = (v + k) % n
            
            # Verify it's a valid cycle
            valid_cycle = True
            for i in range(n):
                u = cycle[i]
                w = cycle[(i + 1) % n]
                if _edge_key(u, w) not in all_edges:
                    valid_cycle = False
                    break
                if _edge_key(u, w) in used_edges:
                    valid_cycle = False
                    break
            
            if valid_cycle:
                cycles.append(cycle)
    
    return cycles

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s safety margin
    
    adj = _build_adj()
    n = P['n']
    
    used_edges = set()
    cycles = []
    
    # First, try circulant construction for fast guaranteed cycles
    circulant_cycles = _try_circulant_steps(adj, n, used_edges, deadline)
    for cycle in circulant_cycles:
        if time.time() > deadline - 5:
            break
        # Mark edges as used
        valid = True
        for j in range(len(cycle)):
            u = cycle[j]
            v = cycle[(j + 1) % len(cycle)]
            edge = _edge_key(u, v)
            if edge in used_edges:
                valid = False
                break
            used_edges.add(edge)
        if valid:
            cycles.append(cycle)
    
    # Then, use backtracking to find additional cycles
    for i in range(10 - len(cycles)):
        if time.time() > deadline - 5:
            break
        
        # Try different starting vertices for diversity
        start_vertex = i % n
        cycle = _find_hamiltonian_cycle(adj, n, used_edges, deadline, start_vertex)
        if cycle is None:
            # Try other starting vertices
            found = False
            for start in range(n):
                if time.time() > deadline - 5:
                    break
                cycle = _find_hamiltonian_cycle(adj, n, used_edges, deadline, start)
                if cycle is not None:
                    found = True
                    break
            if not found:
                break
        
        # Mark edges as used
        for j in range(len(cycle)):
            u = cycle[j]
            v = cycle[(j + 1) % len(cycle)]
            edge = _edge_key(u, v)
            used_edges.add(edge)
        
        cycles.append(cycle)
    
    # Ensure we have at least one cycle
    if not cycles:
        # Fallback: try a simpler approach
        if time.time() < deadline:
            cycle = _find_hamiltonian_cycle(adj, n, used_edges, deadline, 0)
            if cycle is not None:
                for j in range(len(cycle)):
                    u = cycle[j]
                    v = cycle[(j + 1) % len(cycle)]
                    edge = _edge_key(u, v)
                    used_edges.add(edge)
                cycles.append(cycle)
    
    # Convert to lists of Python ints
    result = []
    for cycle in cycles:
        result.append([int(v) for v in cycle])
    
    return result
# EVOLVE-BLOCK-END
