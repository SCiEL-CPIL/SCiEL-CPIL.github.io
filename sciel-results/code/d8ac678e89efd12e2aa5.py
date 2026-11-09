"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Check if the graph is regular and determine the degree
    degrees = [len(adj[i]) for i in range(n)]
    if all(d == degrees[0] for d in degrees):
        k = degrees[0]
    else:
        k = min(degrees)
    
    # We want to find up to k edge-disjoint Hamiltonian cycles
    # Use a greedy approach with backtracking
    
    def find_hamiltonian_cycle(adj, n, start=0):
        """Find a Hamiltonian cycle in the graph using DFS with backtracking."""
        visited = [False] * n
        path = [start]
        visited[start] = True
        
        def backtrack(u):
            if len(path) == n:
                # Check if we can close the cycle
                if start in adj[u]:
                    return True
                return False
            
            for v in list(adj[u]):
                if not visited[v]:
                    # Symmetry breaking: avoid exploring equivalent paths
                    # Only allow visiting vertices in increasing order for non-start vertices
                    # This is a strong heuristic that works well for many graphs
                    if v > start or len(path) == 1:
                        # Additional pruning: ensure we don't get stuck
                        if len(path) == n - 1:
                            if start not in adj[v]:
                                continue
                        
                        visited[v] = True
                        path.append(v)
                        if backtrack(v):
                            return True
                        path.pop()
                        visited[v] = False
            
            return False
        
        if backtrack(start):
            return path[:]
        return None
    
    # Try to find k edge-disjoint Hamiltonian cycles
    cycles = []
    remaining_edges = set()
    for u, v in edges:
        if u < v:
            remaining_edges.add((u, v))
        else:
            remaining_edges.add((v, u))
    
    adj_remaining = {i: set() for i in range(n)}
    for u, v in remaining_edges:
        adj_remaining[u].add(v)
        adj_remaining[v].add(u)
    
    for _ in range(k):
        # Try to find a Hamiltonian cycle in the remaining graph
        cycle = find_hamiltonian_cycle(adj_remaining, n)
        if cycle is None:
            # Try different starting vertices
            found = False
            for start in range(n):
                cycle = find_hamiltonian_cycle(adj_remaining, n, start)
                if cycle is not None:
                    found = True
                    break
            if not found:
                break
        
        # Remove the edges used in this cycle
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            if u < v:
                remaining_edges.discard((u, v))
                adj_remaining[u].discard(v)
                adj_remaining[v].discard(u)
            else:
                remaining_edges.discard((v, u))
                adj_remaining[v].discard(u)
                adj_remaining[u].discard(v)
        
        cycles.append(cycle)
    
    # If we couldn't find k cycles, try a different approach
    # Use a randomized restart strategy
    if len(cycles) < k:
        best_cycles = cycles[:]
        
        deadline = time.time() + 80  # 80 seconds to be safe
        attempts = 0
        
        while time.time() < deadline and attempts < 100:
            attempts += 1
            adj_attempt = {i: set(adj[j]) for i, j in zip(range(n), range(n))}
            # Note: this creates a shallow copy, need to be careful
            adj_attempt = {i: set() for i in range(n)}
            for u, v in edges:
                adj_attempt[u].add(v)
                adj_attempt[v].add(u)
            
            found_cycles = []
            for _ in range(k):
                cycle = find_hamiltonian_cycle(adj_attempt, n)
                if cycle is None:
                    for start in range(n):
                        cycle = find_hamiltonian_cycle(adj_attempt, n, start)
                        if cycle is not None:
                            break
                if cycle is None:
                    break
                
                # Remove edges
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    if u in adj_attempt[v]:
                        adj_attempt[u].remove(v)
                        adj_attempt[v].remove(u)
                
                found_cycles.append(cycle)
            
            if len(found_cycles) > len(best_cycles):
                best_cycles = found_cycles[:]
                if len(best_cycles) == k:
                    break
    
    cycles = best_cycles
    
    # Fallback: ensure we return at least one valid cycle
    if not cycles:
        # The seed cycle [0, 1, 2, ..., 20] is valid
        cycles = [[i for i in range(n)]]
    
    # Convert to list of lists of ints
    result = [list(c) for c in cycles]
    
    return result
# EVOLVE-BLOCK-END
