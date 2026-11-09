"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill
    n = P['n']
    all_edges = P['edges']
    
    # Build adjacency
    adj = defaultdict(set)
    for u, v in all_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    def edge_key(u, v):
        if u > v:
            u, v = v, u
        return (u, v)
    
    def find_hamiltonian_cycle(available_edges, start, rng, max_steps=500000):
        """Find a Hamiltonian cycle starting from 'start' using backtracking with Warnsdorff heuristic."""
        available_set = set()
        for e in available_edges:
            available_set.add(edge_key(e[0], e[1]))
        
        # Check if start has at least 2 available neighbors
        start_neighbors = [v for v in adj[start] if edge_key(start, v) in available_set]
        if len(start_neighbors) < 2:
            return None
        
        path = [start]
        visited = {start}
        used_edges = set()
        
        def backtrack():
            if len(path) == n:
                # Check if we can close the cycle
                if edge_key(path[-1], start) in available_set and edge_key(path[-1], start) not in used_edges:
                    return True
                return False
            
            current = path[-1]
            # Get unvisited neighbors that have available edges
            candidates = []
            for v in adj[current]:
                if v not in visited and edge_key(current, v) in available_set and edge_key(current, v) not in used_edges:
                    # Warnsdorff: count unvisited neighbors of v
                    unvisited_count = 0
                    for w in adj[v]:
                        if w not in visited and edge_key(v, w) in available_set and edge_key(v, w) not in used_edges:
                            unvisited_count += 1
                    candidates.append((unvisited_count, v))
            
            if not candidates:
                return False
            
            # Sort by Warnsdorff heuristic (fewest unvisited neighbors first)
            candidates.sort(key=lambda x: x[0])
            
            for _, v in candidates:
                ek = edge_key(current, v)
                path.append(v)
                visited.add(v)
                used_edges.add(ek)
                
                if backtrack():
                    return True
                
                path.pop()
                visited.remove(v)
                used_edges.discard(ek)
            
            return False
        
        if backtrack():
            return list(path)
        return None
    
    def find_hamiltonian_cycle_random(available_edges, rng, max_attempts=200):
        """Try multiple random starts to find a Hamiltonian cycle."""
        # Try different starting vertices, prioritizing those with high degree
        vertices = list(range(n))
        # Sort by degree (descending) for better heuristics
        vertices.sort(key=lambda v: len([e for e in available_edges if v in e]), reverse=True)
        
        # Shuffle for diversity
        rng.shuffle(vertices)
        
        for start in vertices:
            cycle = find_hamiltonian_cycle(available_edges, start, rng)
            if cycle is not None:
                return cycle
        return None
    
    def find_hamiltonian_cycle_deep(available_edges, rng, max_steps=2000000):
        """More aggressive search with multiple starts and deeper backtracking."""
        vertices = list(range(n))
        vertices.sort(key=lambda v: len([e for e in available_edges if v in e]), reverse=True)
        
        # Try each vertex as start, with randomized neighbor ordering
        for start in vertices:
            # Reset and try
            path = [start]
            visited = {start}
            used_edges = set()
            available_set = set()
            for e in available_edges:
                available_set.add(edge_key(e[0], e[1]))
            
            steps = 0
            
            def backtrack_deep():
                nonlocal steps
                steps += 1
                if steps > max_steps:
                    return False
                
                if len(path) == n:
                    if edge_key(path[-1], start) in available_set and edge_key(path[-1], start) not in used_edges:
                        return True
                    return False
                
                current = path[-1]
                candidates = []
                for v in adj[current]:
                    if v not in visited and edge_key(current, v) in available_set and edge_key(current, v) not in used_edges:
                        unvisited_count = 0
                        for w in adj[v]:
                            if w not in visited and edge_key(v, w) in available_set and edge_key(v, w) not in used_edges:
                                unvisited_count += 1
                        candidates.append((unvisited_count, v))
                
                if not candidates:
                    return False
                
                candidates.sort(key=lambda x: x[0])
                
                for _, v in candidates:
                    ek = edge_key(current, v)
                    path.append(v)
                    visited.add(v)
                    used_edges.add(ek)
                    
                    if backtrack_deep():
                        return True
                    
                    path.pop()
                    visited.remove(v)
                    used_edges.discard(ek)
                
                return False
            
            if backtrack_deep():
                return list(path)
        
        return None
    
    # Main search: iteratively find edge-disjoint Hamiltonian cycles
    best_cycles = []
    available_edges = list(all_edges)
    rng = random.Random(42)
    
    # First, try to find as many cycles as possible
    max_cycles = 10
    for cycle_idx in range(max_cycles):
        if time.time() > deadline:
            break
        
        if not available_edges:
            break
        
        found = False
        # Try multiple strategies
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Shuffle available edges to get different search paths
            shuffled_edges = list(available_edges)
            rng.shuffle(shuffled_edges)
            
            cycle = find_hamiltonian_cycle_random(shuffled_edges, rng)
            if cycle is None:
                cycle = find_hamiltonian_cycle_deep(available_edges, rng)
            
            if cycle is not None:
                # Verify the cycle
                valid = True
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    if edge_key(u, v) not in set(edge_key(e[0], e[1]) for e in available_edges):
                        valid = False
                        break
                if valid and len(set(cycle)) == n:
                    best_cycles.append(cycle)
                    # Remove used edges
                    used_in_cycle = set()
                    for i in range(len(cycle)):
                        u = cycle[i]
                        v = cycle[(i + 1) % len(cycle)]
                        used_in_cycle.add(edge_key(u, v))
                    available_edges = [e for e in available_edges if edge_key(e[0], e[1]) not in used_in_cycle]
                    found = True
                    break
        
        if not found:
            break
    
    # Return the best result found
    if best_cycles:
        # Ensure proper types
        result = []
        for cycle in best_cycles:
            result.append([int(v) for v in cycle])
        return result
    else:
        # Fallback: return a single cycle if we can find one, else empty
        return []
# EVOLVE-BLOCK-END
