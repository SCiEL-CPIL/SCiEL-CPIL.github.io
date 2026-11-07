"""Find edge-disjoint Hamiltonian cycles on the 21-vertex 6-regular host graph."""
import time
import random
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adjacency(edges, n):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _is_connected(adj, n, start, visited=None):
    """Check if all unvisited vertices are reachable from start in the remaining graph."""
    if visited is None:
        visited = [False] * n
        visited[start] = True
    
    # BFS/DFS to see if we can reach all unvisited vertices
    to_visit = [start]
    seen = {start}
    while to_visit:
        u = to_visit.pop()
        for v in adj[u]:
            if not visited[v] and v not in seen:
                seen.add(v)
                to_visit.append(v)
    
    # All unvisited vertices must be in seen
    for v in range(n):
        if not visited[v] and v not in seen:
            return False
    return True

def _find_hamiltonian_cycle(adj, n, start=0, rng=None, deadline=None):
    """Find a Hamiltonian cycle starting at `start` using backtracking + Warnsdorff heuristic."""
    if deadline is not None and time.time() > deadline:
        return None
    
    visited = [False] * n
    visited[start] = True
    path = [start]
    
    # Pre-check: ensure all other vertices are reachable
    if not _is_connected(adj, n, start, visited):
        return None
    
    def dfs(current):
        if deadline is not None and time.time() > deadline:
            return None
        if len(path) == n:
            if start in adj[current]:
                return list(path)
            return None
        
        # Get unvisited neighbors
        neighbors = [v for v in adj[current] if not visited[v]]
        if not neighbors:
            return None
        
        # Connectivity check: after moving to each neighbor, ensure remaining vertices are reachable
        # (Do this for the best candidate to fail fast)
        if len(neighbors) > 1:
            # Quick check: is the graph still connected enough?
            unvisited_count = n - len(path)
            if unvisited_count > 0:
                # For each neighbor, check if it can reach all other unvisited vertices
                valid_neighbors = []
                for v in neighbors:
                    temp_visited = visited.copy()
                    temp_visited[v] = True
                    if _is_connected(adj, n, v, temp_visited):
                        valid_neighbors.append(v)
                    temp_visited[v] = False
                if valid_neighbors:
                    neighbors = valid_neighbors
                # If no valid neighbors, we're stuck
        
        # Warnsdorff: sort by number of onward unvisited neighbors (ascending)
        if rng is not None:
            scored = []
            for v in neighbors:
                onward = sum(1 for w in adj[v] if not visited[w] and w != start)
                # Add small random perturbation
                scored.append((onward + rng.random() * 0.3, v))
            scored.sort()
            neighbors = [v for _, v in scored]
        else:
            neighbors.sort(key=lambda v: sum(1 for w in adj[v] if not visited[w] and w != start))
        
        for v in neighbors:
            visited[v] = True
            path.append(v)
            result = dfs(v)
            if result is not None:
                return result
            path.pop()
            visited[v] = False
        
        return None
    
    return dfs(start)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    n = P['n']
    edges = P['edges']
    
    # Build edge set for quick lookup
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))
    
    # Build adjacency
    adj = _build_adjacency(edges, n)
    
    # Try to find 3 edge-disjoint Hamiltonian cycles
    # Strategy: iteratively find a Hamiltonian cycle, remove its edges, repeat
    
    best_cycles = []
    
    # Use a fixed seed for reproducibility, but try multiple seeds if first fails
    # Try more seeds with tighter per-attempt time limits
    seeds = [42, 123, 999, 7, 314, 271, 137, 555, 888, 101, 202, 303, 456, 789, 111, 222, 333, 444, 555, 666]
    
    for seed in seeds:
        if time.time() > deadline:
            break
        
        # Allocate remaining time proportionally to cycles still needed
        remaining_time = deadline - time.time()
        if remaining_time < 1.0:
            break
        
        rng = random.Random(seed)
        cycles = []
        current_adj = {v: set(neighbors) for v, neighbors in adj.items()}
        remaining_edges = set(edge_set)
        
        # Time budget per cycle attempt: use most of remaining time for first cycle,
        # less for subsequent ones
        for cycle_idx in range(3):
            if time.time() > deadline:
                break
            
            # Allocate time: give more time to earlier cycles
            time_for_this_cycle = remaining_time * (0.6 if cycle_idx == 0 else 0.3 if cycle_idx == 1 else 0.1)
            cycle_deadline = time.time() + time_for_this_cycle
            if cycle_deadline > deadline:
                cycle_deadline = deadline
            
            # Try multiple starting vertices with different strategies
            found = False
            
            # Strategy 1: Try starts in order of increasing degree in remaining graph
            # (more constrained vertices first)
            starts_by_degree = sorted(range(n), key=lambda v: len(current_adj[v]))
            
            # Also try random order
            starts_random = list(range(n))
            rng.shuffle(starts_random)
            
            # Interleave: try constrained starts first, then random
            all_starts = starts_by_degree[:5] + starts_random[5:]
            rng.shuffle(all_starts)
            # Put the most constrained ones first
            all_starts = sorted(all_starts, key=lambda v: (len(current_adj[v]), rng.random()))
            
            for start in all_starts:
                if time.time() > cycle_deadline:
                    break
                
                # Find Hamiltonian cycle in current_adj
                cycle = _find_hamiltonian_cycle(current_adj, n, start=start, rng=rng, deadline=cycle_deadline)
                
                if cycle is not None:
                    # Verify the cycle uses only remaining edges
                    valid = True
                    cycle_edges = set()
                    for i in range(len(cycle)):
                        u = cycle[i]
                        v = cycle[(i + 1) % len(cycle)]
                        e = (min(u, v), max(u, v))
                        if e not in remaining_edges:
                            valid = False
                            break
                        cycle_edges.add(e)
                    
                    if valid and len(cycle) == n and len(set(cycle)) == n:
                        cycles.append(cycle)
                        # Remove these edges
                        for e in cycle_edges:
                            remaining_edges.discard(e)
                            u, v = e
                            current_adj[u].discard(v)
                            current_adj[v].discard(u)
                        found = True
                        break
            
            if not found:
                break
        
        if len(cycles) > len(best_cycles):
            best_cycles = cycles
        
        if len(best_cycles) == 3:
            break
    
    # Return the best set of cycles found
    if not best_cycles:
        # Fallback: return a single cycle
        adj_full = _build_adjacency(edges, n)
        cycle = _find_hamiltonian_cycle(adj_full, n, start=0, rng=random.Random(0), deadline=deadline)
        if cycle is not None:
            return [cycle]
        return [list(range(n))]
    
    # Ensure types are correct
    result = []
    for cycle in best_cycles:
        result.append([int(v) for v in cycle])
    return result
