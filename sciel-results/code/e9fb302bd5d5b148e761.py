"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adjacency(edges):
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _check_feasibility_fast(adj, remaining_edges, n):
    """Check if each vertex still has degree >= 2 in remaining edges."""
    for v in range(n):
        deg = 0
        for u in adj[v]:
            if _edge_key(v, u) in remaining_edges:
                deg += 1
                if deg >= 2:
                    break
        if deg < 2:
            return False
    return True

def _find_hamiltonian_cycle_optimized(adj, remaining_edges, n, deadline=None, start_vertex=0):
    """Find a single Hamiltonian cycle using optimized DFS with better pruning."""
    if deadline is not None and time.time() > deadline:
        return None
    
    # Precompute edge lookup for O(1) access
    edge_set = remaining_edges
    
    # Use iterative deepening with time budget
    def dfs(path, visited, depth):
        if deadline is not None and time.time() > deadline:
            return None
        if len(path) == n:
            first = path[0]
            last = path[-1]
            ek = _edge_key(first, last)
            if ek in edge_set:
                return path[:]
            return None
        
        last = path[-1]
        # Get unvisited neighbors with available edges
        candidates = []
        for neighbor in adj[last]:
            if not visited[neighbor]:
                ek = _edge_key(last, neighbor)
                if ek in edge_set:
                    # Count available edges from neighbor to unvisited vertices
                    avail_count = 0
                    for nb in adj[neighbor]:
                        if not visited[nb] and _edge_key(neighbor, nb) in edge_set:
                            avail_count += 1
                    # Also count edge back to start if last vertex
                    if len(path) == n - 1 and _edge_key(neighbor, path[0]) in edge_set:
                        avail_count += 1
                    candidates.append((avail_count, neighbor))
        
        if not candidates:
            return None
        
        # Sort by fewest available edges first (MRV heuristic)
        candidates.sort()
        
        for _, neighbor in candidates:
            ek = _edge_key(last, neighbor)
            if ek not in edge_set:
                continue
            visited[neighbor] = True
            edge_set.remove(ek)
            result = dfs(path + [neighbor], visited, depth + 1)
            if result is not None:
                return result
            visited[neighbor] = False
            edge_set.add(ek)
        return None
    
    visited = [False] * n
    visited[start_vertex] = True
    # Remove the starting edge from consideration temporarily
    result = dfs([start_vertex], visited, 0)
    return result

def _try_decompose_into_cycles(adj, available_edges, max_cycles=10, deadline=None):
    """Iteratively find edge-disjoint Hamiltonian cycles with multiple strategies."""
    n = P['n']
    cycles = []
    remaining = set(available_edges)
    
    for _ in range(max_cycles):
        if deadline is not None and time.time() > deadline:
            break
        if not _check_feasibility_fast(adj, remaining, n):
            break
        
        # Try multiple starting vertices
        found = None
        for start in range(n):
            if deadline is not None and time.time() > deadline:
                break
            cycle = _find_hamiltonian_cycle_optimized(adj, remaining, n, deadline, start)
            if cycle is not None:
                found = cycle
                break
        
        if found is None:
            break
        
        cycles.append(found)
        for i in range(len(found)):
            u = found[i]
            v = found[(i + 1) % len(found)]
            remaining.discard(_edge_key(u, v))
    
    return cycles

def _find_hamiltonian_cycles(adj, available_edges, max_cycles=10, deadline=None):
    """Find up to max_cycles edge-disjoint Hamiltonian cycles with randomized restarts."""
    n = P['n']
    best_cycles = []
    
    # Try deterministic first
    cycles = _try_decompose_into_cycles(adj, available_edges, max_cycles, deadline)
    if len(cycles) > len(best_cycles):
        best_cycles = cycles
    
    if deadline is not None and time.time() > deadline:
        return best_cycles
    
    # Try randomized restarts if time permits
    for restart in range(30):
        if deadline is not None and time.time() > deadline:
            break
        if len(best_cycles) >= max_cycles:
            break
        
        # Randomize the order of starting vertices
        starts = list(range(n))
        random.shuffle(starts)
        
        old_state = random.getstate()
        random.seed(restart * 999 + 42)
        
        cycles = _try_decompose_into_cycles(adj, available_edges, max_cycles, deadline)
        
        random.setstate(old_state)
        
        if len(cycles) > len(best_cycles):
            best_cycles = cycles
    
    return best_cycles

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80 second safety margin (10s buffer before 90s kill)
    
    adj = _build_adjacency(P['edges'])
    available = set(_edge_key(u, v) for u, v in P['edges'])
    
    # Try to find as many edge-disjoint Hamiltonian cycles as possible
    cycles = _find_hamiltonian_cycles(adj, available, max_cycles=10, deadline=deadline)
    
    if len(cycles) >= 1:
        return [list(c) for c in cycles]
    else:
        # Fallback: return the known Hamiltonian cycle
        return [list(range(P['n']))]
# EVOLVE-BLOCK-END
