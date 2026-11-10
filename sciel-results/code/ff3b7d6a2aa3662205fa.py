"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_hamiltonian_cycle(adj, n, rng, deadline, max_steps=500000):
    """Find a Hamiltonian cycle using randomized DFS with backtracking."""
    start = rng.randint(0, n - 1)
    
    def backtrack(path, used_edges, steps):
        if steps >= max_steps:
            return None
        if time.time() > deadline:
            return None
        if len(path) == n:
            if start in adj[path[-1]]:
                return list(path)
            return None
        last = path[-1]
        candidates = [v for v in adj[last] if (min(last, v), max(last, v)) not in used_edges]
        # Prefer vertices with fewer remaining options (degree heuristic)
        candidates.sort(key=lambda v: len([w for w in adj[v] if (min(v, w), max(v, w)) not in used_edges and w not in set(path)]))
        rng.shuffle(candidates)
        # Re-sort to preserve heuristic but with random tie-breaking
        for v in candidates:
            if v == start and len(path) < n - 1:
                continue
            new_used = used_edges | {(min(last, v), max(last, v))}
            result = backtrack(path + [v], new_used, steps + 1)
            if result is not None:
                return result
        return None
    
    return backtrack([start], set(), 0)

def _try_find_cycles(n, edges, rng, deadline, max_cycles=10):
    """Try to find multiple edge-disjoint Hamiltonian cycles."""
    adj = _build_adj(edges, n)
    cycles = []
    remaining_edges = set((min(u, v), max(u, v)) for u, v in edges)
    
    for _ in range(max_cycles):
        if time.time() > deadline - 5:
            break
        # Build adjacency from remaining edges
        rem_adj = [set() for _ in range(n)]
        for (u, v) in remaining_edges:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
        
        # Check if each vertex has degree >= 2 (necessary for Hamiltonian cycle)
        valid = True
        for i in range(n):
            if len(rem_adj[i]) < 2:
                valid = False
                break
        if not valid:
            break
        
        # Try multiple random starts
        found = None
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            cycle = _find_hamiltonian_cycle(rem_adj, n, rng, deadline)
            if cycle is not None:
                found = cycle
                break
        
        if found is None:
            break
        
        # Remove edges used by this cycle
        cycle_set = set()
        for i in range(len(found)):
            u, v = found[i], found[(i + 1) % len(found)]
            e = (min(u, v), max(u, v))
            if e in remaining_edges:
                remaining_edges.remove(e)
                cycle_set.add(e)
        
        cycles.append(found)
    
    return cycles

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s to leave 10s safety margin
    rng = random.Random(42)
    n = P['n']
    edges = P['edges']
    
    best_cycles = None
    best_count = 0
    
    # Multiple random restarts with different seeds
    for seed in range(50):
        if time.time() > deadline - 10:
            break
        rng = random.Random(seed)
        cycles = _try_find_cycles(n, edges, rng, deadline)
        if len(cycles) > best_count:
            best_count = len(cycles)
            best_cycles = cycles
            if best_count >= 10:
                break
    
    if best_cycles is None or len(best_cycles) == 0:
        # Fallback: return a single Hamiltonian cycle if we can find one
        rng = random.Random(0)
        adj = _build_adj(edges, n)
        cycle = _find_hamiltonian_cycle(adj, n, rng, deadline)
        if cycle is not None:
            return [list(cycle)]
        # Last resort: return trivial cycle
        return [list(range(n))]
    
    # Ensure all cycles use the same vertex set
    vertex_set = set(range(n))
    valid_cycles = []
    for cycle in best_cycles:
        if set(cycle) == vertex_set and len(cycle) == n:
            valid_cycles.append(list(cycle))
    
    if not valid_cycles:
        return [list(range(n))]
    
    return valid_cycles
# EVOLVE-BLOCK-END
