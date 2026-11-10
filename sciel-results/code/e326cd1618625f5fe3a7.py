"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adjacency(edges, n):
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _build_edge_set(edges):
    return set((min(u,v), max(u,v)) for u,v in edges)

def _find_ham_cycle_backtrack(adj, n, forbidden_edges, deadline=None):
    """Find a Hamiltonian cycle avoiding forbidden edges using DFS with strong pruning."""
    if deadline is not None and time.time() > deadline:
        return None
    
    def dfs(path, visited, current, remaining):
        if deadline is not None and time.time() > deadline:
            return None
        if len(path) == n:
            if current in adj.get(0, set()) and (0, current) not in forbidden_edges and (current, 0) not in forbidden_edges:
                return path
            return None
        
        if len(path) == n - 1:
            for neighbor in adj.get(current, set()):
                if neighbor not in visited:
                    if neighbor == 0:
                        continue  # Can't close yet
                    if (min(current, neighbor), max(current, neighbor)) in forbidden_edges:
                        continue
                    return path + [neighbor]
            return None
        
        # Pruning: check if unvisited vertices can still be reached
        unvisited = set(range(n)) - visited - {0}
        if not unvisited:
            return None
        
        # Try neighbors in order of fewest unvisited connections (MRV heuristic)
        candidates = []
        for neighbor in adj.get(current, set()):
            if neighbor in visited or neighbor == 0:
                continue
            if (min(current, neighbor), max(current, neighbor)) in forbidden_edges:
                continue
            # Count how many unvisited neighbors this neighbor has
            deg = sum(1 for nb in adj.get(neighbor, set()) if nb in unvisited and nb != 0)
            candidates.append((deg, neighbor))
        
        candidates.sort()
        
        for _, neighbor in candidates:
            new_remaining = remaining - 1
            result = dfs(path + [neighbor], visited | {neighbor}, neighbor, new_remaining)
            if result is not None:
                return result
        return None
    
    return dfs([0], {0}, 0, n-1)

def _find_ham_cycle_random(adj, n, forbidden_edges, deadline=None, rng=None):
    """Find a Hamiltonian cycle with randomized neighbor ordering."""
    if rng is None:
        rng = random.Random()
    
    def dfs(path, visited, current):
        if deadline is not None and time.time() > deadline:
            return None
        if len(path) == n:
            if current in adj.get(0, set()) and (0, current) not in forbidden_edges and (current, 0) not in forbidden_edges:
                return path
            return None
        
        if len(path) == n - 1:
            for neighbor in adj.get(current, set()):
                if neighbor not in visited:
                    if neighbor == 0:
                        continue
                    if (min(current, neighbor), max(current, neighbor)) in forbidden_edges:
                        continue
                    return path + [neighbor]
            return None
        
        neighbors = list(adj.get(current, set()))
        rng.shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor in visited or neighbor == 0:
                continue
            if (min(current, neighbor), max(current, neighbor)) in forbidden_edges:
                continue
            result = dfs(path + [neighbor], visited | {neighbor}, neighbor)
            if result is not None:
                return result
        return None
    
    return dfs([0], {0}, 0)

def _remove_cycle_edges(edges, cycle, n):
    """Remove edges used by the cycle from the edge list."""
    cycle_edges = set()
    for i in range(len(cycle)):
        u = cycle[i]
        v = cycle[(i + 1) % len(cycle)]
        cycle_edges.add((min(u, v), max(u, v)))
    
    remaining = []
    for u, v in edges:
        key = (min(u, v), max(u, v))
        if key not in cycle_edges:
            remaining.append([u, v])
    return remaining

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    n = P['n']
    edges = [list(e) for e in P['edges']]
    
    # First cycle: try natural ordering first
    first_cycle = list(range(n))
    adj0 = _build_adjacency(edges, n)
    valid = True
    for i in range(n):
        u = first_cycle[i]
        v = first_cycle[(i + 1) % n]
        if v not in adj0.get(u, set()):
            valid = False
            break
    if not valid:
        first_cycle = _find_ham_cycle_backtrack(adj0, n, set(), deadline)
        if first_cycle is None:
            return []
    
    cycles = [first_cycle]
    forbidden = _build_edge_set([edges[i] for i in range(len(edges))])
    
    # Remove first cycle's edges from forbidden set
    for i in range(len(first_cycle)):
        u = first_cycle[i]
        v = first_cycle[(i + 1) % len(first_cycle)]
        forbidden.discard((min(u, v), max(u, v)))
    
    remaining_edges = _remove_cycle_edges(edges, first_cycle, n)
    
    # Find second cycle with multiple strategies
    second_cycle = None
    
    # Strategy 1: Deterministic backtracking with MRV
    if time.time() < deadline - 5:
        adj1 = _build_adjacency(remaining_edges, n)
        second_cycle = _find_ham_cycle_backtrack(adj1, n, forbidden, deadline)
    
    # Strategy 2: Randomized restarts if deterministic failed
    if second_cycle is None and time.time() < deadline - 3:
        adj1 = _build_adjacency(remaining_edges, n)
        rng = random.Random(42)
        for attempt in range(200):
            if time.time() > deadline - 3:
                break
            second_cycle = _find_ham_cycle_random(adj1, n, forbidden, deadline, rng)
            if second_cycle is not None:
                break
    
    if second_cycle is not None:
        cycles.append(second_cycle)
        
        # Update forbidden edges
        for i in range(len(second_cycle)):
            u = second_cycle[i]
            v = second_cycle[(i + 1) % len(second_cycle)]
            forbidden.discard((min(u, v), max(u, v)))
        
        remaining_edges = _remove_cycle_edges(remaining_edges, second_cycle, n)
        
        # Find third cycle
        third_cycle = None
        
        if time.time() < deadline - 2:
            adj2 = _build_adjacency(remaining_edges, n)
            third_cycle = _find_ham_cycle_backtrack(adj2, n, forbidden, deadline)
        
        if third_cycle is None and time.time() < deadline - 1:
            adj2 = _build_adjacency(remaining_edges, n)
            rng2 = random.Random(123)
            for attempt in range(100):
                if time.time() > deadline - 1:
                    break
                third_cycle = _find_ham_cycle_random(adj2, n, forbidden, deadline, rng2)
                if third_cycle is not None:
                    break
        
        if third_cycle is not None:
            cycles.append(third_cycle)
    
    # Convert to proper types
    result = []
    for cycle in cycles:
        result.append([int(v) for v in cycle])
    
    return result
# EVOLVE-BLOCK-END
