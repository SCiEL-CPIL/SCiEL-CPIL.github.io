# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj(n, edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_ham_cycle_smart(vertices, adj, used_edges, start=None, deadline=None, rng=None):
    """
    Find a Hamiltonian cycle on the given vertex set using only unused edges.
    Uses backtracking with constraint propagation (forced moves) and Warnsdorff's rule.
    """
    k = len(vertices)
    if k < 3:
        return None
    vset = frozenset(vertices)
    if start is None:
        start = min(vset)
    if rng is None:
        rng = random
    
    # Pre-compute available neighbors for each vertex given used_edges
    avail = defaultdict(set)
    for u in vertices:
        for v in adj[u]:
            if v in vset:
                e = (min(u, v), max(u, v))
                if e not in used_edges:
                    avail[u].add(v)
    
    # Check if start has at least 2 available neighbors
    if len(avail[start]) < 2:
        return None

    path = [start]
    visited = {start}
    node_count = 0

    def propagate_forced_moves():
        """
        Apply constraint propagation: if a vertex has only one available unvisited neighbor,
        it must go there. This can cascade.
        Returns True if consistent, False if contradiction.
        """
        changed = True
        while changed:
            changed = False
            # For each unvisited vertex, check if it has exactly one available unvisited neighbor
            # (or if it's the last vertex, it needs exactly start)
            for v in list(vset - visited):
                if v == start:
                    continue  # start is already visited
                unvisited_neighbors = [n for n in avail[v] if n not in visited]
                if len(unvisited_neighbors) == 0:
                    # Dead end unless this is the last vertex
                    if len(path) == k - 1:
                        if start in avail[v]:
                            # This must be the last vertex
                            pass
                        else:
                            return False
                    else:
                        return False
                elif len(unvisited_neighbors) == 1:
                    # Forced move: v must connect to unvisited_neighbors[0]
                    forced_target = unvisited_neighbors[0]
                    # Check if this creates a contradiction
                    if forced_target == start and len(path) != k - 1:
                        return False
                    if forced_target in visited and forced_target != start:
                        return False
                    # We'll handle this by adding to a queue of forced moves
                    # For simplicity, we just note this constraint
                    # In a full implementation, we'd queue these
                    pass
        return True

    def get_candidates(current):
        candidates = []
        for nxt in avail[current]:
            if nxt not in visited:
                # Warnsdorff's rule: prefer nodes with fewer available exits
                if len(path) == k - 1:
                    if start in avail[nxt]:
                        candidates.append((0, nxt))
                else:
                    unvisited_neighbors = [n for n in avail[nxt] if n not in visited]
                    if len(unvisited_neighbors) == 0:
                        # Dead end, unless this is the last node
                        if len(path) == k - 1:
                            if start in avail[nxt]:
                                candidates.append((0, nxt))
                        else:
                            continue
                    else:
                        # Penalize vertices that are forced (only 1 unvisited neighbor)
                        # to avoid creating future dead ends
                        penalty = 0
                        if len(unvisited_neighbors) == 1:
                            penalty = 10  # High penalty for forced moves
                        candidates.append((len(unvisited_neighbors) + penalty, nxt))
        
        if candidates:
            if rng:
                scored = [(h, rng.random(), n) for h, n in candidates]
                scored.sort()
                return [n for _, _, n in scored]
            else:
                candidates.sort(key=lambda x: x[0])
                return [n for _, n in candidates]
        return []

    def dfs(current):
        nonlocal node_count
        node_count += 1
        if deadline is not None and node_count % 1000 == 0 and time.time() > deadline:
            return None
        
        if len(path) == k:
            if start in avail[current]:
                return list(path)
            return None
        
        # Try to apply constraint propagation before branching
        # Check for vertices that are forced
        forced_moves = []
        for v in vset - visited:
            if v == start:
                continue
            unvisited_neighbors = [n for n in avail[v] if n not in visited]
            if len(unvisited_neighbors) == 1:
                forced_moves.append((v, unvisited_neighbors[0]))
        
        # If there are forced moves, try to apply them
        if forced_moves:
            # Find a forced move that involves the current vertex or can be applied
            for v, target in forced_moves:
                if v == current or target == current:
                    # This forced move involves current
                    if target not in visited:
                        visited.add(target)
                        path.append(target)
                        result = dfs(target)
                        if result is not None:
                            return result
                        path.pop()
                        visited.remove(target)
                    break
        
        for nxt in get_candidates(current):
            visited.add(nxt)
            path.append(nxt)
            result = dfs(nxt)
            if result is not None:
                return result
            path.pop()
            visited.remove(nxt)
        return None
    
    return dfs(start)

def _find_cycles_for_vertex_set(vertices, adj, time_limit, max_cycles=10, seed=42):
    """Find max number of edge-disjoint cycles on the given vertex set."""
    vertices = list(vertices)
    used_edges = set()
    cycles = []
    start_time = time.time()
    deadline = start_time + time_limit
    rng = random.Random(seed)
    
    for _ in range(max_cycles):
        if time.time() > deadline - 0.1:
            break
        
        # Randomize the starting vertex to find different cycles
        start = rng.choice(vertices)
        
        cycle = _find_ham_cycle_smart(vertices, adj, used_edges, start=start, deadline=deadline, rng=rng)
        if cycle is None:
            # Try a few more random starts before giving up
            found = False
            for _retry in range(8):
                if time.time() > deadline - 0.1:
                    break
                start = rng.choice(vertices)
                cycle = _find_ham_cycle_smart(vertices, adj, used_edges, start=start, deadline=deadline, rng=rng)
                if cycle is not None:
                    found = True
                    break
            if not found:
                break
        
        valid = True
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i+1) % len(cycle)]
            e = (min(u, v), max(u, v))
            if e in used_edges:
                valid = False
                break
            used_edges.add(e)
        
        if valid:
            cycles.append(cycle)
    
    return cycles

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin before 90s kill
    
    n = P['n']
    edges = P['edges']
    all_edges_set = set()
    for u, v in edges:
        all_edges_set.add((min(u, v), max(u, v)))
    
    adj = _build_adj(n, edges)
    full_vertices = list(range(n))
    
    best_cycles = []
    best_count = 0
    
    # Try multiple random seeds
    for seed in [42, 123, 7, 999, 555, 1, 2, 3, 11, 22, 33, 44, 55, 66, 77, 88, 99, 100]:
        if time.time() > deadline - 2:
            break
        
        cycles = _find_cycles_for_vertex_set(full_vertices, adj, deadline - time.time(), max_cycles=10, seed=seed)
        
        if len(cycles) > best_count:
            best_count = len(cycles)
            best_cycles = cycles
        
        if best_count >= 10:
            break
    
    # If we didn't find many, try a few more seeds or different strategies
    if best_count < 5:
        for seed in range(200, 300):
            if time.time() > deadline - 1:
                break
            cycles = _find_cycles_for_vertex_set(full_vertices, adj, deadline - time.time(), max_cycles=10, seed=seed)
            if len(cycles) > best_count:
                best_count = len(cycles)
                best_cycles = cycles
            if best_count >= 10:
                break

    # Ensure best_cycles is non-empty
    if not best_cycles:
        cycles = _find_cycles_for_vertex_set(full_vertices, adj, deadline - time.time(), max_cycles=10)
        if cycles:
            best_cycles = cycles

    # Convert to proper types (list of tuples)
    result = []
    for cycle in best_cycles[:10]:
        result.append(tuple(cycle))
    
    return result

# EVOLVE-BLOCK-END
