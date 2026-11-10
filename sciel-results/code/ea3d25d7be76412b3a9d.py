"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _build_adj(edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_cycles_bfs(adj, remaining, max_cycles=None, deadline=None):
    """Find edge-disjoint cycles using BFS-based approach."""
    cycles = []
    remaining_set = set(remaining)
    adj_rem = _build_adj(remaining_set)
    
    while remaining_set:
        if deadline and time.time() > deadline:
            break
        if max_cycles and len(cycles) >= max_cycles:
            break
        
        # Find a vertex with degree >= 2
        candidates = [v for v in adj_rem if len(adj_rem[v]) >= 2]
        if not candidates:
            break
        
        # Pick the vertex with highest degree
        start = max(candidates, key=lambda v: len(adj_rem[v]))
        
        # BFS to find a cycle
        cycle = _bfs_cycle(start, adj_rem, remaining_set)
        if cycle and len(cycle) >= 3:
            cycles.append(cycle)
            # Remove edges from remaining
            for i in range(len(cycle)):
                a, b = cycle[i], cycle[(i + 1) % len(cycle)]
                e = tuple(sorted((a, b)))
                remaining_set.discard(e)
                adj_rem[a].discard(b)
                adj_rem[b].discard(a)
        else:
            # No more cycles can be found
            break
    
    return cycles, remaining_set

def _bfs_cycle(start, adj, remaining):
    """Find a cycle starting from start using BFS."""
    # Use BFS to find shortest cycle through start
    # Actually, we want ANY cycle, so let's use a simple approach
    # Find a neighbor, then try to find a path back to start
    
    # Try each neighbor of start
    for nb in list(adj[start]):
        if nb == start:
            continue
        e = tuple(sorted((start, nb)))
        if e not in remaining:
            continue
        
        # BFS from nb to start, avoiding the direct edge start-nb
        visited = {start: None, nb: start}
        queue = deque([nb])
        
        while queue:
            node = queue.popleft()
            if node == start:
                # Reconstruct path
                path = [node]
                while visited[node] is not None:
                    node = visited[node]
                    path.append(node)
                path.reverse()
                if len(path) >= 3:
                    return path
                continue
            
            for nb2 in adj[node]:
                if nb2 in visited:
                    continue
                e2 = tuple(sorted((node, nb2)))
                if e2 not in remaining:
                    continue
                if node == start and nb2 == nb:
                    continue  # Avoid the direct edge back
                visited[nb2] = node
                queue.append(nb2)
    
    return None

def _find_cycles_dfs(adj, remaining, max_depth=15, deadline=None):
    """Find edge-disjoint cycles using DFS with backtracking."""
    cycles = []
    remaining_set = set(remaining)
    adj_rem = _build_adj(remaining_set)
    
    while remaining_set:
        if deadline and time.time() > deadline:
            break
        
        # Find a vertex with degree >= 2
        candidates = [v for v in adj_rem if len(adj_rem[v]) >= 2]
        if not candidates:
            break
        
        # Pick a random candidate
        random.shuffle(candidates)
        start = candidates[0]
        
        # DFS to find a cycle
        cycle = _dfs_cycle(start, adj_rem, remaining_set, max_depth)
        if cycle and len(cycle) >= 3:
            cycles.append(cycle)
            for i in range(len(cycle)):
                a, b = cycle[i], cycle[(i + 1) % len(cycle)]
                e = tuple(sorted((a, b)))
                remaining_set.discard(e)
                adj_rem[a].discard(b)
                adj_rem[b].discard(a)
        else:
            # Try other candidates
            found = False
            for s in candidates[1:]:
                if deadline and time.time() > deadline:
                    break
                cycle = _dfs_cycle(s, adj_rem, remaining_set, max_depth)
                if cycle and len(cycle) >= 3:
                    cycles.append(cycle)
                    for i in range(len(cycle)):
                        a, b = cycle[i], cycle[(i + 1) % len(cycle)]
                        e = tuple(sorted((a, b)))
                        remaining_set.discard(e)
                        adj_rem[a].discard(b)
                        adj_rem[b].discard(a)
                    found = True
                    break
            if not found:
                break
    
    return cycles, remaining_set

def _dfs_cycle(start, adj, remaining, max_depth=15):
    """Find a cycle starting from start using DFS."""
    path = [start]
    visited = {start}
    
    def dfs(node, depth):
        if depth >= max_depth:
            return None
        if node == start and len(path) >= 3:
            return path[:]
        
        for nb in adj[node]:
            if nb == start:
                if len(path) >= 3:
                    return path[:] + [start]
                continue
            if nb in visited:
                continue
            e = tuple(sorted((node, nb)))
            if e not in remaining:
                continue
            
            visited.add(nb)
            path.append(nb)
            result = dfs(nb, depth + 1)
            if result:
                return result
            path.pop()
            visited.discard(nb)
        
        return None
    
    return dfs(start, 1)

def _decompose_graph(edges, deadline):
    """Decompose edges into cycles and single edges."""
    if not edges:
        return []
    
    edges_set = set(tuple(sorted(e)) for e in edges)
    
    # Phase 1: Use BFS to find many cycles quickly
    cycles1, remaining1 = _find_cycles_bfs(_build_adj(edges_set), edges_set, deadline=deadline)
    
    # Phase 2: Use DFS to find more cycles in remaining edges
    if remaining1:
        cycles2, remaining2 = _find_cycles_dfs(_build_adj(remaining1), remaining1, max_depth=12, deadline=deadline)
    else:
        cycles2, remaining2 = [], set()
    
    # Phase 3: Add remaining edges as single-edge pieces
    pieces = cycles1 + cycles2
    for e in remaining2:
        pieces.append([e[0], e[1]])
    
    return pieces

def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    start_time = time.time()
    # Evaluator kills at 150s, leave 10s margin
    total_deadline = start_time + 140.0
    
    for i, spec in enumerate(specs):
        # Allocate time proportionally
        elapsed = time.time() - start_time
        remaining_time = total_deadline - elapsed
        per_graph_time = remaining_time / (len(specs) - i)
        graph_deadline = time.time() + per_graph_time * 0.95  # Leave 5% margin
        
        edges = _instance(*spec)
        pieces = _decompose_graph(edges, graph_deadline)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
