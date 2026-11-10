import time
import random
from collections import defaultdict, deque
from random import Random

def construct():
    start_time = time.time()
    TIME_LIMIT = 145.0  # Safety margin before 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def _build_adj(edges):
        adj = defaultdict(set)
        for a, b in edges:
            adj[a].add(b)
            adj[b].add(a)
        return adj

    def _find_cycle(adj, n):
        """Find a cycle in the graph using DFS. Returns list of vertices forming the cycle, or None."""
        color = [0] * n  # 0=white, 1=gray, 2=black
        parent = [-1] * n
        cycle = []

        def dfs(u):
            color[u] = 1
            for v in adj[u]:
                if color[v] == 1:
                    # Found a back edge u->v, extract cycle
                    # Cycle is v, ..., u, v
                    path = [v]
                    cur = u
                    while cur != v:
                        path.append(cur)
                        cur = parent[cur]
                    path.reverse()
                    cycle.extend(path)
                    return True
                elif color[v] == 0:
                    parent[v] = u
                    if dfs(v):
                        return True
            color[u] = 2
            return False

        for i in range(n):
            if color[i] == 0:
                if dfs(i):
                    return cycle
        return None

    def _find_longest_cycle(adj, n, max_length=None):
        """Try to find a reasonably long cycle using DFS with backtracking."""
        # Simple approach: find any cycle, then try to extend it
        cycle = _find_cycle(adj, n)
        if cycle is None:
            return None
        return cycle

    def _decompose_greedy_with_cycles(edges, n, seed):
        """
        Decompose by repeatedly finding cycles and removing them.
        Uses a priority: prefer longer cycles to reduce piece count.
        """
        rng = Random(seed)
        remaining = set(edges)
        pieces = []

        while remaining:
            if time.time() - start_time > TIME_LIMIT - 2:
                # Emergency: dump remaining as single edges
                for e in remaining:
                    pieces.append(list(e))
                return pieces

            adj = _build_adj(remaining)
            
            # Try to find a cycle
            # To encourage longer cycles, we can try multiple random starts
            best_cycle = None
            best_len = 0
            
            # Try a few random orderings of vertices to find longer cycles
            for attempt in range(5):
                if time.time() - start_time > TIME_LIMIT - 2:
                    break
                # Shuffle adjacency lists for diversity
                adj_shuffled = {u: list(v) for u, v in adj.items()}
                for u in adj_shuffled:
                    rng.shuffle(adj_shuffled[u])
                
                cycle = _find_cycle(adj_shuffled, n)
                if cycle and len(cycle) > best_len:
                    best_len = len(cycle)
                    best_cycle = cycle
            
            if best_cycle:
                # Remove edges of the cycle
                for i in range(len(best_cycle)):
                    a = best_cycle[i]
                    b = best_cycle[(i + 1) % len(best_cycle)]
                    e = tuple(sorted((a, b)))
                    if e in remaining:
                        remaining.remove(e)
                pieces.append(best_cycle)
            else:
                # No cycle found; pick a random edge
                e = tuple(sorted(rng.choice(list(remaining))))
                remaining.remove(e)
                pieces.append(list(e))

        return pieces

    def _decompose_eulerian(edges, n, seed):
        """
        Strategy: find the largest 2-edge-connected component,
        decompose it using Eulerian circuits to maximize cycles,
        then handle remaining edges.
        
        A key insight: in a 2-edge-connected graph, we can find an Eulerian circuit
        after adding dummy edges, which gives us a long cycle. More generally,
        we can repeatedly find cycles in 2-edge-connected components.
        
        Simpler robust approach: 
        1. Repeatedly find a cycle, remove it.
        2. To get longer cycles, when we find a cycle, check if we can extend it.
        """
        rng = Random(seed)
        remaining = set(edges)
        pieces = []

        while remaining:
            if time.time() - start_time > TIME_LIMIT - 2:
                for e in remaining:
                    pieces.append(list(e))
                return pieces

            adj = _build_adj(remaining)
            
            # Find a cycle
            cycle = _find_cycle(adj, n)
            
            if cycle:
                # Try to extend the cycle by checking if any edge in the cycle
                # can be "shortcut" to include more vertices
                # Actually, a simpler heuristic: after finding a cycle, 
                # check if there's a path between two non-adjacent vertices in the cycle
                # that doesn't use cycle edges, to make a longer cycle.
                
                # For now, just use the found cycle
                for i in range(len(cycle)):
                    a = cycle[i]
                    b = cycle[(i + 1) % len(cycle)]
                    e = tuple(sorted((a, b)))
                    if e in remaining:
                        remaining.remove(e)
                pieces.append(cycle)
            else:
                # No cycle; must be a forest
                e = tuple(sorted(rng.choice(list(remaining))))
                remaining.remove(e)
                pieces.append(list(e))

        return pieces

    def _decompose_2ecc_based(edges, n, seed):
        """
        More sophisticated: 
        1. Find all bridges, remove them (they must be single-edge pieces).
        2. Work on the remaining 2-edge-connected components.
        3. In each 2-edge-connected component, repeatedly find cycles.
        """
        rng = Random(seed)
        remaining = set(edges)
        pieces = []
        
        # First, identify and remove bridges
        def find_bridges(edge_set, n):
            adj = _build_adj(edge_set)
            disc = [-1] * n
            low = [-1] * n
            parent = [-1] * n
            time_counter = [0]
            bridges = set()
            
            def dfs_bridge(u):
                disc[u] = low[u] = time_counter[0]
                time_counter[0] += 1
                for v in adj[u]:
                    if disc[v] == -1:
                        parent[v] = u
                        dfs_bridge(v)
                        low[u] = min(low[u], low[v])
                        if low[v] > disc[u]:
                            bridges.add(tuple(sorted((u, v))))
                    elif v != parent[u]:
                        low[u] = min(low[u], disc[v])
            
            for i in range(n):
                if disc[i] == -1:
                    dfs_bridge(i)
            
            return bridges

        # Iteratively remove bridges
        changed = True
        while changed:
            if time.time() - start_time > TIME_LIMIT - 2:
                break
            changed = False
            if not remaining:
                break
            bridges = find_bridges(remaining, n)
            if bridges:
                for b in bridges:
                    if b in remaining:
                        remaining.remove(b)
                        pieces.append(list(b))
                changed = True

        # Now remaining should have no bridges (or be empty)
        # Repeatedly find cycles
        while remaining:
            if time.time() - start_time > TIME_LIMIT - 2:
                for e in remaining:
                    pieces.append(list(e))
                return pieces

            adj = _build_adj(remaining)
            cycle = _find_cycle(adj, n)
            
            if cycle:
                for i in range(len(cycle)):
                    a = cycle[i]
                    b = cycle[(i + 1) % len(cycle)]
                    e = tuple(sorted((a, b)))
                    if e in remaining:
                        remaining.remove(e)
                pieces.append(cycle)
            else:
                # Shouldn't happen if no bridges, but fallback
                e = tuple(sorted(rng.choice(list(remaining))))
                remaining.remove(e)
                pieces.append(list(e))

        return pieces

    def _decompose_with_cycle_packing(edges, n, seed):
        """
        Combined strategy:
        1. Remove bridges first.
        2. In the remaining graph, use a smarter cycle-finding that tries to find longer cycles.
        3. Use multiple restarts with different random seeds.
        """
        rng = Random(seed)
        remaining = set(edges)
        pieces = []
        
        # Step 1: Remove bridges
        def find_bridges(edge_set, n):
            adj = _build_adj(edge_set)
            disc = [-1] * n
            low = [-1] * n
            parent = [-1] * n
            time_counter = [0]
            bridges = set()
            
            def dfs_bridge(u):
                disc[u] = low[u] = time_counter[0]
                time_counter[0] += 1
                for v in adj[u]:
                    if disc[v] == -1:
                        parent[v] = u
                        dfs_bridge(v)
                        low[u] = min(low[u], low[v])
                        if low[v] > disc[u]:
                            bridges.add(tuple(sorted((u, v))))
                    elif v != parent[u]:
                        low[u] = min(low[u], disc[v])
            
            for i in range(n):
                if disc[i] == -1:
                    dfs_bridge(i)
            
            return bridges

        changed = True
        while changed and time.time() - start_time < TIME_LIMIT - 2:
            changed = False
            if not remaining:
                break
            bridges = find_bridges(remaining, n)
            if bridges:
                for b in bridges:
                    if b in remaining:
                        remaining.remove(b)
                        pieces.append(list(b))
                changed = True

        # Step 2: Find cycles in remaining
        while remaining and time.time() - start_time < TIME_LIMIT - 2:
            adj = _build_adj(remaining)
            
            # Try to find a longer cycle by trying multiple DFS starts
            best_cycle = None
            best_len = 0
            
            # Try from each vertex as start
            for start in range(n):
                if not any(start in adj[u] for u in adj):
                    continue
                # Temporarily modify adj to start from this vertex
                adj_copy = {u: list(v) for u, v in adj.items()}
                for u in adj_copy:
                    rng.shuffle(adj_copy[u])
                
                cycle = _find_cycle(adj_copy, n)
                if cycle and len(cycle) > best_len:
                    best_len = len(cycle)
                    best_cycle = cycle
                
                if time.time() - start_time > TIME_LIMIT - 2:
                    break
            
            if best_cycle:
                for i in range(len(best_cycle)):
                    a = best_cycle[i]
                    b = best_cycle[(i + 1) % len(best_cycle)]
                    e = tuple(sorted((a, b)))
                    if e in remaining:
                        remaining.remove(e)
                pieces.append(best_cycle)
            else:
                e = tuple(sorted(rng.choice(list(remaining))))
                remaining.remove(e)
                pieces.append(list(e))

        return pieces

    # Main: try multiple strategies and restarts, keep the best
    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))

    for spec in specs:
        edges = instance(*spec)
        n = spec[0]
        best_pieces = None
        best_count = float('inf')

        # Try the bridge-removal + cycle packing strategy with multiple seeds
        for restart_seed in range(20):
            if time.time() - start_time > TIME_LIMIT - 3:
                break
            try:
                pieces = _decompose_with_cycle_packing(edges, n, 5000 + restart_seed)
                count = len(pieces)
                if count < best_count:
                    best_count = count
                    best_pieces = pieces
            except Exception:
                continue

        # Also try the simpler 2ecc-based approach
        for restart_seed in range(10):
            if time.time() - start_time > TIME_LIMIT - 3:
                break
            try:
                pieces = _decompose_2ecc_based(edges, n, 6000 + restart_seed)
                count = len(pieces)
                if count < best_count:
                    best_count = count
                    best_pieces = pieces
            except Exception:
                continue

        if best_pieces is None:
            best_pieces = [list(e) for e in edges]

        results.append(best_pieces)

    return results
