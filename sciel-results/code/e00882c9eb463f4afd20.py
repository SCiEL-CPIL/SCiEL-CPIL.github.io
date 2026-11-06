"""Decomposition of random graphs into few cycles and edges via iterative 2-factor extraction."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict


def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _find_cycles_in_2factor(adj, n):
    """Given a 2-regular graph (adjacency as dict of sets), extract all cycles."""
    visited = set()
    cycles = []
    for start in range(n):
        if start in visited or start not in adj or not adj[start]:
            continue
        # Walk the cycle
        cycle = [start]
        visited.add(start)
        current = start
        while True:
            neighbors = list(adj[current])
            # Pick the next vertex (not the previous one, unless it's a 2-cycle which shouldn't happen)
            prev = cycle[-2] if len(cycle) >= 2 else None
            nxt = None
            for nb in neighbors:
                if nb != prev:
                    nxt = nb
                    break
            if nxt is None:
                # Only one neighbor left, must be the previous one (shouldn't happen in 2-factor unless len=2)
                nxt = neighbors[0]
            
            if nxt == start:
                break
            if nxt in visited:
                # Should not happen in a proper 2-factor unless we entered a different cycle, 
                # but since we remove visited nodes, this implies we closed the loop incorrectly or hit a visited node.
                # In a 2-factor, components are disjoint cycles. If we hit a visited node that isn't start, 
                # it means our adjacency was not a simple 2-factor or we made an error. 
                # Let's just break to be safe.
                break
            
            cycle.append(nxt)
            visited.add(nxt)
            current = nxt
        
        if len(cycle) >= 3:
            cycles.append(cycle)
        elif len(cycle) == 2:
            # 2-cycle (double edge) shouldn't exist in simple graph 2-factor
            pass
    return cycles


def _extract_2factor(edges, n):
    """
    Try to find a large 2-factor (collection of edge-disjoint cycles covering as many edges as possible).
    We use a greedy approach: repeatedly find a cycle, remove its edges, and repeat.
    To maximize the number of edges covered, we prioritize longer cycles or cycles that use high-degree vertices.
    However, a simpler robust method for small graphs is to use a matching-based approach or just greedy longest cycle.
    Given the time constraint and graph size (n<=30), we can afford a bit more search.
    
    Strategy: 
    1. Build remaining edge set.
    2. Repeatedly find the longest cycle in the remaining graph.
    3. Remove its edges.
    4. Repeat until no cycles remain.
    
    This is similar to the parent but potentially more effective if we use a better cycle finder.
    Let's try a randomized DFS that specifically looks for long cycles.
    """
    rem = set(edges)
    rem_adj = defaultdict(set)
    for u, v in rem:
        rem_adj[u].add(v)
        rem_adj[v].add(u)
    
    cycles = []
    
    # We will try to extract cycles greedily.
    # To avoid getting stuck in short cycles, we can use a heuristic: 
    # At each step, find the longest cycle in the current remaining graph.
    # Since n is small (<=30), we can use a DFS that explores all paths up to a certain length or uses backtracking to find the longest cycle.
    
    while rem:
        # Find the longest cycle in the current graph
        best_cycle = None
        best_len = 0
        
        # Try starting from each vertex with degree >= 2
        candidates = [v for v in rem_adj if len(rem_adj[v]) >= 2]
        if not candidates:
            break
        
        # Sort candidates by degree descending to find long cycles faster
        candidates.sort(key=lambda v: -len(rem_adj[v]))
        
        for start in candidates:
            if not rem:
                break
            # DFS to find the longest cycle starting and ending at start
            # We use a recursive DFS with pruning.
            # Since n is small, we can allow some depth.
            
            # To speed up, we can limit the search or use a heuristic.
            # Let's try a randomized restart approach for the DFS.
            
            found = False
            for _ in range(5):  # 5 random restarts per start vertex
                if not rem:
                    break
                # Shuffle neighbors to randomize path
                neighbor_order = {v: list(rem_adj[v]) for v in rem_adj}
                for v in neighbor_order:
                    random.shuffle(neighbor_order[v])
                
                path = [start]
                visited = {start}
                
                def dfs(current, depth):
                    nonlocal best_cycle, best_len, found
                    if found:
                        return
                    if depth > 2:
                        if start in rem_adj[current]:
                            cycle = path + [start]
                            if len(cycle) - 1 > best_len:
                                best_len = len(cycle) - 1
                                best_cycle = cycle
                                found = True
                            return
                    for nb in neighbor_order.get(current, []):
                        if nb == start:
                            if depth > 2:
                                cycle = path + [start]
                                if len(cycle) - 1 > best_len:
                                    best_len = len(cycle) - 1
                                    best_cycle = cycle
                                    found = True
                                return
                        elif nb not in visited:
                            visited.add(nb)
                            path.append(nb)
                            dfs(nb, depth + 1)
                            path.pop()
                            visited.discard(nb)
                            if found:
                                return
                
                dfs(start, 1)
                if found:
                    break
            
            if found:
                break
        
        if best_cycle and best_len >= 3:
            # Remove edges of best_cycle
            for i in range(len(best_cycle) - 1):
                u, v = best_cycle[i], best_cycle[i + 1]
                e = (u, v) if u < v else (v, u)
                if e in rem:
                    rem.discard(e)
                    rem_adj[u].discard(v)
                    rem_adj[v].discard(u)
                    if not rem_adj[u]:
                        del rem_adj[u]
                    if not rem_adj[v]:
                        del rem_adj[v]
            cycles.append(best_cycle[:-1])
        else:
            # No more cycles found
            break
    
    return cycles, rem


def _decompose(edges, deadline, rng):
    """Decompose edge set into cycles and single edges, minimizing piece count."""
    n = max(max(u, v) for u, v in edges) + 1 if edges else 0
    rem = set(edges)
    rem_adj = defaultdict(set)
    for u, v in rem:
        rem_adj[u].add(v)
        rem_adj[v].add(u)
    
    pieces = []
    
    # Iteratively extract cycles
    # We use a greedy approach: at each step, find a cycle that maximizes the number of edges covered.
    # To be more effective, we can try to find the longest cycle each time.
    
    while rem and time.time() < deadline:
        # Find the longest cycle in the remaining graph
        best_cycle = None
        best_len = 0
        
        # Try starting from each vertex with degree >= 2
        candidates = [v for v in rem_adj if len(rem_adj[v]) >= 2]
        if not candidates:
            break
        
        # Sort candidates by degree descending
        candidates.sort(key=lambda v: -len(rem_adj[v]))
        
        for start in candidates:
            if not rem:
                break
            
            # Use a DFS to find a long cycle starting at start
            # We'll do a limited DFS with backtracking
            # To avoid exponential blowup, we limit the depth or use heuristics.
            # Since n is small, we can try a few random orders.
            
            found = False
            for attempt in range(10):  # 10 attempts per start vertex
                if not rem:
                    break
                # Create a copy of adjacency with shuffled neighbors
                adj_copy = {v: list(rem_adj[v]) for v in rem_adj}
                for v in adj_copy:
                    rng.shuffle(adj_copy[v])
                
                path = [start]
                visited = {start}
                
                def dfs(current, depth):
                    nonlocal best_cycle, best_len, found
                    if found:
                        return
                    if depth > 2:
                        if start in rem_adj[current]:
                            cycle = path + [start]
                            if len(cycle) - 1 > best_len:
                                best_len = len(cycle) - 1
                                best_cycle = cycle
                                found = True
                            return
                    for nb in adj_copy.get(current, []):
                        if nb == start:
                            if depth > 2:
                                cycle = path + [start]
                                if len(cycle) - 1 > best_len:
                                    best_len = len(cycle) - 1
                                    best_cycle = cycle
                                    found = True
                                return
                        elif nb not in visited:
                            visited.add(nb)
                            path.append(nb)
                            dfs(nb, depth + 1)
                            path.pop()
                            visited.discard(nb)
                            if found:
                                return
                
                dfs(start, 1)
                if found:
                    break
            
            if found:
                break
        
        if best_cycle and best_len >= 3:
            # Remove edges of best_cycle
            for i in range(len(best_cycle) - 1):
                u, v = best_cycle[i], best_cycle[i + 1]
                e = (u, v) if u < v else (v, u)
                if e in rem:
                    rem.discard(e)
                    rem_adj[u].discard(v)
                    rem_adj[v].discard(u)
                    if not rem_adj[u]:
                        del rem_adj[u]
                    if not rem_adj[v]:
                        del rem_adj[v]
            pieces.append(best_cycle[:-1])
        else:
            # No more cycles, take a single edge
            if not rem:
                break
            u, v = next(iter(rem))
            rem.discard((u, v))
            rem_adj[u].discard(v)
            rem_adj[v].discard(u)
            if not rem_adj[u]:
                del rem_adj[u]
            if not rem_adj[v]:
                del rem_adj[v]
            pieces.append([u, v])
    
    # Add remaining edges as single-edge pieces
    for u, v in rem:
        pieces.append([u, v])
    
    return pieces


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = _instance(*spec)
        rng = random.Random(42)
        pieces = _decompose(edges, deadline, rng)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
