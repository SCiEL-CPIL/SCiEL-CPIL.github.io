"""Decompose three fixed instance graphs into few cycles and edges (Erdős-Gallai task)."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

_DEADLINE = 150.0


def instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _cycle_edges(cyc):
    m = len(cyc)
    return {tuple(sorted((cyc[i], cyc[(i + 1) % m]))) for i in range(m)}


def _find_longest_cycle_bfs(adj, n):
    """
    Find a long cycle using a modified BFS/DFS approach.
    We look for a path from a start vertex back to itself with length >= 3.
    Strategy: 
    1. For each vertex s, do a BFS to find the furthest vertex t.
    2. Then try to find a path from s to t that is long, and check if it closes.
    Actually, a simpler and effective heuristic for small graphs (n<=30):
    Use DFS to find a simple path. We want the longest simple path that forms a cycle.
    
    Better approach for small n:
    Since n is small (20-30), we can afford a more exhaustive search with pruning.
    However, 30! is too big.
    
    Let's use a heuristic:
    1. Pick a start vertex s.
    2. Perform a DFS to find a simple path starting at s.
    3. We want to maximize the length of the path such that the last vertex is adjacent to s.
    
    We can do multiple DFS runs with different tie-breaking strategies.
    """
    best_cycle = None
    best_len = 0
    
    # Heuristic: try starting from each vertex, or just a few random ones
    # Since n is small, let's try a few starts.
    starts = list(range(n))
    random.shuffle(starts)
    # Limit starts to save time if n is large, but n<=30 so it's fine
    # Actually, let's just try all starts but limit the DFS depth/time
    
    for s in starts:
        if not adj[s]:
            continue
        
        # DFS to find a path from s back to s
        # State: path, visited
        # We want the longest path.
        # This is NP-hard in general, but for n=30, a good heuristic works.
        
        # Let's use a greedy approach with backtracking limited.
        # Or simpler: Randomized greedy path extension.
        
        # Let's try to build a path from s, and see if we can close it.
        # We prioritize neighbors that have high degree (to keep options open) or low degree (to finish off leaves)?
        # Usually, to make a cycle, you want to avoid getting stuck.
        
        # Let's try a specific heuristic:
        # 1. Start at s.
        # 2. At each step, pick a neighbor v not in path.
        #    Prefer v with smallest degree (Warnsdorff's rule for Hamiltonian cycles) to avoid isolating vertices.
        #    Or prefer v with largest degree to stay in the core?
        #    Let's try both and pick the best cycle found.
        
        def dfs_greedy(start, strategy):
            path = [start]
            visited = {start}
            
            while True:
                last = path[-1]
                neighbors = [v for v in adj[last] if v not in visited]
                
                if not neighbors:
                    break
                
                # Sort neighbors based on strategy
                if strategy == 0:
                    # Prefer low degree (Warnsdorff)
                    neighbors.sort(key=lambda v: len(adj[v]))
                else:
                    # Prefer high degree
                    neighbors.sort(key=lambda v: -len(adj[v]))
                
                # Try the first choice
                nxt = neighbors[0]
                path.append(nxt)
                visited.add(nxt)
                
                # If we can close the cycle now, and the path is long enough, we could stop?
                # No, we want the LONGEST cycle. So keep going if possible.
                
            # Check if path closes
            if len(path) >= 3 and path[0] in adj[path[-1]]:
                return path
            return None
        
        # Try both strategies
        for strat in [0, 1]:
            cyc = dfs_greedy(s, strat)
            if cyc and len(cyc) > best_len:
                best_len = len(cyc)
                best_cycle = cyc
                
    return best_cycle


def _find_cycle_karp(adj, n):
    """
    Use a DP-based approach to find the longest cycle?
    Too slow for n=30 (2^30 states).
    
    Let's stick to the greedy heuristic but improve it.
    """
    pass


def _decompose(edges, deadline):
    n = max((max(e) for e in edges), default=0) + 1
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    remaining = {tuple(sorted(e)) for e in edges}
    pieces = []
    t_start = time.time()
    
    # Main loop: find cycles and remove them
    # To minimize pieces, we want to find LONG cycles first.
    # A cycle of length k removes k edges and counts as 1 piece.
    # An edge removes 1 edge and counts as 1 piece.
    # So we want to maximize the sum of cycle lengths.
    
    # Strategy:
    # 1. Repeatedly find the longest cycle in the remaining graph.
    # 2. Remove its edges.
    # 3. Repeat until no cycles are left.
    # 4. Remaining edges are single pieces.
    
    # Optimization: If finding the longest cycle is too slow, use a greedy heuristic.
    # Given n <= 30, we can afford a somewhat expensive search per cycle.
    
    while remaining:
        if time.time() > deadline:
            break
            
        # Find a cycle in the remaining graph
        # We build the adjacency list for the remaining graph
        rem_adj = [set() for _ in range(n)]
        for u, v in remaining:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
        
        # Try to find a long cycle
        # We'll use a randomized restart approach with the greedy heuristic
        best_cyc = None
        best_len = 0
        
        # Number of restarts depends on time available
        # Let's do a few restarts
        rng = random.Random() # Local RNG for randomization
        
        # Estimate time per cycle search
        # Let's allow up to 0.1s per cycle search on average?
        # With 150s total, and maybe 50-100 cycles, we have ~1.5s per cycle.
        # But we want to be safe.
        
        # Let's try a fixed number of randomized greedy paths
        for _ in range(20):
            # Random start
            s = rng.randrange(n)
            if not rem_adj[s]:
                continue
            
            # Randomized greedy path
            path = [s]
            visited = {s}
            
            while True:
                last = path[-1]
                neighbors = [v for v in rem_adj[last] if v not in visited]
                if not neighbors:
                    break
                
                # Random choice, but maybe biased?
                # Let's just pick random
                nxt = rng.choice(neighbors)
                path.append(nxt)
                visited.add(nxt)
                
            if len(path) >= 3 and path[0] in rem_adj[path[-1]]:
                if len(path) > best_len:
                    best_len = len(path)
                    best_cyc = path[:]
            
            # Also try Warnsdorff-like on the remaining graph
            # Re-initialize
            path = [s]
            visited = {s}
            while True:
                last = path[-1]
                neighbors = [v for v in rem_adj[last] if v not in visited]
                if not neighbors:
                    break
                # Prefer low degree
                neighbors.sort(key=lambda v: len(rem_adj[v]))
                nxt = neighbors[0]
                path.append(nxt)
                visited.add(nxt)
                
            if len(path) >= 3 and path[0] in rem_adj[path[-1]]:
                if len(path) > best_len:
                    best_len = len(path)
                    best_cyc = path[:]
                    
        if best_cyc:
            # Remove edges of the cycle
            ce = _cycle_edges(best_cyc)
            # Verify all edges are in remaining
            if ce <= remaining:
                pieces.append(best_cyc)
                for e in ce:
                    remaining.discard(e)
            else:
                # Should not happen, but fallback
                break
        else:
            # No cycle found, or only trivial ones?
            # If no cycle of length >= 3, we must emit edges.
            # But wait, if there is a cycle, we should have found it?
            # Not necessarily, our heuristic might miss it.
            # Let's just pick an edge.
            e = min(remaining)
            pieces.append(list(e))
            remaining.discard(e)
            
    # Safety: add any remaining edges as single pieces
    for e in sorted(remaining):
        pieces.append(list(e))
        
    return pieces


def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE - 5.0
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    
    # Allocate time budget to each graph
    # Graphs are small, so we can afford more time per graph if needed.
    # Let's give each graph (deadline - t0) / 3 roughly, but ensure we finish all.
    # Actually, just use the global deadline.
    
    for spec in specs:
        edges = instance(*spec)
        if time.time() > deadline:
            # Fallback: all edges as single pieces
            results.append([list(e) for e in edges])
            continue
        
        pieces = _decompose(edges, deadline)
        results.append(pieces)
        
    return results

# EVOLVE-BLOCK-END
