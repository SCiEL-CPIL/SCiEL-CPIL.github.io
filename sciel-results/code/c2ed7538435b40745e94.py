import time
import random
import sys
import heapq
from collections import defaultdict

sys.setrecursionlimit(100000)

def _instance(n, p, seed):
    """Copied from the verifier (must match exactly)."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_long_cycle(adj, n, rng, max_attempts=30):
    """
    Find a long simple cycle in the graph represented by adj.
    Uses a randomized greedy strategy with Warnsdorff-like heuristic.
    """
    best_cycle = None
    best_len = 0
    
    # Try starting from each vertex
    for start in range(n):
        if not adj[start]:
            continue
            
        for _ in range(max_attempts):
            path = [start]
            visited = {start}
            current = start
            
            while True:
                neighbors = list(adj[current])
                unvisited_neighbors = [v for v in neighbors if v not in visited]
                
                can_close = start in neighbors and len(path) >= 3
                
                if not unvisited_neighbors:
                    if can_close:
                        if len(path) > best_len:
                            best_len = len(path)
                            best_cycle = path[:]
                    break
                else:
                    # Warnsdorff-like heuristic: pick neighbor with fewest unvisited neighbors
                    def score(v):
                        return sum(1 for w in adj[v] if w not in visited)
                    
                    unvisited_neighbors.sort(key=lambda v: (score(v), rng.random()))
                    next_node = unvisited_neighbors[0]
                    path.append(next_node)
                    visited.add(next_node)
                    current = next_node

    return best_cycle

def _decompose_greedy(adj, n, deadline, rng):
    """
    Greedily extract cycles from the graph.
    Returns a list of pieces (cycles or edges).
    """
    pieces = []
    remaining = sum(len(s) for s in adj) // 2
    
    while remaining > 0:
        if time.time() > deadline:
            break
            
        # Find a long cycle
        cyc = _find_long_cycle(adj, n, rng, max_attempts=25)
        
        if cyc:
            m = len(cyc)
            for i in range(m):
                a = cyc[i]
                b = cyc[(i + 1) % m]
                adj[a].discard(b)
                adj[b].discard(a)
            remaining -= m
            pieces.append(cyc)
        else:
            # No cycle found, remove an edge
            edge_removed = False
            for u in range(n):
                for v in list(adj[u]):
                    if u < v:
                        adj[u].discard(v)
                        adj[v].discard(u)
                        remaining -= 1
                        pieces.append([u, v])
                        edge_removed = True
                        break
                if edge_removed:
                    break
            if not edge_removed:
                break

    return pieces

def _solve_one(edges, n, deadline):
    best = None
    best_count = float('inf')
    
    # Use multiple random seeds for the internal randomization
    num_attempts = 25
    base_seed = 42
    
    for i in range(num_attempts):
        if time.time() > deadline - 1.0:
            break
            
        rng = random.Random(base_seed + i)
        adj = _build_adj(edges, n)
        pieces = _decompose_greedy(adj, n, deadline, rng)
        cnt = len(pieces)
        
        if cnt < best_count:
            best_count = cnt
            best = pieces

    return best

def construct():
    start_time = time.time()
    total_budget = 140.0
    per_graph_budget = total_budget / 3.0
    
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    
    for i, (n, p, seed) in enumerate(specs):
        edges = _instance(n, p, seed)
        graph_deadline = start_time + per_graph_budget * (i + 1)
        
        if graph_deadline > start_time + total_budget - 2.0:
            graph_deadline = start_time + total_budget - 2.0
            
        pieces = _solve_one(edges, n, graph_deadline)
        results.append(pieces)
        
        if time.time() > start_time + total_budget - 5.0:
            for j in range(i + 1, len(specs)):
                nn, pp, ss = specs[j]
                ee = _instance(nn, pp, ss)
                results.append([list(e) for e in ee])
            break
            
    return results
