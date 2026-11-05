import time
import random
import sys
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

def _find_long_cycle(adj, n, max_attempts=20, deadline=None):
    """
    Find a long simple cycle in the graph represented by adj.
    Uses a greedy randomized strategy with degree-based heuristics.
    Returns the cycle as a list of vertices or None.
    """
    best_cycle = None
    best_len = 0
    
    # Try starting from each vertex, prioritizing high-degree vertices
    candidates = [v for v in range(n) if adj[v]]
    if not candidates:
        return None
    
    # Sort by degree descending, but add some randomness
    candidates.sort(key=lambda v: len(adj[v]), reverse=True)
    # Shuffle top candidates to add diversity
    rng_local = random.Random()
    
    for start in candidates:
        if deadline and time.time() > deadline:
            break
        if not adj[start]:
            continue
            
        # Run multiple random attempts from this start
        for attempt in range(max_attempts):
            if deadline and time.time() > deadline:
                break
                
            path = [start]
            visited = {start}
            current = start
            
            while True:
                neighbors = list(adj[current])
                # Filter out visited neighbors
                unvisited_neighbors = [v for v in neighbors if v not in visited]
                
                # Check if we can close the cycle
                can_close = start in neighbors and len(path) >= 3
                
                if not unvisited_neighbors:
                    # Stuck. Can we close?
                    if can_close:
                        # Found a cycle
                        if len(path) > best_len:
                            best_len = len(path)
                            best_cycle = path[:]
                    break
                else:
                    # We have unvisited neighbors.
                    # Use degree-based heuristic: prefer neighbors with higher degree
                    # to keep more options open
                    scored = [(len(adj[v]), v) for v in unvisited_neighbors]
                    scored.sort(reverse=True)
                    
                    # With some probability, pick randomly for diversity
                    if rng_local.random() < 0.3:
                        next_node = rng_local.choice(unvisited_neighbors)
                    else:
                        # Pick from top 3 highest-degree neighbors
                        top_k = scored[:min(3, len(scored))]
                        next_node = rng_local.choice([v for _, v in top_k])
                    
                    path.append(next_node)
                    visited.add(next_node)
                    current = next_node

    return best_cycle

def _decompose(edges, n, deadline, rnd=None):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    pieces = []
    remaining = len(edges)
    
    while remaining > 0:
        if time.time() > deadline:
            break
            
        # Find a long cycle
        cyc = _find_long_cycle(adj, n, max_attempts=10, deadline=deadline)
        
        if cyc:
            # Remove cycle edges
            m = len(cyc)
            # The cycle is [v0, v1, ..., v_{m-1}]
            # Edges are (v0, v1), (v1, v2), ..., (v_{m-1}, v0)
            for i in range(m):
                a = cyc[i]
                b = cyc[(i + 1) % m]
                adj[a].discard(b)
                adj[b].discard(a)
            remaining -= m
            pieces.append(cyc)
        else:
            # No cycle found, remove an edge
            # Find an edge
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
    num_attempts = 50 # Adjust based on time
    base_seed = 42
    
    for i in range(num_attempts):
        if time.time() > deadline - 1.0:
            break
            
        random.seed(base_seed + i)
        pieces = _decompose(edges, n, deadline, None)
        cnt = len(pieces)
        
        if cnt < best_count:
            best_count = cnt
            best = pieces
            # If we found a very good solution, we can reduce attempts
            if cnt <= len(edges) // 3 + 2:
                # Good enough, stop early
                break
            
    return best

def construct():
    start_time = time.time()
    total_budget = 140.0 # Safety margin for 150s limit
    per_graph_budget = total_budget / 3.0
    
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    
    for i, (n, p, seed) in enumerate(specs):
        edges = _instance(n, p, seed)
        # Allocate time for this graph, ensuring we don't exceed total budget
        graph_deadline = start_time + per_graph_budget * (i + 1)
        
        # Ensure we leave at least 2 seconds for the rest
        if graph_deadline > start_time + total_budget - 2.0:
            graph_deadline = start_time + total_budget - 2.0
            
        pieces = _solve_one(edges, n, graph_deadline)
        results.append(pieces)
        
        # If we are running out of time, fill remaining with trivial solutions
        if time.time() > start_time + total_budget - 5.0:
            for j in range(i + 1, len(specs)):
                nn, pp, ss = specs[j]
                ee = _instance(nn, pp, ss)
                results.append([list(e) for e in ee])
            break
            
    return results
