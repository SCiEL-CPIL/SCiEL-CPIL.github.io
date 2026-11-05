import time
import random
import sys
from collections import defaultdict, deque
import numpy as np

_T0 = time.time()
_DEADLINE = 145.0
_MARGIN = 5.0

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = set(tuple(sorted(e)) for e in zip(order, order[1:]))
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_cycle_heuristic(edges, rng_seed):
    """
    Attempts to find a long cycle in the graph defined by 'edges'.
    Uses randomized DFS with degree-based heuristic to prefer dense regions.
    Returns a list of vertices forming the cycle, or None if no cycle found.
    """
    if not edges:
        return None
    
    # Build adjacency list
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Find vertices with degree >= 2
    candidates = [u for u in adj if len(adj[u]) >= 2]
    if not candidates:
        return None
    
    rng = random.Random(rng_seed)
    
    # Try a few random starts if the first doesn't yield a good result? 
    # Or just one DFS with randomized tie-breaking.
    # Let's try one DFS with specific heuristic: 
    # 1. Pick random start
    # 2. DFS, at each step, choose next unvisited neighbor with highest degree.
    # 3. If stuck, backtrack.
    # 4. If back edge found, return cycle.
    
    start = rng.choice(candidates)
    
    # To find longer cycles, we might want to avoid closing the cycle too early.
    # Standard DFS finds *a* cycle. To find a *long* one, we can try to extend the path as much as possible.
    # A simple heuristic: 
    #   path = [start]
    #   while True:
    #      current = path[-1]
    #      unvisited_neighbors = [v for v in adj[current] if v not in path]
    #      if unvisited_neighbors:
    #          # Pick one with max degree
    #          next_v = max(unvisited_neighbors, key=lambda x: len(adj[x]))
    #          path.append(next_v)
    #      else:
    #          # Check for back edge to any vertex in path
    #          back_edge = None
    #          for v in adj[current]:
    #              if v in path:
    #                  # Found cycle
    #                  idx = path.index(v)
    #                  cycle = path[idx:]
    #                  if len(cycle) >= 3:
    #                      return cycle
    #                  # If cycle is too short, maybe try to find another?
    #                  # For now, return it.
    #          # If no back edge, dead end, backtrack
    #          if len(path) == 1:
    #              return None
    #          path.pop()
    
    # Let's implement this greedy path extension.
    path = [start]
    visited_in_path = {start}
    
    while path:
        current = path[-1]
        
        # Find unvisited neighbors
        unvisited_neighbors = []
        for v in adj[current]:
            if v not in visited_in_path:
                unvisited_neighbors.append(v)
        
        if unvisited_neighbors:
            # Heuristic: pick neighbor with highest degree to stay in core
            # Add some randomness to break ties and explore different branches
            # Sort by degree descending, then shuffle top few?
            # Or just pick max degree.
            next_v = max(unvisited_neighbors, key=lambda x: len(adj[x]))
            path.append(next_v)
            visited_in_path.add(next_v)
        else:
            # No unvisited neighbors. Check for back edge to any vertex in path.
            # We want the longest cycle, so we should look for the "earliest" vertex in the path 
            # that is connected to current? No, that would make a shorter cycle.
            # We want to close the cycle with the "latest" possible vertex in the path? 
            # Actually, any back edge forms a cycle. 
            # The cycle length is index(current) - index(v) + 1.
            # To maximize length, we want v to be as early in the path as possible.
            best_v = None
            best_idx = -1
            for v in adj[current]:
                if v in visited_in_path:
                    idx = path.index(v)
                    # Ensure it's a valid cycle (length >= 3)
                    # If idx == 0, it's a cycle back to start.
                    # If idx > 0, it's a cycle within the path.
                    if idx < len(path) - 1: # Must not be the immediate parent to avoid 2-cycles in undirected (though we treat as simple graph)
                        if best_v is None or idx < best_idx:
                            best_v = v
                            best_idx = idx
            
            if best_v is not None:
                # Extract cycle
                cycle = path[best_idx:]
                if len(cycle) >= 3:
                    return cycle
                else:
                    # Cycle too short? Shouldn't happen in simple graph if we avoid parents.
                    # But if we did, we might want to backtrack and try another branch.
                    pass
            
            # Backtrack
            path.pop()
            visited_in_path.discard(path[-1] if path else None)
            if not path:
                break
    
    return None

def _decompose(edges):
    n = max((max(u, v) for u, v in edges), default=0) + 1
    present = set((min(u, v), max(u, v)) for u, v in edges)
    if not present:
        return []
    
    pieces = []
    remaining = set(present)
    
    # Time check
    while remaining:
        if time.time() > _DEADLINE - _MARGIN:
            # Fallback: add all remaining as singles
            for e in remaining:
                pieces.append([e[0], e[1]])
            break
        
        best_cycle = None
        best_len = 0
        
        # Try multiple random seeds to find a long cycle
        # The heuristic is deterministic given the seed, so varying seed varies the path taken.
        for seed in range(10):
            cycle = _find_cycle_heuristic(remaining, seed)
            if cycle:
                l = len(cycle)
                if l > best_len:
                    best_len = l
                    best_cycle = cycle
                # If we found a very long cycle (e.g., > 10), we might stop early?
                # For now, keep checking a few.
        
        if best_cycle:
            pieces.append(best_cycle)
            # Remove edges of the cycle
            for i in range(len(best_cycle)):
                u = best_cycle[i]
                v = best_cycle[(i + 1) % len(best_cycle)]
                e = (min(u, v), max(u, v))
                remaining.discard(e)
        else:
            # No cycles left. Add all remaining edges as singles.
            for e in remaining:
                pieces.append([e[0], e[1]])
            remaining = set()
            
    return pieces

def construct():
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > _DEADLINE - _MARGIN:
            # Fill remaining with all-singles
            e2 = _instance(*spec)
            results.append([[u, v] for u, v in e2])
            continue
            
        edges = _instance(*spec)
        pieces = _decompose(edges)
        results.append(pieces)
        
    return results
