"""Rewritten program: greedy long-cycle extraction via randomized DFS + backtracking."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import defaultdict

def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _decompose(edges, deadline, rng):
    """Decompose edge set into cycles and single edges, minimizing piece count."""
    # Build adjacency
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    rem = set(edges)
    rem_adj = defaultdict(set)
    for u, v in rem:
        rem_adj[u].add(v)
        rem_adj[v].add(u)
    
    pieces = []

    def find_longest_cycle_beam(start, beam_width=10, max_depth=50):
        """
        Find a long cycle starting and ending at `start` using Beam Search.
        Returns a list of vertices [v0, v1, ..., vk, v0] or None.
        """
        if not rem_adj[start]:
            return None
            
        # State: (path, current_node)
        # We want to maximize length.
        # Beam: list of (path, current_node)
        # Initialize beam with (start, start)
        beam = [([start], start)]
        
        best_cycle = None
        best_len = 0
        
        for _ in range(max_depth):
            if time.time() > deadline:
                break
            new_beam = []
            for path, current in beam:
                # Try to close the cycle
                if len(path) >= 3 and start in rem_adj[current]:
                    cycle_len = len(path)
                    if cycle_len > best_len:
                        best_len = cycle_len
                        best_cycle = path + [start]
                
                # Expand to neighbors
                # Sort neighbors by degree to keep options open (heuristic)
                neighbors = sorted(rem_adj[current], key=lambda x: -len(rem_adj[x]))
                for nb in neighbors:
                    if nb == start:
                        # Only allow closing if we have a valid cycle (len >= 3)
                        # This is handled above. If we add it to beam, it will close in next step?
                        # Better to handle closing explicitly or allow it in beam but prune if len < 3.
                        if len(path) >= 3:
                            # This is a valid cycle
                            cycle_len = len(path)
                            if cycle_len > best_len:
                                best_len = cycle_len
                                best_cycle = path + [start]
                        continue # Don't add start to beam for expansion
                    
                    if nb in path:
                        continue # Simple cycle constraint
                    
                    new_path = path + [nb]
                    new_beam.append((new_path, nb))
            
            if not new_beam:
                break
                
            # Keep top `beam_width` states
            # Heuristic for ranking: prefer longer paths, or paths with higher degree nodes at end?
            # Longer is better.
            new_beam.sort(key=lambda x: -len(x[0]))
            beam = new_beam[:beam_width]
            
            # Optimization: If we have a cycle, we might stop early if we can't beat it?
            # Max possible length from a path of length L is L + (remaining nodes)?
            # For now, just keep searching.
            
        return best_cycle

    max_iterations = 10000
    iteration = 0
    
    while rem and time.time() < deadline:
        iteration += 1
        if iteration > max_iterations:
            break
            
        # Pick a starting vertex: prefer high degree
        candidates = list(rem_adj.keys())
        if not candidates:
            break
        candidates.sort(key=lambda v: -len(rem_adj[v]))
        
        best_cycle = None
        best_len = 0
        
        # Try a few random starting vertices (weighted toward high degree)
        # We try more starts to find a better global cycle
        num_tries = min(30, len(candidates))
        for _ in range(num_tries):
            if time.time() > deadline:
                break
            # Weighted random selection
            r = rng.random() ** 2  # bias toward top
            idx = int(r * len(candidates))
            if idx >= len(candidates):
                idx = len(candidates) - 1
            start = candidates[idx]
            
            if not rem_adj[start]:
                continue
            
            cycle = find_longest_cycle_beam(start)
            if cycle and len(cycle) - 1 > best_len:
                best_len = len(cycle) - 1
                best_cycle = cycle
        
        # If no cycle found, take a single edge
        if best_cycle and best_len >= 3:
            # Remove edges from cycle
            for i in range(len(best_cycle) - 1):
                u, v = best_cycle[i], best_cycle[i + 1]
                e = (u, v) if u < v else (v, u)
                rem.discard(e)
                rem_adj[u].discard(v)
                rem_adj[v].discard(u)
                if not rem_adj[u]:
                    del rem_adj[u]
                if not rem_adj[v]:
                    del rem_adj[v]
            # Store cycle as list of vertices (closing edge implied)
            pieces.append(best_cycle[:-1])
        else:
            # Take a single edge
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
    
    # If time ran out, add all remaining edges as single-edge pieces
    for u, v in rem:
        pieces.append([u, v])
    
    return pieces


def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin: 150s limit, leave 10s buffer
    
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = _instance(*spec)
        rng = random.Random(42)
        pieces = _decompose(edges, deadline, rng)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
