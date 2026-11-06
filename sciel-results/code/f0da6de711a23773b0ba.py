import time
import random
import sys
from collections import defaultdict
import heapq

sys.setrecursionlimit(100000)

def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_long_cycle_beam(adj, start, end, width, deadline):
    """
    Find a long simple cycle from start to end (closing the cycle) using beam search.
    Returns the path (list of nodes) or None.
    """
    # State: (neg_length, current_node, path_tuple, visited_frozenset)
    # We use a max-heap (min-heap with negative length) to prioritize longer paths.
    # However, we want to find *any* cycle, preferably a long one.
    # Beam search keeps top K states.
    
    initial_state = (0, start, (start,), frozenset({start}))
    beam = [initial_state]
    
    # We want to find a path from start to end? 
    # No, the cycle is start -> ... -> end -> start.
    # So we start at start, want to reach end, such that (end, start) is an edge.
    # And the path must be simple.
    
    best_cycle = None
    best_len = 0
    
    # Beam search steps
    max_steps = 50  # Safety limit on beam search depth/iterations
    
    for _ in range(max_steps):
        if time.time() > deadline:
            break
        if not beam:
            break
            
        new_beam = []
        for neg_len, curr, path, visited in beam:
            curr_len = -neg_len
            # Check if we can close the cycle
            if curr == end and curr_len >= 2: # path length >= 3 (start, ..., end)
                # Verify edge (end, start) exists
                if start in adj.get(end, set()):
                    # Found a cycle
                    if curr_len + 1 > best_len:
                        best_len = curr_len + 1
                        best_cycle = list(path)
                        # We can continue searching for even longer cycles
                        # Or return immediately? 
                        # Let's continue to see if we can find a longer one.
                        # But beam search might not guarantee global max.
                        # For now, we record the best found.
            
            # Expand
            for nb in adj.get(curr, set()):
                if nb in visited:
                    continue
                # If nb is end, we can only take it if we can close (which we check above)
                # If nb is start, we can't take it unless we are at end (handled above)
                if nb == start:
                    continue
                
                new_visited = visited | {nb}
                new_path = path + (nb,)
                new_neg_len = neg_len - 1 # length increases by 1
                
                # Pruning: if we already have a cycle longer than this path can possibly become?
                # Max possible length is n.
                
                new_beam.append((new_neg_len, nb, new_path, new_visited))
        
        if not new_beam:
            break
            
        # Keep top 'width' states based on length (most negative neg_len)
        new_beam.sort(key=lambda x: x[0])
        beam = new_beam[:width]
        
        # Optimization: If we have a cycle, and the max possible length of any state in beam 
        # is less than best_len, we can stop.
        max_potential = -beam[0][0] + 1 # current max len in beam + 1 (to close)
        if best_cycle and max_potential <= best_len:
            break

    return best_cycle

def _greedy_decompose(edges, n, deadline, seed):
    """
    Greedy cycle extraction using beam search to find long cycles.
    """
    remaining = set(edges)
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    pieces = []
    rng = random.Random(seed)
    
    while remaining:
        if time.time() >= deadline:
            break
        
        # Pick a random edge to start
        start_edge = next(iter(remaining))
        u, v = start_edge
        
        # Try to find a long cycle containing this edge
        # We can try to find a cycle from u to v, or v to u.
        # Let's try both and pick the longest.
        
        cycle1 = _find_long_cycle_beam(adj, u, v, width=50, deadline=deadline)
        cycle2 = _find_long_cycle_beam(adj, v, u, width=50, deadline=deadline)
        
        best_cycle = None
        best_len = 0
        if cycle1 and len(cycle1) > best_len:
            best_cycle = cycle1
            best_len = len(cycle1)
        if cycle2 and len(cycle2) > best_len:
            best_cycle = cycle2
            best_len = len(cycle2)
        
        if best_cycle:
            # Extract edges from cycle
            cycle_edges = set()
            for i in range(len(best_cycle)):
                a, b = best_cycle[i], best_cycle[(i + 1) % len(best_cycle)]
                e = (a, b) if a < b else (b, a)
                cycle_edges.add(e)
            
            # Remove from remaining
            for e in cycle_edges:
                remaining.discard(e)
                a, b = e
                if b in adj[a]: adj[a].discard(b)
                if a in adj[b]: adj[b].discard(a)
            
            pieces.append(best_cycle)
        else:
            # No cycle found involving this edge.
            # Add as singleton.
            pieces.append([u, v])
            remaining.discard(start_edge)
            if v in adj[u]: adj[u].discard(v)
            if u in adj[v]: adj[v].discard(u)
            
    # Add any remaining edges as singletons
    for e in remaining:
        pieces.append(list(e))
        
    return pieces

def _solve_one(n, p, seed, deadline):
    edges = _instance(n, p, seed)
    
    # Run multiple restarts to find a better decomposition
    best_pieces = None
    best_count = float('inf')
    
    # Reserve some time for the final best run?
    # Just run as many as possible within deadline.
    # Each run should be fast.
    
    start_time = time.time()
    max_restarts = 50
    
    for i in range(max_restarts):
        if time.time() >= deadline - 1.0: # Safety margin
            break
            
        run_deadline = deadline - 0.5
        # Give each restart a portion of the time? 
        # Or just let it run until the global deadline?
        # Beam search is bounded by width and steps, so it's fast.
        # The loop is the main cost.
        
        pieces = _greedy_decompose(edges, n, run_deadline, seed + i)
        
        if len(pieces) < best_count:
            best_count = len(pieces)
            best_pieces = pieces
            
    if best_pieces is None:
        # Fallback: all singletons
        best_pieces = [list(e) for e in edges]
        
    return best_pieces

def construct():
    t0 = time.time()
    total_budget = 120.0
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    
    # Allocate time evenly, with a buffer
    per_spec_time = (total_budget - 5.0) / len(specs)
    
    for i, spec in enumerate(specs):
        deadline = time.time() + per_spec_time
        try:
            pieces = _solve_one(*spec, deadline)
        except Exception:
            pieces = [list(e) for e in _instance(*spec)]
        results.append(pieces)
        
    return results

if __name__ == "__main__":
    res = construct()
    print([len(r) for r in res])
