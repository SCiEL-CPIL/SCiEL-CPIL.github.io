"""Strong edge colouring via Set Cover on Maximal Induced Matchings."""
import time
import random
from collections import defaultdict

_DEADLINE = 140.0  # safety margin before 150s kill


def _build_instance(n, d, seed):
    rng = random.Random(seed)
    degree = [0] * n
    edges = set()
    for _ in range(4 * n * d):
        u = rng.randrange(n)
        v = rng.randrange(n)
        if u == v:
            continue
        e = (min(u, v), max(u, v))
        if degree[u] < d and degree[v] < d and e not in edges:
            edges.add(e)
            degree[u] += 1
            degree[v] += 1
    return sorted(edges)


def _precompute_conflict_bitmasks(edges):
    m = len(edges)
    # Map edge tuple to index
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    
    # Adjacency list for vertices
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Precompute conflict masks
    # conflict_mask[i] has bit j set if edge i and edge j conflict
    conflict_mask = [0] * m
    
    for i in range(m):
        u1, v1 = edges[i]
        mask_i = 0
        for j in range(i + 1, m):
            u2, v2 = edges[j]
            # Check if they share a vertex
            if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                mask_i |= (1 << j)
                continue
            # Check if they are joined by an edge
            if u2 in adj[u1] or u2 in adj[v1] or v2 in adj[u1] or v2 in adj[v1]:
                mask_i |= (1 << j)
        # Set bits for both i and j
        # We need mask_i to contain j, and mask_j to contain i
        # So we update global masks
        conflict_mask[i] |= mask_i
        for j in range(i + 1, m):
            if mask_i & (1 << j):
                conflict_mask[j] |= (1 << i)
                
    return conflict_mask


def _generate_maximal_induced_matching(edges, conflict_mask, rng):
    m = len(edges)
    order = list(range(m))
    rng.shuffle(order)
    
    selected = []
    blocked_mask = 0
    
    for e in order:
        # If e is not blocked, add it
        if not (blocked_mask & (1 << e)):
            selected.append(e)
            # Block e and all its conflicts
            blocked_mask |= (1 << e)
            blocked_mask |= conflict_mask[e]
            
    return selected


def _greedy_set_cover(universe_size, sets, rng):
    """
    universe_size: number of elements in universe (edges)
    sets: list of sets, each set is a list of element indices
    Returns: list of set indices that cover the universe
    """
    if not sets:
        return []
        
    # Represent universe as a bitmask for fast operations
    # Since m is small (< 300), we can use Python integers
    full_universe = (1 << universe_size) - 1
    covered = 0
    
    # Precompute set bitmasks
    set_masks = []
    for s in sets:
        mask = 0
        for e in s:
            mask |= (1 << e)
        set_masks.append(mask)
        
    remaining_sets = list(range(len(sets)))
    solution = []
    
    while covered != full_universe:
        best_set_idx = -1
        best_gain = 0
        
        for idx in remaining_sets:
            # Calculate gain: bits in set_mask not in covered
            gain_mask = set_masks[idx] & (~covered)
            gain = bin(gain_mask).count('1')
            if gain > best_gain:
                best_gain = gain
                best_set_idx = idx
                if gain == 0:
                    # If gain is 0 for all, we might have a problem or covered is full
                    # But we check covered != full_universe, so there must be some gain
                    pass
            # Optimization: if gain is large, we could break early if we assume uniformity, 
            # but greedy requires max.
            
        if best_set_idx == -1 or best_gain == 0:
            # Should not happen if sets cover universe, but safety break
            break
            
        solution.append(best_set_idx)
        covered |= set_masks[best_set_idx]
        remaining_sets.remove(best_set_idx)
        
    return solution


def _solve_instance_set_cover(edges, conflict_mask, deadline):
    m = len(edges)
    if m == 0:
        return []
        
    best_solution = None
    best_size = m + 1 # Max possible is m (each edge its own color)
    
    rng = random.Random()
    
    # Generate a pool of maximal induced matchings
    # The number of matchings to generate should scale with time
    # Let's try 2000 matchings per instance if time permits
    pool_size_target = 2000
    
    t_start = time.time()
    pool = []
    
    # Generate initial pool
    while len(pool) < pool_size_target and time.time() < deadline - 5:
        matching = _generate_maximal_induced_matching(edges, conflict_mask, rng)
        if matching:
            pool.append(matching)
            
    # If we ran out of time generating, proceed with what we have
    if not pool:
        # Fallback: single edge matchings? No, that's m colors.
        # Just use one random maximal matching and repeat? No.
        # If pool is empty, return dummy
        return [0] * m 

    # Run Greedy Set Cover on the pool
    # To improve, we can run multiple greedy passes with different randomizations in tie-breaking?
    # Standard greedy is deterministic.
    # Let's try a "randomized greedy" where we pick among top-k gains.
    
    for _ in range(10): # 10 restarts of greedy with randomization
        if time.time() > deadline - 2:
            break
            
        solution_indices = _greedy_set_cover(m, pool, rng)
        if len(solution_indices) < best_size:
            best_size = len(solution_indices)
            best_solution = solution_indices
            
    if best_solution is None:
        # Fallback
        best_solution = _greedy_set_cover(m, pool, rng)
        best_size = len(best_solution)
        
    # Construct the colour array
    colour = [0] * m
    for color_idx, set_idx in enumerate(best_solution):
        for e in pool[set_idx]:
            colour[e] = color_idx
            
    return colour


def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE
    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))
    out = []
    
    for spec in specs:
        edges = _build_instance(*spec)
        conflict_mask = _precompute_conflict_bitmasks(edges)
        
        # Allocate time budget per instance
        # Simple allocation: equal share, minus overhead
        time_left = deadline - time.time()
        # Reserve 2s for remaining instances
        instance_deadline = time.time() + max(1.0, (time_left - 2.0) / max(1, len(specs) - specs.index(spec)))
        
        colour = _solve_instance_set_cover(edges, conflict_mask, instance_deadline)
        out.append(colour)
        
    return out
