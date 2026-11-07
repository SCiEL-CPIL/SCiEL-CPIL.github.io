"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

_N = P['n']
_EDGES = P['edges']
_MINV = P['minimum_vertices']

# Build adjacency bitsets (full graph) for fast induced-degree computation.
_ADJ = [0] * _N
for _a, _b in _EDGES:
    _ADJ[_a] |= (1 << _b)
    _ADJ[_b] |= (1 << _a)

# Precompute full-graph degrees to know the max possible distinct degree count
_FULL_DEGS = [_ADJ[v].bit_count() for v in range(_N)]
_MAX_POSSIBLE_DEGS = len(set(_FULL_DEGS))

_score_cache = {}


def _induced_degree(v, mask):
    # Degree of v within the vertex set represented by mask.
    return (_ADJ[v] & mask).bit_count()


def _score(mask):
    # Number of distinct degrees among selected vertices in the induced subgraph.
    if mask in _score_cache:
        return _score_cache[mask]
    degs = set()
    m = mask
    while m:
        lsb = m & (-m)
        v = lsb.bit_length() - 1
        degs.add(_induced_degree(v, mask))
        m ^= lsb
    result = len(degs)
    _score_cache[mask] = result
    return result


def _mask_to_list(mask):
    result = []
    m = mask
    while m:
        lsb = m & (-m)
        v = lsb.bit_length() - 1
        result.append(int(v))
        m ^= lsb
    result.sort()
    return list(result)


def _exhaustive_search(k, deadline):
    """Exhaustively search all k-vertex subsets."""
    best_mask = None
    best_score = 0
    for combo in combinations(range(_N), k):
        if time.time() > deadline:
            break
        mask = 0
        for v in combo:
            mask |= (1 << v)
        s = _score(mask)
        if s > best_score:
            best_score = s
            best_mask = mask
    return best_mask, best_score


import math


def _simulated_annealing(initial_mask, deadline, rng):
    """Simulated annealing metaheuristic for the maximum diversity problem.
    
    Uses swap and single-bit moves with a geometric cooling schedule.
    Accepts worse moves with probability exp(-delta/T) to escape local optima.
    """
    current_mask = initial_mask
    current_score = _score(initial_mask)
    best_mask = current_mask
    best_score = current_score
    
    # Cooling schedule parameters
    T0 = 5.0          # initial temperature
    T_end = 0.01      # final temperature
    # Number of iterations: we'll run until deadline
    # Use geometric cooling: T = T0 * (T_end/T0)^(it/total)
    # Estimate total iterations based on time budget
    elapsed = time.time() - (deadline - 55.0)  # rough estimate
    remaining = max(deadline - time.time(), 1.0)
    # Each iteration should take ~1ms, so ~1000 iters/sec
    total_iters = int(remaining * 500)  # conservative estimate
    total_iters = max(total_iters, 1000)
    
    cooling_ratio = (T_end / T0) ** (1.0 / total_iters)
    
    for it in range(total_iters):
        if time.time() > deadline:
            break
        
        T = T0 * (cooling_ratio ** it)
        
        # Generate a random move
        move_type = rng.random()
        
        if move_type < 0.7:
            # Swap move: remove one vertex, add another
            in_verts = [v for v in range(_N) if current_mask & (1 << v)]
            out_verts = [v for v in range(_N) if not (current_mask & (1 << v))]
            if not in_verts or not out_verts:
                continue
            vi = rng.choice(in_verts)
            vo = rng.choice(out_verts)
            new_mask = current_mask ^ (1 << vi) ^ (1 << vo)
        else:
            # Single-bit move: add or remove one vertex
            v = rng.randint(0, _N - 1)
            bit = 1 << v
            new_mask = current_mask ^ bit
            if new_mask.bit_count() < _MINV:
                continue
        
        new_score = _score(new_mask)
        delta = new_score - current_score
        
        if delta >= 0:
            # Accept improvement
            current_mask = new_mask
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_mask = new_mask
        else:
            # Accept worse move with probability exp(-delta/T)
            if T > 0:
                accept_prob = math.exp(delta / T)
                if rng.random() < accept_prob:
                    current_mask = new_mask
                    current_score = new_score
    
    return best_mask, best_score


def construct():
    deadline = time.time() + 55.0  # safety margin under 90s limit
    rng = random.Random(42)

    best_mask = None
    best_score = 0

    def consider(mask):
        nonlocal best_mask, best_score
        if mask.bit_count() < _MINV:
            return
        s = _score(mask)
        if s > best_score:
            best_score = s
            best_mask = mask

    # Phase 1: Exhaustive search for k=8, 9, 10 (all tractable)
    for k in [8, 9, 10]:
        if time.time() > deadline:
            break
        bm, bs = _exhaustive_search(k, deadline)
        if bm is not None and bs > best_score:
            best_mask = bm
            best_score = bs

    # If we already hit the theoretical max, we're done
    if best_score >= _MAX_POSSIBLE_DEGS:
        return _mask_to_list(best_mask)

    # Phase 2: Simulated annealing with multiple restarts from various sizes
    if time.time() < deadline:
        # Give SA a generous time budget: use most of the remaining time
        sa_deadline = deadline - 2.0  # leave 2s for final processing
        
        # Multiple SA restarts with different initial subsets
        num_sa_restarts = 50
        sa_time_per_restart = (sa_deadline - time.time()) / num_sa_restarts if num_sa_restarts > 0 else 1.0
        sa_time_per_restart = max(sa_time_per_restart, 0.5)
        
        for trial in range(num_sa_restarts):
            if time.time() > sa_deadline:
                break
            
            # Random initial subset
            k = rng.randint(_MINV, _N)
            verts = rng.sample(range(_N), k)
            mask = 0
            for v in verts:
                mask |= (1 << v)
            consider(mask)
            
            # Run SA from this start with its own deadline
            sa_end = time.time() + sa_time_per_restart
            lm, ls = _simulated_annealing(mask, sa_end, rng)
            if ls > best_score:
                best_score = ls
                best_mask = lm
        
        # Final polish: one more SA run with full remaining time from best so far
        if time.time() < sa_deadline and best_mask is not None:
            final_sa_end = deadline - 1.0
            lm, ls = _simulated_annealing(best_mask, final_sa_end, rng)
            if ls > best_score:
                best_score = ls
                best_mask = lm

    # Fallback: ensure we always have a valid result.
    if best_mask is None:
        seed_mask = 0
        for v in range(_MINV):
            seed_mask |= (1 << v)
        best_mask = seed_mask
        best_score = _score(best_mask)

    # Convert mask to a list of plain Python ints.
    return _mask_to_list(best_mask)
# EVOLVE-BLOCK-END
