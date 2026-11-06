"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import heapq
from collections import deque

N = 9
TOTAL_VERTICES = 1 << N

# Precompute neighbors for each vertex
_NEIGHBORS = [None] * TOTAL_VERTICES
for _v in range(TOTAL_VERTICES):
    _NEIGHBORS[_v] = [_v ^ (1 << b) for b in range(N)]

# Precompute adjacency bitmask for quick conflict checks
_ADJ_MASK = [0] * TOTAL_VERTICES
for _v in range(TOTAL_VERTICES):
    _m = 0
    for _nb in _NEIGHBORS[_v]:
        _m |= (1 << _nb)
    _ADJ_MASK[_v] = _m


def _greedy_extend(path, used_mask, bit_order=None):
    """Greedy extension of a path with given bit priority order."""
    if bit_order is None:
        bit_order = list(range(N))
    cur = path[-1]
    while True:
        nxt = None
        for b in bit_order:
            v = cur ^ (1 << b)
            if used_mask & (1 << v):
                continue
            # Check induced constraint
            conflict = _ADJ_MASK[v] & (used_mask ^ (1 << v) ^ (1 << cur))
            if conflict:
                continue
            nxt = v
            break
        if nxt is None:
            break
        path.append(nxt)
        used_mask |= (1 << nxt)
        cur = nxt
    return path, used_mask


def _score_state(path, used_mask):
    """
    Score a partial snake state for beam search.
    Higher score = more promising (more remaining options at frontier).
    Uses: number of valid next vertices from current position,
    weighted by how many of THOSE also have future options.
    """
    cur = path[-1]
    next_options = 0
    for v in _NEIGHBORS[cur]:
        if used_mask & (1 << v):
            continue
        conflict = _ADJ_MASK[v] & (used_mask ^ (1 << v) ^ (1 << cur))
        if conflict:
            continue
        next_options += 1
        # Bonus: count how many of v's neighbors are still open
        for nv in _NEIGHBORS[v]:
            if used_mask & (1 << nv):
                continue
            if nv == cur:
                continue
            c2 = _ADJ_MASK[nv] & (used_mask | (1 << v)) ^ (1 << nv) ^ (1 << v)
            if not c2:
                next_options += 0.5
    return next_options


def _beam_search(start_path, start_used, beam_width, deadline, max_steps):
    """
    Beam search over partial snake extensions.
    Maintains a priority queue of states scored by _score_state.
    Returns the best (longest) path found.
    """
    # State: (negative_score, step_counter, path_tuple, used_mask)
    # Use a counter for tie-breaking to avoid comparing paths
    counter = 0
    beam = []
    init_score = _score_state(start_path, start_used)
    heapq.heappush(beam, (-init_score, counter, tuple(start_path), start_used))
    counter += 1
    
    best_path = list(start_path)
    best_len = len(best_path)
    
    for _ in range(max_steps):
        if time.time() > deadline:
            break
        if not beam:
            break
        
        # Pop best state
        neg_score, _, path_tuple, used_mask = heapq.heappop(beam)
        path = list(path_tuple)
        
        if len(path) > best_len:
            best_path = path
            best_len = len(path)
        
        cur = path[-1]
        
        # Generate all valid extensions
        candidates = []
        for v in _NEIGHBORS[cur]:
            if used_mask & (1 << v):
                continue
            conflict = _ADJ_MASK[v] & (used_mask ^ (1 << v) ^ (1 << cur))
            if conflict:
                continue
            candidates.append(v)
        
        for v in candidates:
            new_path = path + [v]
            new_used = used_mask | (1 << v)
            new_score = _score_state(new_path, new_used)
            heapq.heappush(beam, (-new_score, counter, tuple(new_path), new_used))
            counter += 1
        
        # Trim beam to width
        while len(beam) > beam_width:
            heapq.heappop(beam)
    
    return best_path


def _beam_search_multi_restart(deadline, beam_width=50, max_steps=500):
    """
    Run beam search from multiple random starting seeds.
    Returns the best path found.
    """
    best_path = []
    best_len = 0
    
    # Try several different greedy seeds as starting points
    for trial in range(30):
        if time.time() > deadline:
            break
        
        # Create a short random seed (2-5 vertices)
        seed_len = random.randint(2, 5)
        seed = [0]
        used = 1
        cur = 0
        for _ in range(seed_len - 1):
            options = []
            for b in range(N):
                v = cur ^ (1 << b)
                if used & (1 << v):
                    continue
                conflict = _ADJ_MASK[v] & (used ^ (1 << v) ^ (1 << cur))
                if not conflict:
                    options.append(v)
            if not options:
                break
            cur = random.choice(options)
            seed.append(cur)
            used |= (1 << cur)
        
        # Extend greedily first to get a reasonable base
        seed_path, seed_used = _greedy_extend(list(seed), used)
        
        if len(seed_path) > best_len:
            best_path = list(seed_path)
            best_len = len(best_path)
        
        # Now run beam search from the greedy-extended path
        if time.time() > deadline - 5:
            break
        
        extended = _beam_search(seed_path, seed_used, beam_width, deadline, max_steps)
        if len(extended) > best_len:
            best_path = extended
            best_len = len(extended)
    
    return best_path


def _backtrack_from_position(path, pos, used_mask, deadline, max_extend=30):
    """
    Backtrack from a specific position in the path, trying alternative
    continuations. Returns the best extended path.
    """
    sub_path = path[:pos + 1]
    sub_used = 0
    for v in sub_path:
        sub_used |= (1 << v)
    
    # Greedy extend first
    greedy_path, greedy_used = _greedy_extend(list(sub_path), sub_used)
    best = list(greedy_path)
    best_len = len(best)
    
    # Then try beam search from the truncated point
    if time.time() < deadline - 2:
        extended = _beam_search(sub_path, sub_used, 30, deadline, 200)
        if len(extended) > best_len:
            best = extended
            best_len = len(extended)
    
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 150s limit
    
    # Phase 1: Get a good initial seed via greedy
    path = [0]
    used_mask = 1
    path, used_mask = _greedy_extend(path, used_mask)
    best_path = list(path)
    best_len = len(best_path)
    
    # Phase 2: Beam search from the greedy seed
    if time.time() < deadline - 15:
        extended = _beam_search(best_path, used_mask, beam_width=80, deadline=deadline, max_steps=800)
        if len(extended) > best_len:
            best_path = extended
            best_len = len(best_path)
            used_mask = 0
            for v in best_path:
                used_mask |= (1 << v)
    
    # Phase 3: Multi-restart beam search with different seeds
    if time.time() < deadline - 10:
        multi_best = _beam_search_multi_restart(deadline, beam_width=40, max_steps=400)
        if len(multi_best) > best_len:
            best_path = multi_best
            best_len = len(best_path)
            used_mask = 0
            for v in best_path:
                used_mask |= (1 << v)
    
    # Phase 4: Local improvement - try re-extending from various positions
    if time.time() < deadline - 5:
        for _ in range(100):
            if time.time() > deadline - 3:
                break
            if len(best_path) <= 3:
                break
            pos = random.randint(1, len(best_path) - 2)
            improved = _backtrack_from_position(best_path, pos, used_mask, deadline, max_extend=20)
            if len(improved) > best_len:
                best_path = improved
                best_len = len(best_path)
                used_mask = 0
                for v in best_path:
                    used_mask |= (1 << v)
    
    # Phase 5: Final beam search pass if time remains
    if time.time() < deadline - 2:
        extended = _beam_search(best_path, used_mask, beam_width=100, deadline=deadline, max_steps=600)
        if len(extended) > best_len:
            best_path = extended
            best_len = len(best_path)
    
    return best_path
# EVOLVE-BLOCK-END
