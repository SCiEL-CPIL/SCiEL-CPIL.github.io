"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict


def _greedy_cover(v, k, t, rng, deadline):
    """Bitset-accelerated greedy covering with randomized tie-breaking."""
    # Index all t-subsets
    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    t_index = {ts: i for i, ts in enumerate(all_t)}
    
    # Precompute for each point x and each (t-1)-subset of other points,
    # which t-subsets are covered when x is added to that (t-1)-subset.
    # For greedy: when we have a partial block B, adding x covers
    # all t-subsets that contain x and (t-1) elements from B.
    
    # We'll track uncovered as a boolean array
    uncovered = bytearray([1]) * n_t
    n_uncovered = n_t
    blocks = []
    
    while n_uncovered > 0:
        if time.time() > deadline:
            break
        # Find the first uncovered t-subset to seed a new block
        first_idx = -1
        for i in range(n_t):
            if uncovered[i]:
                first_idx = i
                break
        if first_idx == -1:
            break
        seed = list(all_t[first_idx])
        block = set(seed)
        
        # Greedily add points to the block
        while len(block) < k:
            if time.time() > deadline:
                break
            best_gain = -1
            best_points = []
            for x in range(v):
                if x in block:
                    continue
                # Count how many uncovered t-subsets are covered by adding x
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    idx = t_index.get(ts)
                    if idx is not None and uncovered[idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_points = [x]
                elif gain == best_gain and gain > 0:
                    best_points.append(x)
            if not best_points:
                break
            block.add(rng.choice(best_points))
        
        if len(block) != k:
            break
        block = sorted(block)
        blocks.append(block)
        
        # Update uncovered
        for ts in itertools.combinations(block, t):
            idx = t_index[ts]
            if uncovered[idx]:
                uncovered[idx] = 0
                n_uncovered -= 1
    
    return blocks


def _is_valid(v, k, t, blocks):
    """Check if the covering is valid."""
    all_t = set(itertools.combinations(range(v), t))
    covered = set()
    for block in blocks:
        for ts in itertools.combinations(block, t):
            covered.add(ts)
    return all_t.issubset(covered)


def _remove_redundant(v, k, t, blocks, rng, deadline):
    """Try to remove redundant blocks."""
    if not blocks:
        return blocks
    n = len(blocks)
    # Precompute coverage counts
    t_index = {ts: i for i, ts in enumerate(itertools.combinations(range(v), t))}
    n_t = len(t_index)
    
    # Count how many blocks cover each t-subset
    cover_count = [0] * n_t
    for block in blocks:
        for ts in itertools.combinations(block, t):
            cover_count[t_index[ts]] += 1
    
    # Try removing blocks in random order
    order = list(range(n))
    rng.shuffle(order)
    
    kept = [True] * n
    for i in order:
        if time.time() > deadline:
            break
        if not kept[i]:
            continue
        block = blocks[i]
        # Check if removing this block leaves any t-subset uncovered
        can_remove = True
        for ts in itertools.combinations(block, t):
            idx = t_index[ts]
            if cover_count[idx] <= 1:
                can_remove = False
                break
        if can_remove:
            kept[i] = False
            for ts in itertools.combinations(block, t):
                cover_count[t_index[ts]] -= 1
    
    return [blocks[i] for i in range(n) if kept[i]]


def construct(v, k, t):
    """Construct a covering design C(v,k,t) using bitset-accelerated greedy with restarts."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_blocks = None
    best_len = float('inf')
    
    # Multiple randomized restarts
    max_restarts = 50
    for attempt in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
        rng = random.Random(attempt * 7919 + 13)
        
        # Phase 1: Greedy covering
        blocks = _greedy_cover(v, k, t, rng, deadline - 2.0)
        
        if not blocks:
            continue
        
        # Phase 2: Remove redundant blocks
        blocks = _remove_redundant(v, k, t, blocks, rng, deadline - 1.0)
        
        # Verify validity
        if not _is_valid(v, k, t, blocks):
            continue
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks[:]
            # If we've found a very good solution, reduce remaining time per restart
            if best_len <= 70:
                break
    
    # Fallback: if no valid solution found, run a final greedy without time pressure
    if best_blocks is None:
        rng = random.Random(0)
        best_blocks = _greedy_cover(v, k, t, rng, deadline)
        if best_blocks:
            best_blocks = _remove_redundant(v, k, t, best_blocks, rng, deadline)
    
    return best_blocks if best_blocks else []
# EVOLVE-BLOCK-END
