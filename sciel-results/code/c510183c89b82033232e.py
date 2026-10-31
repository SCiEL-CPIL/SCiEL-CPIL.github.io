"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time

def _generate_all_blocks(v, k):
    """Generate all k-subsets of range(v)."""
    return list(itertools.combinations(range(v), k))

def _cover_count(block, t):
    """Return the set of t-subsets covered by a block."""
    return set(itertools.combinations(block, t))

def _greedy_cover(v, k, t, all_blocks, random_seed=None):
    """Maximum-coverage greedy: repeatedly pick block covering most uncovered t-subsets."""
    if random_seed is not None:
        rng = random.Random(random_seed)
    else:
        rng = random.Random()
    
    all_t_subsets = set(itertools.combinations(range(v), t))
    uncovered = set(all_t_subsets)
    blocks = []
    
    # Precompute coverage for each block
    block_cover = {b: set(itertools.combinations(b, t)) for b in all_blocks}
    
    while uncovered:
        best_block = None
        best_count = -1
        best_tiebreak = -1
        
        for b in all_blocks:
            cov = len(block_cover[b] & uncovered)
            if cov > best_count or (cov == best_count and cov > 0 and rng.random() > 0.5):
                best_count = cov
                best_block = b
                best_tiebreak = rng.random()
        
        if best_count <= 0:
            break
        
        blocks.append(best_block)
        uncovered -= block_cover[best_block]
    
    return blocks

def _local_search(blocks, v, k, t, all_blocks, max_iter=50, random_seed=None):
    """Try to reduce block count by removing redundant blocks and swapping."""
    if random_seed is not None:
        rng = random.Random(random_seed)
    else:
        rng = random.Random()
    
    blocks = list(blocks)
    all_t_subsets = set(itertools.combinations(range(v), t))
    
    for iteration in range(max_iter):
        improved = False
        
        # Try removing each block if redundant
        for i in range(len(blocks) - 1, -1, -1):
            # Check if block i is redundant
            without_i = blocks[:i] + blocks[i+1:]
            covered = set()
            for b in without_i:
                covered |= set(itertools.combinations(b, t))
            if covered == all_t_subsets:
                blocks.pop(i)
                improved = True
                break
        
        if improved:
            continue
        
        # Try swapping a block with another that covers more uncovered
        # Find which t-subsets are covered by exactly one block
        cover_count = {}
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cover_count[ts] = cover_count.get(ts, 0) + 1
        
        # Try to find a block that can be replaced by one that covers more unique subsets
        for i in range(len(blocks)):
            unique_to_i = set()
            for ts in itertools.combinations(blocks[i], t):
                if cover_count[ts] == 1:
                    unique_to_i.add(ts)
            
            if not unique_to_i:
                continue
            
            # Find a replacement block that covers all of unique_to_i
            for b in all_blocks:
                if b == blocks[i]:
                    continue
                b_cover = set(itertools.combinations(b, t))
                if unique_to_i.issubset(b_cover):
                    # Check if swapping doesn't break coverage
                    new_blocks = blocks[:i] + [b] + blocks[i+1:]
                    new_covered = set()
                    for nb in new_blocks:
                        new_covered |= set(itertools.combinations(nb, t))
                    if new_covered == all_t_subsets:
                        blocks[i] = b
                        improved = True
                        break
            if improved:
                break
        
        if not improved:
            break
    
    return blocks

def construct(v, k, t):
    """Construct a covering design using greedy + local search with multiple restarts."""
    start_time = time.time()
    time_limit = 110  # Leave margin for 120s limit
    
    all_blocks = _generate_all_blocks(v, k)
    all_t_subsets = set(itertools.combinations(range(v), t))
    
    best_blocks = None
    best_len = float('inf')
    
    # Multiple randomized greedy restarts
    num_restarts = 20
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit - 30:
            break
        
        seed = restart * 1000 + 42
        blocks = _greedy_cover(v, k, t, all_blocks, random_seed=seed)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
        
        # Local search on this solution
        if time.time() - start_time > time_limit - 20:
            break
        
        improved = _local_search(blocks, v, k, t, all_blocks, max_iter=30, random_seed=seed)
        if len(improved) < best_len:
            best_len = len(improved)
            best_blocks = improved
    
    # Final local search on best solution
    if best_blocks is not None and time.time() - start_time < time_limit - 5:
        final = _local_search(best_blocks, v, k, t, all_blocks, max_iter=50, random_seed=999)
        if len(final) < best_len:
            best_blocks = final
            best_len = len(final)
    
    if best_blocks is None:
        # Fallback: simple greedy
        best_blocks = _greedy_cover(v, k, t, all_blocks)
    
    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
