import random
import itertools
import math
import time

def construct(v, k, t):
    """
    Find a covering design using Simulated Annealing with local search.
    """
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    t_sub_index = {ts: i for i, ts in enumerate(all_t_subs)}
    
    # Precompute for each possible block (k-subset), the list of t-subset indices it covers
    # There are C(19,7) = 50388 blocks.
    # Each block covers C(7,4) = 35 t-subsets.
    # Total memory: 50388 * 35 integers. This is fine.
    all_blocks = list(itertools.combinations(range(v), k))
    block_t_subs = {}
    for b in all_blocks:
        block_t_subs[b] = [t_sub_index[ts] for ts in itertools.combinations(b, t)]
    
    # Block index for quick lookup
    block_index = {b: i for i, b in enumerate(all_blocks)}
    
    # Initial solution: Greedy
    def greedy_init():
        uncovered = set(range(n_t_subs))
        cover_count = [0] * n_t_subs
        blocks = []
        
        while uncovered:
            best_block = None
            best_gain = -1
            # To speed up, we can just pick a random uncovered t-subset and find a block covering it
            # Or iterate through all blocks. 50k blocks * 35 t-subs = 1.7M ops per step.
            # Number of steps ~ 150. Total 255M ops. Might be too slow in Python if done naively.
            # Optimization: Pick a random uncovered t-subset, and find a block covering it that has max gain.
            # Or just pick a random block and check gain.
            
            # Let's use a randomized greedy:
            # Pick a random uncovered t-subset.
            # Find all blocks covering it.
            # Pick the one that covers the most other uncovered t-subsets.
            
            seed_t_idx = random.choice(list(uncovered))
            seed_t = all_t_subs[seed_t_idx]
            
            # Find blocks containing seed_t
            # This is slow to do from scratch.
            # Let's precompute: for each t-subset, which blocks contain it?
            # That's expensive memory: 7752 t-subs * (19 choose 7 / 7 choose 4 ... no)
            # Number of blocks containing a specific t-subset: C(19-4, 7-4) = C(15,3) = 455.
            # So we can precompute `blocks_containing_t[t_idx]`.
            
            # Let's do that precomputation outside greedy_init or in construct.
            pass # Placeholder
            
            # Actually, let's just use the current program's greedy for init, it's proven to work.
            # But we want a "new approach". The SA part is the new approach.
            # The init can be simple.
            
            # Simple random init: Add random blocks until valid.
            # This will be large (maybe 300-400 blocks), but SA will reduce it.
            while uncovered:
                b = random.choice(all_blocks)
                if b in block_index and b not in blocks: # blocks is list of tuples
                    # Check if it covers any uncovered
                    gains = 0
                    for ts_idx in block_t_subs[b]:
                        if ts_idx in uncovered:
                            gains += 1
                    if gains > 0:
                        blocks.append(b)
                        for ts_idx in block_t_subs[b]:
                            if ts_idx in uncovered:
                                uncovered.remove(ts_idx)
                            cover_count[ts_idx] += 1
            return blocks

    # Precompute blocks_containing_t for faster greedy or init
    blocks_containing_t = [[] for _ in range(n_t_subs)]
    for b, ts_idxs in block_t_subs.items():
        for ts_idx in ts_idxs:
            blocks_containing_t[ts_idx].append(b)

    def greedy_init_fast():
        uncovered = set(range(n_t_subs))
        cover_count = [0] * n_t_subs
        blocks = []
        used_blocks = set()
        
        while uncovered:
            # Pick random uncovered t-subset
            t_idx = random.choice(list(uncovered))
            # Pick random block containing it
            candidates = blocks_containing_t[t_idx]
            b = random.choice(candidates)
            
            if b in used_blocks:
                continue
                
            # Check gain
            gain = 0
            for ts_idx in block_t_subs[b]:
                if ts_idx in uncovered:
                    gain += 1
            
            if gain > 0:
                blocks.append(b)
                used_blocks.add(b)
                for ts_idx in block_t_subs[b]:
                    if ts_idx in uncovered:
                        uncovered.remove(ts_idx)
                    cover_count[ts_idx] += 1
        return blocks

    # Local Search: Remove redundant blocks
    def local_search(blocks, cover_count):
        # cover_count is list of int
        # blocks is list of tuples
        # We can remove a block if for all t in block, cover_count[t] > 1
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                b = blocks[i]
                redundant = True
                for ts_idx in block_t_subs[b]:
                    if cover_count[ts_idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block i
                    for ts_idx in block_t_subs[b]:
                        cover_count[ts_idx] -= 1
                    blocks.pop(i)
                    changed = True
        return blocks, cover_count

    # Initialize
    blocks = greedy_init_fast()
    cover_count = [0] * n_t_subs
    for b in blocks:
        for ts_idx in block_t_subs[b]:
            cover_count[ts_idx] += 1
    
    # Ensure it's valid and minimal
    blocks, cover_count = local_search(blocks, cover_count)
    
    best_blocks = list(blocks)
    best_len = len(best_blocks)
    
    # Simulated Annealing
    start_time = time.time()
    T = 10.0
    cooling = 0.99
    min_T = 0.01
    
    # Random seed
    random.seed(42)
    
    while T > min_T and (time.time() - start_time) < 110:
        # Propose move
        move_type = random.random()
        new_blocks = list(blocks)
        new_cover = list(cover_count)
        
        if move_type < 0.4:
            # Remove random block
            if len(new_blocks) > 0:
                idx = random.randint(0, len(new_blocks) - 1)
                b = new_blocks[idx]
                for ts_idx in block_t_subs[b]:
                    new_cover[ts_idx] -= 1
                new_blocks.pop(idx)
        elif move_type < 0.8:
            # Add random block
            b = random.choice(all_blocks)
            # Check if already in blocks
            # Using a set for blocks would be faster, but we have list.
            # For now, just add. Duplicates are handled by local_search or cost.
            # Actually, duplicates are bad. Let's check.
            # To keep it simple, we assume we add a block not in the list.
            # Finding if it's in list is O(N). N is ~150. Fine.
            if b not in new_blocks:
                for ts_idx in block_t_subs[b]:
                    new_cover[ts_idx] += 1
                new_blocks.append(b)
        else:
            # Swap
            if len(new_blocks) > 0:
                idx = random.randint(0, len(new_blocks) - 1)
                old_b = new_blocks[idx]
                new_b = random.choice(all_blocks)
                if new_b != old_b:
                    # Remove old
                    for ts_idx in block_t_subs[old_b]:
                        new_cover[ts_idx] -= 1
                    # Add new
                    for ts_idx in block_t_subs[new_b]:
                        new_cover[ts_idx] += 1
                    new_blocks[idx] = new_b
        
        # Calculate cost
        # Cost = len(blocks) if valid else len(blocks) + 1000 * uncovered
        uncovered = 0
        for c in new_cover:
            if c == 0:
                uncovered += 1
        
        if uncovered == 0:
            new_cost = len(new_blocks)
        else:
            new_cost = len(new_blocks) + 1000 * uncovered
        
        # Current cost
        if cover_count[min(range(n_t_subs))] == 0: # Check valid
             cur_uncovered = sum(1 for c in cover_count if c == 0)
        else:
             cur_uncovered = 0
        if cur_uncovered == 0:
            cur_cost = len(blocks)
        else:
            cur_cost = len(blocks) + 1000 * cur_uncovered
            
        delta = new_cost - cur_cost
        
        # Accept
        if delta < 0 or random.random() < math.exp(-delta / T):
            blocks = new_blocks
            cover_count = new_cover
            
            # Local Search to clean up
            blocks, cover_count = local_search(blocks, cover_count)
            
            # Update best
            # Check if valid
            if min(cover_count) > 0:
                if len(blocks) < best_len:
                    best_len = len(blocks)
                    best_blocks = list(blocks)
        
        T *= cooling

    return [list(b) for b in best_blocks]
