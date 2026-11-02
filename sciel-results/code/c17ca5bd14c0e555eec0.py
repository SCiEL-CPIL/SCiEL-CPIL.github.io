import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115  # Safety margin

    # Precompute all t-subsets
    all_ts = list(itertools.combinations(range(v), t))
    
    # Map t-subset to index
    ts_to_idx = {ts: i for i, ts in enumerate(all_ts)}
    n_ts = len(all_ts)
    
    # Precompute for each possible block (k-subset) the set of t-subsets it covers
    # We'll represent this as a bitmask for fast operations
    # n_ts = C(24,3) = 2024, so we need 2024 bits = 253 bytes per mask
    # But Python ints can handle this natively and bit operations are fast
    
    def ts_mask(ts):
        return 1 << ts_to_idx[ts]
    
    # Generate all possible blocks
    all_blocks = list(itertools.combinations(range(v), k))
    
    # Precompute coverage mask for each block
    block_masks = []
    for block in all_blocks:
        mask = 0
        for ts in itertools.combinations(block, t):
            mask |= ts_mask(ts)
        block_masks.append(mask)
    
    # Full mask: all t-subsets need to be covered
    full_mask = (1 << n_ts) - 1
    
    def count_bits(mask):
        return bin(mask).count('1')
    
    def greedy_construct():
        uncovered = full_mask
        blocks = []
        
        while uncovered:
            # Pick block that covers most uncovered t-subsets
            best_idx = -1
            best_gain = -1
            for i, bm in enumerate(block_masks):
                gain = count_bits(uncovered & bm)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1:
                break
            blocks.append(all_blocks[best_idx])
            uncovered &= ~block_masks[best_idx]
        
        return blocks
    
    def random_greedy_construct(seed):
        random.seed(seed)
        uncovered = full_mask
        blocks = []
        
        while uncovered:
            # Pick randomly from top candidates to add diversity
            candidates = []
            for i, bm in enumerate(block_masks):
                gain = count_bits(uncovered & bm)
                if gain > 0:
                    candidates.append((gain, i))
            
            if not candidates:
                break
            
            # Sort by gain descending, pick from top few
            candidates.sort(key=lambda x: -x[0])
            top_n = min(len(candidates), max(1, len(candidates) // 10))
            chosen = random.choice(candidates[:top_n])
            blocks.append(all_blocks[chosen[1]])
            uncovered &= ~block_masks[chosen[1]]
        
        return blocks
    
    def remove_redundant(blocks):
        if not blocks:
            return blocks
        
        # Count coverage for each t-subset
        cover_count = [0] * n_ts
        for block in blocks:
            for ts in itertools.combinations(block, t):
                cover_count[ts_to_idx[ts]] += 1
        
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                block = blocks[i]
                is_redundant = True
                for ts in itertools.combinations(block, t):
                    if cover_count[ts_to_idx[ts]] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for ts in itertools.combinations(block, t):
                        cover_count[ts_to_idx[ts]] -= 1
                    blocks.pop(i)
                    changed = True
                    break
        return blocks
    
    def local_search(blocks, max_time=5.0):
        """Try to reduce number of blocks by swapping/removing."""
        ls_start = time.time()
        n = len(blocks)
        
        # Compute coverage counts
        cover_count = [0] * n_ts
        for block in blocks:
            for ts in itertools.combinations(block, t):
                cover_count[ts_to_idx[ts]] += 1
        
        best_len = n
        best_blocks = list(blocks)
        
        # Try removing one block if possible
        changed = True
        while changed and (time.time() - ls_start) < max_time:
            changed = False
            for i in range(len(best_blocks)):
                # Check if block i is redundant
                redundant = True
                for ts in itertools.combinations(best_blocks[i], t):
                    idx = ts_to_idx[ts]
                    if cover_count[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    for ts in itertools.combinations(best_blocks[i], t):
                        cover_count[ts_to_idx[ts]] -= 1
                    best_blocks.pop(i)
                    best_len = len(best_blocks)
                    changed = True
                    break
        
        # Try pairwise replacement: remove two blocks, add one
        if (time.time() - ls_start) < max_time:
            for i in range(len(best_blocks)):
                if time.time() - ls_start > max_time:
                    break
                for j in range(i + 1, len(best_blocks)):
                    if time.time() - ls_start > max_time:
                        break
                    
                    # Temporarily remove blocks i and j
                    for ts in itertools.combinations(best_blocks[i], t):
                        cover_count[ts_to_idx[ts]] -= 1
                    for ts in itertools.combinations(best_blocks[j], t):
                        cover_count[ts_to_idx[ts]] -= 1
                    
                    # Check what's uncovered
                    uncovered = 0
                    for idx in range(n_ts):
                        if cover_count[idx] == 0:
                            uncovered |= (1 << idx)
                    
                    if uncovered:
                        # Find a block that covers all uncovered
                        # This is expensive, so skip if too many uncovered
                        if count_bits(uncovered) > k:  # A block can cover at most C(7,3)=35
                            pass
                        else:
                            for b_idx, bm in enumerate(block_masks):
                                if (uncovered & ~bm) == 0:
                                    # Block b_idx covers all uncovered
                                    # But we need to make sure it doesn't conflict
                                    new_blocks = [best_blocks[x] for x in range(len(best_blocks)) if x != i and x != j]
                                    new_blocks.append(all_blocks[b_idx])
                                    new_len = len(new_blocks)
                                    if new_len < best_len:
                                        best_blocks = new_blocks
                                        best_len = new_len
                                    break
                    
                    # Restore coverage
                    for ts in itertools.combinations(best_blocks[i], t):
                        cover_count[ts_to_idx[ts]] += 1
                    for ts in itertools.combinations(best_blocks[j], t):
                        cover_count[ts_to_idx[ts]] += 1
        
        return best_blocks
    
    # Main logic
    best_blocks = None
    best_len = float('inf')
    
    # 1. Deterministic Greedy
    blocks = greedy_construct()
    blocks = remove_redundant(blocks)
    best_blocks = blocks
    best_len = len(blocks)
    
    # 2. Try local search on the greedy result
    if time.time() - start_time < time_limit - 10:
        improved = local_search(list(best_blocks), max_time=min(10, time_limit - (time.time() - start_time) - 5))
        improved = remove_redundant(improved)
        if len(improved) < best_len:
            best_blocks = improved
            best_len = len(best_blocks)
    
    # 3. Random restarts
    attempts = 0
    while time.time() - start_time < time_limit - 5:
        seed = random.randint(0, 1000000)
        blocks = random_greedy_construct(seed)
        blocks = remove_redundant(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
        
        # Try local search if it's competitive
        if len(blocks) <= best_len + 3 and time.time() - start_time < time_limit - 10:
            improved = local_search(list(blocks), max_time=min(5, time_limit - (time.time() - start_time) - 5))
            improved = remove_redundant(improved)
            if len(improved) < best_len:
                best_len = len(improved)
                best_blocks = improved
        
        attempts += 1
        if attempts > 500:
            break
    
    return best_blocks
