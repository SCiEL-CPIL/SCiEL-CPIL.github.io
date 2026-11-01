import itertools
import time
import random

def construct(v, k, t):
    deadline = time.time() + 115
    
    # Precompute all t-subsets
    all_tsets = list(itertools.combinations(range(v), t))
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}
    n_tsets = len(all_tsets)
    
    # Cache for block -> list of tset indices
    block_to_tsets = {}
    
    def get_block_tsets(block):
        """Return list of tset indices covered by this block (tuple)."""
        if block in block_to_tsets:
            return block_to_tsets[block]
        
        covered = []
        for ts in itertools.combinations(block, t):
            idx = tset_to_idx[ts]
            covered.append(idx)
        
        block_to_tsets[block] = covered
        return covered

    def greedy_construct(seed):
        random.seed(seed)
        coverage = [0] * n_tsets
        blocks = []
        uncovered_count = n_tsets
        
        while uncovered_count > 0 and time.time() < deadline:
            # Find an uncovered tset
            # To avoid O(N) scan every time, we could maintain a list, but N=2024 is small.
            candidates = [i for i in range(n_tsets) if coverage[i] == 0]
            if not candidates:
                break
            target_idx = random.choice(candidates)
            target_tset = all_tsets[target_idx]
            
            # Build a block greedily around target_tset
            block_points = list(target_tset)
            
            while len(block_points) < k and time.time() < deadline:
                partial_set = set(block_points)
                best_x = -1
                best_gain = -1
                
                for x in range(v):
                    if x in partial_set:
                        continue
                    
                    gain = 0
                    for R in itertools.combinations(block_points, t - 1):
                        s_tuple = tuple(sorted(list(R) + [x]))
                        idx = tset_to_idx[s_tuple]
                        if coverage[idx] == 0:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                
                if best_x == -1 or best_gain == 0:
                    remaining = [x for x in range(v) if x not in partial_set]
                    if remaining:
                        best_x = random.choice(remaining)
                    else:
                        break
                block_points.append(best_x)
            
            block = tuple(sorted(block_points))
            blocks.append(block)
            covered_tsets = get_block_tsets(block)
            for idx in covered_tsets:
                if coverage[idx] == 0:
                    uncovered_count -= 1
                coverage[idx] += 1
        
        return blocks, coverage

    def local_search(blocks, coverage, seed):
        random.seed(seed)
        
        # Precompute tsets for each block
        block_tsets = [get_block_tsets(b) for b in blocks]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Pass 1: Remove redundant blocks
            i = 0
            while i < len(blocks):
                redundant = True
                for idx in block_tsets[i]:
                    if coverage[idx] < 2:
                        redundant = False
                        break
                
                if redundant:
                    for idx in block_tsets[i]:
                        coverage[idx] -= 1
                    blocks.pop(i)
                    block_tsets.pop(i)
                    improved = True
                else:
                    i += 1
            
            # Pass 2: Try to swap points
            if not improved and blocks:
                # Try a few random swaps
                # To save time, we don't try all blocks, just a random subset
                num_swaps_to_try = min(10, len(blocks))
                for _ in range(num_swaps_to_try):
                    if time.time() >= deadline:
                        break
                    i = random.randint(0, len(blocks) - 1)
                    b = blocks[i]
                    b_set = set(b)
                    
                    unique_tsets = [idx for idx in block_tsets[i] if coverage[idx] == 1]
                    
                    if not unique_tsets:
                        # This block is not critical, any swap is safe regarding uniqueness
                        # But we want to increase redundancy
                        pass
                    
                    best_swap = None
                    best_score = -1
                    
                    # Try swapping points
                    for p in b:
                        for q in range(v):
                            if q in b_set:
                                continue
                            
                            new_b_list = list(b)
                            new_b_list.remove(p)
                            new_b_list.append(q)
                            new_b = tuple(sorted(new_b_list))
                            
                            new_tsets = get_block_tsets(new_b)
                            new_tset_set = set(new_tsets)
                            
                            # Check if all unique_tsets are still covered
                            valid = True
                            for idx in unique_tsets:
                                if idx not in new_tset_set:
                                    valid = False
                                    break
                            
                            if not valid:
                                continue
                            
                            # Score: increase in coverage count >= 2
                            old_score = sum(1 for idx in block_tsets[i] if coverage[idx] >= 2)
                            new_score = sum(1 for idx in new_tsets if coverage[idx] >= 2)
                            
                            if new_score > old_score:
                                if new_score > best_score:
                                    best_score = new_score
                                    best_swap = (p, q, new_b, new_tsets)
                    
                    if best_swap:
                        p, q, new_b, new_tsets = best_swap
                        for idx in block_tsets[i]:
                            coverage[idx] -= 1
                        for idx in new_tsets:
                            coverage[idx] += 1
                        
                        blocks[i] = new_b
                        block_tsets[i] = new_tsets
                        improved = True
                        break # Restart pass after a successful swap? Or continue?
                        # Breaking the inner loop, but the outer 'improved' loop will restart.
        
        return blocks, coverage

    best_blocks = None
    best_len = float('inf')
    
    num_attempts = 0
    start_time = time.time()
    
    while time.time() < deadline - 5:
        seed = num_attempts
        blocks, coverage = greedy_construct(seed)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [list(b) for b in blocks]
        
        blocks, coverage = local_search(blocks, coverage, seed)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [list(b) for b in blocks]
        
        num_attempts += 1
        
        if best_len <= 76:
            if time.time() > start_time + 20:
                break

    if best_blocks is None:
        blocks, _ = greedy_construct(0)
        return [list(b) for b in blocks]
        
    return best_blocks
