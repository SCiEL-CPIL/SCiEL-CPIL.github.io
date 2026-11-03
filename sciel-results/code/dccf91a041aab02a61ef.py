import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Precompute all t-subsets as bitmasks
    all_subsets = []
    for comb in itertools.combinations(range(v), t):
        mask = 0
        for i in comb:
            mask |= (1 << i)
        all_subsets.append(mask)
        
    # Precompute subsets_by_point: list of (subset_mask, subset_mask_without_point)
    subsets_by_point = [[] for _ in range(v)]
    for mask in all_subsets:
        m = mask
        while m:
            lsb = m & -m
            x = lsb.bit_length() - 1
            subsets_by_point[x].append((mask, mask ^ lsb))
            m ^= lsb
            
    def greedy_pass(uncovered_initial, use_random=False, seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
            
        uncovered = set(uncovered_initial)
        blocks = []
        
        while uncovered:
            if use_random:
                # Efficient random choice from set
                # Converting to list is O(N)
                start_mask = rng.choice(list(uncovered))
            else:
                start_mask = min(uncovered)
                
            block_mask = start_mask
            uncovered.discard(start_mask)
            
            while block_mask.bit_count() < k:
                best_gain = -1
                best_x = -1
                
                for x in range(v):
                    if block_mask & (1 << x):
                        continue
                    
                    gain = 0
                    for S, S_wo_x in subsets_by_point[x]:
                        if S in uncovered:
                            if (S_wo_x & ~block_mask) == 0:
                                gain += 1
                                
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain and gain > 0:
                        if use_random:
                            if rng.random() < 0.5:
                                best_x = x
                        else:
                            if x < best_x:
                                best_x = x
            
            if best_x == -1:
                candidates = [x for x in range(v) if not (block_mask & (1 << x))]
                if use_random:
                    best_x = rng.choice(candidates)
                else:
                    best_x = min(candidates)
            
            block_mask |= (1 << best_x)
            
            for S, S_wo_x in subsets_by_point[best_x]:
                if S in uncovered:
                    if (S & ~block_mask) == 0:
                        uncovered.discard(S)
                        
            block_list = []
            m = block_mask
            while m:
                lsb = m & -m
                block_list.append(lsb.bit_length() - 1)
                m ^= lsb
            block_list.sort()
            blocks.append(block_list)
            
        return blocks

    best_blocks = greedy_pass(all_subsets, use_random=False)
    best_len = len(best_blocks)
    
    restart_count = 0
    while time.time() < deadline:
        curr_blocks = greedy_pass(all_subsets, use_random=True)
        restart_count += 1
        if len(curr_blocks) < best_len:
            best_len = len(curr_blocks)
            best_blocks = curr_blocks
            if best_len <= 135:
                break

    return best_blocks
