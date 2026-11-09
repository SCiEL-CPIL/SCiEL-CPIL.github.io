import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    timeout = 140.0  # Safety margin for 150s limit
    
    # Precomputation
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_id = {sub: i for i, sub in enumerate(all_t_subs)}
    all_k_subs = list(itertools.combinations(range(v), k))
    
    # Precompute coverage for each block
    # block -> set of t-subset IDs
    block_to_t_ids = {}
    for block in all_k_subs:
        covered_ids = set()
        for t_sub in itertools.combinations(block, t):
            covered_ids.add(t_sub_to_id[t_sub])
        block_to_t_ids[block] = covered_ids

    def randomized_greedy_pass():
        uncovered = set(range(len(all_t_subs)))
        selected_blocks = []
        
        while uncovered:
            # Calculate gains
            gains = []
            valid_blocks = []
            for block in all_k_subs:
                gain = len(block_to_t_ids[block] & uncovered)
                if gain > 0:
                    gains.append(gain)
                    valid_blocks.append(block)
            
            if not valid_blocks:
                break
                
            # Select block with probability proportional to gain
            total_gain = sum(gains)
            r = random.random() * total_gain
            cumulative = 0
            chosen_block = valid_blocks[-1]
            for i, g in enumerate(gains):
                cumulative += g
                if r <= cumulative:
                    chosen_block = valid_blocks[i]
                    break
            
            selected_blocks.append(chosen_block)
            uncovered.difference_update(block_to_t_ids[chosen_block])
            
            if time.time() - start_time > timeout:
                break
                
        return selected_blocks

    def reduce_blocks_fast(blocks):
        if not blocks:
            return []
        
        current_blocks = list(blocks)
        changed = True
        while changed:
            changed = False
            # Build index: t_sub -> count of blocks covering it (or just list)
            # We only need to know if count > 1 for redundancy check
            t_sub_count = {}
            for block in current_blocks:
                for t_sub in itertools.combinations(block, t):
                    t_sub_count[t_sub] = t_sub_count.get(t_sub, 0) + 1
            
            new_blocks = []
            for block in current_blocks:
                redundant = True
                for t_sub in itertools.combinations(block, t):
                    if t_sub_count[t_sub] == 1:
                        redundant = False
                        break
                if redundant:
                    changed = True
                else:
                    new_blocks.append(block)
            
            current_blocks = new_blocks
            
        return current_blocks

    best_blocks = None
    best_len = float('inf')
    
    while time.time() - start_time < timeout:
        blocks = randomized_greedy_pass()
        if not blocks:
            continue
            
        blocks = reduce_blocks_fast(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    return best_blocks if best_blocks else []
