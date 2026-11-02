import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0

    if not (v == 19 and k == 8 and t == 5):
        return _generic_greedy(v, k, t)

    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    all_5_subsets = list(itertools.combinations(points, t))
    
    # Precompute sets of 5-subsets for each block
    block_subsets_set = []
    for b in all_blocks:
        subs = set(itertools.combinations(b, t))
        block_subsets_set.append(subs)
        
    block_id_to_block = {i: b for i, b in enumerate(all_blocks)}

    def greedy_once(seed_val):
        random.seed(seed_val)
        uncovered = set(all_5_subsets)
        blocks_used = []
        
        while uncovered:
            if time.time() > deadline:
                return None
            
            best_gain = -1
            best_tie_list = []
            
            for b_idx in range(num_blocks):
                gain = len(block_subsets_set[b_idx] & uncovered)
                
                if gain > best_gain:
                    best_gain = gain
                    best_tie_list = [b_idx]
                elif gain == best_gain:
                    best_tie_list.append(b_idx)
            
            if best_gain == 0:
                break
            
            chosen_idx = random.choice(best_tie_list)
            blocks_used.append(chosen_idx)
            uncovered -= block_subsets_set[chosen_idx]
        
        # Polish: Remove redundant blocks
        # We need cover counts now.
        # Rebuild cover counts from blocks_used
        cover_count = {s: 0 for s in all_5_subsets}
        for b_idx in blocks_used:
            for s in block_subsets_set[b_idx]:
                cover_count[s] += 1
                
        current_blocks = list(blocks_used)
        
        while True:
            if time.time() > deadline:
                break
            
            removed_any = False
            for i in range(len(current_blocks)):
                b_idx = current_blocks[i]
                is_redundant = True
                for s in block_subsets_set[b_idx]:
                    if cover_count[s] <= 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    for s in block_subsets_set[b_idx]:
                        cover_count[s] -= 1
                    current_blocks.pop(i)
                    removed_any = True
                    break
            
            if not removed_any:
                break
        
        return [block_id_to_block[i] for i in current_blocks]

    best_blocks = None
    best_len = float('inf')
    
    for i in range(50): # Try more iterations, set intersection is fast
        if time.time() > deadline:
            break
        res = greedy_once(i)
        if res is not None:
            if len(res) < best_len:
                best_len = len(res)
                best_blocks = res
    
    if best_blocks is None:
        return _generic_greedy(v, k, t)
        
    return best_blocks

def _generic_greedy(v, k, t):
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        block = list(min(uncovered))
        while len(block) < k:
            def gain(x):
                return sum(1 for rest in itertools.combinations(block, t - 1)
                           if tuple(sorted(rest + (x,))) in uncovered)
            block.append(max((x for x in range(v) if x not in block), key=gain))
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    return blocks
