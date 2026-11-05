import itertools
import time
import random

def construct(v, k, t):
    # Parameters for C(19, 8, 5)
    if v != 19 or k != 8 or t != 5:
        # Fallback for other parameters if necessary, though task specifies this
        return greedy_general(v, k, t)
        
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # 1. Pre-computation
    # Map all 5-subsets to unique indices
    points = range(v)
    all_t_subsets = list(itertools.combinations(points, t))
    sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    n_t_subsets = len(all_t_subsets)
    
    # Pre-compute masks for all k-subsets (blocks)
    # Each mask is an integer where bit i is 1 if the i-th t-subset is contained in the block
    all_blocks = list(itertools.combinations(points, k))
    block_masks = []
    for block in all_blocks:
        mask = 0
        # Iterate over all t-subsets of this block
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks.append((block, mask))
        
    # Sort blocks? Not strictly necessary, but might help cache? No.
    
    def run_greedy(seed):
        random.seed(seed)
        uncovered = (1 << n_t_subsets) - 1
        blocks_found = []
        
        while uncovered:
            best_gain = 0
            best_blocks = []
            
            # Iterate through all candidate blocks
            for block, mask in block_masks:
                # Calculate how many new t-subsets this block covers
                gain = (mask & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_blocks = [(block, mask)]
                elif gain == best_gain and gain > 0:
                    best_blocks.append((block, mask))
                    
            if not best_blocks:
                break # Should not happen if uncovered is not 0 and we have enough blocks
                
            # Randomly select from best blocks to escape local minima
            chosen_block, chosen_mask = random.choice(best_blocks)
            blocks_found.append(list(chosen_block))
            uncovered &= ~chosen_mask
            
        return blocks_found

    # Run initial greedy
    best_blocks = run_greedy(0)
    best_count = len(best_blocks)
    
    # Run more greedy attempts if time permits
    # Estimate time per run: ~15-20s?
    # If it's too slow, we only get 1-2 runs.
    # Let's check time after first run.
    t1 = time.time()
    run_time = t1 - start_time
    
    # If first run took > 100s, stop.
    # If it took < 50s, we can try more.
    # The loop inside run_greedy is the bottleneck.
    # 75k blocks * ~350 iterations = 26M ops.
    # Python 26M ops might take 10-20s.
    
    seed = 1
    while time.time() < deadline - 20: # Keep 20s safety
        blocks = run_greedy(seed)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
        seed += 1
        
    return best_blocks

def greedy_general(v, k, t):
    # Fallback for other parameters, simple greedy
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
